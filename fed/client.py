import os
import multiprocessing as MP
from typing import (
    Dict, Tuple, List,
    Callable
)
import flwr as fl
from flwr.common import (
    Scalar,
    FitIns, FitRes,
    Status,
    Code,
    GetParametersRes
)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

import numpy as np
import torchvision
import copy

from .communicate import serialize, deserialize
from .utils.bind import register_method_hook
from .utils._internal import _zippable, _iter_dict
from .config import STATE, LossMode

from reconstruct.mmvae import DecoupledMMVAE
import random
'''
    Try FedBYOL on Cifar10
'''

class Client(fl.client.Client):

    def __init__(self, 
                cid: int,
                mod_state: STATE,
                model: nn.ModuleDict,
                loss_fn: Callable,
                trainloader, testloader=None,
                mmvae_config=None,
                device='cuda',
                online_augment=False,
                augmentation=None,
                state_dir='.client'):
        super(Client, self).__init__()

        self.cid = cid
        self.model = model

        self.mod_state = mod_state
        self.device = device

        self.trainloader, self.testloader = trainloader, testloader
        self.num_examples = len(self.trainloader.dataset)

        # self.state_dir = state_dir
        # if not os.path.isdir(state_dir):
        #     os.makedirs(state_dir, exist_ok=True)
        
        self.loss_fn = loss_fn

        self.loss_mode : LossMode = LossMode.SelfContra 
        self.eta: float = 1.0

        self.mmvae_config = mmvae_config
        self.mmvae: DecoupledMMVAE = None
        if self.mod_state != STATE.BOTH:
            self.mmvae = self._load_mmvae()
        
        self.online_augment = online_augment
        self.augmentation = augmentation
        if online_augment:
            assert augmentation is not None
        else:
            if augmentation is not None:
                self.online_augment = True
                

    def _load_mmvae(self):
        mmave_param_list = {
            'encoders' : self.mmvae_config['encoders']().to(self.device),
            'decoders' : self.mmvae_config['decoders']().to(self.device),
            'latent_dim' : self.mmvae_config['latent_dim'],
            'score_fns' : self.mmvae_config['score_fns']().to(self.device)
        }
        mmvae = DecoupledMMVAE(**mmave_param_list, device=self.device)
        return mmvae

    @property
    def _available_mod(self):
        mod_keys = []
        if self.mod_state == STATE.AUDIO:
            mod_keys.append('audio')
        elif self.mod_state == STATE.IMAGE:
            mod_keys.append('image')
        else:
            mod_keys.append('audio')
            mod_keys.append('image')
        return mod_keys


    def get_parameters(self, config: Dict[str, Scalar]):
        payload = self._export_payload()
        status = Status(code=Code.OK, message='Success')
        return GetParametersRes(
            status=status,
            parameters=payload
        )

    def _export_payload(self):
        payload = {}
        for k in self._available_mod:
            payload[k] = self.model[k].state_dict()

        return serialize(payload)

    def fit(self, ins: FitIns) -> FitRes:
        # load parameters 
        self.set_parameters(ins)
        epoch = ins.config['local_epoch']
        optimizer = self._set_optimizer(ins)

        loss = self._train_impl(
            epoch,
            self.model,
            optimizer
        )
        status = Status(code=Code.OK, message='Success')
        payload = self._export_payload()

        metrics = {
            'cid' : int(self.cid),
            'state' : self.mod_state.value
        }

        for i, v in enumerate(loss):
            metrics[f'epoch_{i}'] = v

        fit_res = FitRes(
            parameters=payload,
            status=status,
            num_examples=self.num_examples,
            metrics=metrics
        )
        return fit_res
    
    def set_parameters(self, fit_ins: FitIns):
        ckp = deserialize(fit_ins.parameters)
        zipped_model_dict = zip(
            _iter_dict(self.model),
            _iter_dict(ckp, exclude_list=['mmvae'])
        )
        for model, state_dict in zipped_model_dict:
            model.load_state_dict(state_dict)
        if self.mod_state is not STATE.BOTH:
            self.mmvae.load_state_dict(ckp['mmvae'])
    
    def to(self, device):
        self._device = device
        components = [self.model]
        if self.online_augment:
            components.append(self.augmentation)
        if self.mmvae is not None:
            components.append(self.mmvae)
        
        for x in components:
            x = x.to(device)
        return self
    
    def _set_optimizer(self, ins: FitIns):
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=ins.config['lr'],
            weight_decay=ins.config['weight_decay']
        )
        return optimizer

    def _train_impl(self, epoch,
                    model, optimizer, 
                    scheduler=None
                ):
        '''
            behavior of self.trainloader:
            if use cross-mod-loss:
                if both-mod-available:
                    (mod_a, mod_b)
                else:
                    (mod_a, )
            else: (hybrid-contra or self-contra)
                (mod_view1, mod_view2)
                if hyrid-contra:
                    when computing cross-mod loss:
                        random sample one of the views
        '''

        round_loss = []
        for _ in range(epoch):
            running_loss = .0
            for inputs in self.trainloader:
                loss = self._fwd_branching(
                    _iter_dict(model),
                    _zippable(inputs),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if scheduler is not None:
                scheduler.step()
            
            avg_loss = running_loss / len(self.trainloader)
            round_loss.append(avg_loss)

        return round_loss
    
    @property
    def _mod_iterator(self):
        return _iter_dict(self.model)
     
    @torch.no_grad()
    def _generate_missing_mod(self, embed: torch.Tensor) -> torch.Tensor:
        if self.mod_state == STATE.AUDIO:
            generated_embed = self.mmvae.reconstruct({'audio' : embed})['image']
        elif self.mod_state == STATE.IMAGE:
            generated_embed = self.mmvae.reconstruct({'image' : embed})['audio']
        return generated_embed.detach()
    


    def _fwd_branching(self, inputs: Tuple[torch.Tensor]):
        '''
            If there is modality missing,
            generate the embedding conditioned on existing modality
            using the mmvae model distributed by the server in this round
        '''
        loss = .0
        if self.loss_mode == LossMode.CrossContra:
            loss = self._compute_cross_mod_loss(inputs)
            # if do self-contrastive learning
        elif self.loss_mode == LossMode.SelfContra:
            loss = self._compute_self_mod_loss(inputs)
        else: # self.loss_mode == HybridContra
            selfcontra_loss = self._compute_self_mod_loss(inputs)
            crosscontra_loss = self._compute_cross_mod_loss(
                (random.choice(inputs), )
            )
            loss = selfcontra_loss + self.eta * crosscontra_loss
        return loss
    
    def _compute_cross_mod_loss(self, inputs: Tuple[torch.Tensor]):
        out = []
        for (model, x) in zip(self._mod_iterator, inputs):
            out.append(model(x))
        
        if self.mod_state != STATE.BOTH:
            possessed_embed = out[0]
            generated_embed = self._generate_missing_mod(possessed_embed)

            mod_x = possessed_embed; mod_y = generated_embed
        else:
            mod_x, mod_y = out

        loss = self.loss_fn(mod_x, mod_y)
        return loss

    def _compute_self_mod_loss(self, inputs: Tuple[torch.Tensor]):
        single_mod_network = next(self._mod_iterator)
        if self.online_augment:
            view1, view2 = self.augmentation(inputs[0])
        else:
            view1, view2 = inputs
        
        multi_view = [view1, view2]
        out = []
        for view in multi_view:
            out.append(single_mod_network(view))

        view1_embed, view2_embed = out
        loss = self.loss_fn(view1_embed, view2_embed)
        return loss

    def _wrap_inputs(self, inputs: Tuple[torch.Tensor]):
        wrapped_inputs = {}
        for mod, x in zip(self._available_mod, inputs):
            wrapped_inputs[mod] = x.to(self.device)
        return wrapped_inputs

        
            
            

