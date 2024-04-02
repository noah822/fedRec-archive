import os
import multiprocessing as MP
from typing import (
    Dict, Tuple, List,
    Callable,
    Iterable,
    Union
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
from torch.utils.data import DataLoader
import h5py

import numpy as np
import torchvision
import copy


import fed.aggregate as aggregate
from utils.train import CustomBatchSampler
from .communicate import serialize, deserialize
from .utils.bind import register_method_hook
from .utils._internal import _zippable, _iter_dict, _to_numpy
from .utils._internal import pairwise_sim_metrics
from .config import STATE, LossMode

from reconstruct.mmvae import DecoupledMMVAE
from model.simclr import StandardPipeline
import random


class MultiModClient(fl.client.Client):
    def __init__(self,
                 cid: int,
                 mod_state: STATE,
                 model_dict: nn.ModuleDict):
        super(MultiModClient, self).__init__()
        self.mod_state = mod_state
        self.model_dict = model_dict
        self.cid = cid
    
    def load_model(self, ckp: Iterable):
        for model, state_dict in zip(self.mod_iterator, ckp):
            model.load_state_dict(state_dict)
    
    @property
    def mod_names(self):
        names = list(self.model_dict.keys())
        return sorted(names)
    @property
    def mod_iterator(self):
        return _iter_dict(self.model_dict)

class Client(fl.client.Client):

    def __init__(self, 
                cid: int,
                mod_state: STATE,
                loss_mode: LossMode,
                model: nn.ModuleDict,
                loss_fn: Callable,
                trainloader,
                testloader=None,
                negative_aug_ratio: float=None,
                mmvae_config=None,
                device='cuda',
                augmentation: Callable=None,
                hparam_list: List[str]=None
            ):
        super(Client, self).__init__()

        self.cid = cid
        self.model = model

        self.mod_state = mod_state

        self.device = device

        self.trainloader, self.testloader = trainloader, testloader
        self.num_examples = len(self.trainloader.dataset)
        
        self.loss_fn = loss_fn

        self.loss_mode : LossMode = loss_mode

        # penalize hybrid-contra loss
        self.eta: float = 1.

        # penalize generated negative logits
        self.alpha: float = .5


        self._gen_negative = negative_aug_ratio is not None
        if self._gen_negative:
            assert mmvae_config is not None

        self._nega_aug_ratio: float = negative_aug_ratio
        self._overriding_nega_aug_ratio: float = None

        self.mmvae_config = mmvae_config
        self.mmvae: DecoupledMMVAE = None
        self._use_mmvae = self.mod_state != STATE.BOTH \
                    and mmvae_config is not None
        
        if self.mod_state == STATE.BOTH:
            if self._gen_negative:
                self._use_mmvae = True
            else:
                self._use_mmvae = False
        else:
            if mmvae_config is not None:
                self._use_mmvae = True


        if self._use_mmvae:
            self.mmvae = self._load_mmvae()
        
        self._online_augment = augmentation is not None
        self.augmentation = augmentation

        self.hparam_list = hparam_list

    @property
    def negative_aug_ratio(self):
        if self._overriding_nega_aug_ratio is not None:
            return self._overriding_nega_aug_ratio
        elif self._nega_aug_ratio is not None:
            return self._nega_aug_ratio
        else:
            return None

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
    

    def config_hparams(self, ins: FitIns):
        if self.hparam_list is not None:
            for hparam_name in self.hparam_list:
                setattr(self, hparam_name, ins.config[hparam_name])

    def fit(self, ins: FitIns) -> FitRes:
        # load parameters 
        self.set_parameters(ins)
        epoch = ins.config['local_epoch']
        optimizer = self._set_optimizer(ins)

        self.config_hparams(ins)

        loss = self._train_impl(
            epoch,
            self.model,
            optimizer,
            ins=ins
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
        exclude_list = ['mmvae']

        if self.mod_state == STATE.AUDIO:
            exclude_list.append('image')
        elif self.mod_state == STATE.IMAGE:
            exclude_list.append('audio')
        else:
            pass
    
        zipped_model_dict = zip(
            _iter_dict(self.model),
            _iter_dict(ckp, exclude_list=exclude_list)
        )
        for model, state_dict in zipped_model_dict:
            model.load_state_dict(state_dict)
        if self._use_mmvae:
            self.mmvae.load_state_dict(ckp['mmvae'])
        
    
    def to(self, device):
        self._device = device
        components = [self.model]
        if self._online_augment:
            components.append(self.augmentation)
        if self.mmvae is not None:
            components.append(self.mmvae)
        
        for x in components:
            x = x.to(device)
        return self
    
    def _set_optimizer(self, ins: FitIns):
        # set modality specific sub-network learning rate
        param_group = []
        for mod, sub_model in self.model.items():
            param_group.append({
                'params' : sub_model.parameters(),
                'lr' : ins.config['lr'][mod]
            })
        optimizer = optim.SGD(
            param_group,
            lr=ins.config['lr']['base'],
            weight_decay=ins.config['weight_decay']
        )
        return optimizer

    def _train_impl(self, epoch,
                    model, optimizer, 
                    scheduler=None,
                    ins: FitIns=None
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
                loss = self._fwd_branching(_zippable(inputs))
                
                if self._fedprox_like:
                    assert 'omega' in self.hparam_list, \
                    'when using fedprox l2 penalty, omega should be specified'
                    assert ins is not None, 'fit ins should be propagated'
                    fedprox_loss = self.add_fedprox_penalty(ins)
                    loss += self.omega * fedprox_loss

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
    


    def _fwd_branching(self, inputs: Tuple[torch.Tensor], **kwargs):
        '''
            If there is modality missing,
            generate the embedding conditioned on existing modality
            using the mmvae model distributed by the server in this round
        '''
        loss = .0
        if self.loss_mode == LossMode.CrossContra:
            loss = self._compute_cross_mod_loss(self.assemble_mod_iterator('cross'), inputs, **kwargs)
            # if do self-contrastive learning
        elif self.loss_mode == LossMode.SelfContra:
            loss = self._compute_self_mod_loss(self.assemble_mod_iterator('self'), inputs, **kwargs)
        else: # self.loss_mode == HybridContra
            selfcontra_loss = self._compute_self_mod_loss(self.assemble_mod_iterator('self'), inputs, **kwargs)
            crosscontra_loss = self._compute_cross_mod_loss(
                self.assemble_mod_iterator('cross'), 
                (random.choice(inputs), ),
                **kwargs
            )
            # loss = (1 - self.eta) * selfcontra_loss + self.eta * crosscontra_loss
            loss = self.eta * selfcontra_loss + (1 - self.eta) * crosscontra_loss
        return loss
    
    def _compute_cross_mod_loss(self,
                                mod_iterator: List[Callable],
                                inputs: Tuple[torch.Tensor],
                                **kwargs):
        out = []
        for (model, x) in zip(mod_iterator, inputs):
            x = x.to(self.device)
            out.append(model(x))
        
        if self.mod_state != STATE.BOTH:
            possessed_embed = out[0]
            generated_embed = self._generate_missing_mod(possessed_embed, **kwargs)

            mod_x = possessed_embed; mod_y = generated_embed
        else:
            mod_x, mod_y = out

        if self._gen_negative:
            owned_mod = 'all'
            batch_size = mod_x.shape[0]
            num_sample = int(batch_size * self.negative_aug_ratio)
            
            # current order: audio, image
            aug1, aug2 = self._random_draw_embed(owned_mod, num_sample)
            loss = self.loss_fn(
                mod_x, mod_y,
                aug1, aug2,
                alpha=self.alpha,
                cross_view_only=True
            )
        else:
            loss = self.loss_fn(mod_x, mod_y, cross_view_only=True)
        return loss

    def _compute_self_mod_loss(self,
                               mod_iterator: List[Callable], 
                               inputs: Tuple[torch.Tensor, torch.Tensor],
                               **kwargs):
        single_mod_network = next(iter(mod_iterator))
        if self._online_augment:
            view1, view2 = self.augmentation(inputs[0].to(self.device))
        else:
            view1, view2 = inputs
        
        view1 = view1.to(self.device)
        view2 = view2.to(self.device)
        
        multi_view = [view1, view2]
        out = []
        for view in multi_view:
            out.append(single_mod_network(view))

        view1_embed, view2_embed = out
        if self._gen_negative:   # deprecated
            # generate negative_aug_ratio * batch_size [round down] negative samples
            batch_size = view1_embed.shape[0]
            num_sample = int(batch_size * self.negative_aug_ratio)
            
            owned_mod = self._available_mod[0]
            negative_aug = self._random_draw_embed(owned_mod, num_sample)
            loss = self.loss_fn['self-contra'](
                view1_embed,
                view2_embed,
                negative_aug,
                alpha=self.alpha
            )
        else:
            loss = self.loss_fn['self-contra'](view1_embed, view2_embed)
        return loss
    
    def _random_draw_embed(self,
                           mod_name: str,
                           num_sample: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        # support generated paired multi-modal embeds
        assert mod_name in (self._available_mod + ['all'])
        generated = self.mmvae.generate(num_sample, self.device)

        if mod_name != 'all':
            return generated[mod_name]
        else:
            return list(_iter_dict(generated))
    
    def _wrap_inputs(self, inputs: Tuple[torch.Tensor]):
        wrapped_inputs = {}
        for mod, x in zip(self._available_mod, inputs):
            wrapped_inputs[mod] = x.to(self.device)
        return wrapped_inputs
    

# single modal linear probing 
# paralleled multi-modal backbone can be decoupled
class LinearProbeClient(fl.client.Client):
    def __init__(self,
                 cid: str,
                 pretrained_backbone: nn.Module,
                 head: nn.Module,
                 loss_fn: nn.Module,
                 trainloader: DataLoader,
                 testloader: DataLoader=None,
                 device: str='cuda'):

        self.cid = cid
        self.backbone, self.head = pretrained_backbone, head

        self.trainloader = trainloader
        self.num_examples = len(trainloader.dataset)

        self.testloader = testloader

        self.device = device
        self.loss_fn = loss_fn
    
    # only head is aggregated
    # keep backbone intact
    def _set_parameters(self, fit_ins: FitIns):
        state_dict = deserialize(fit_ins.parameters)
        self.head.load_state_dict(state_dict)

    def _set_optimizer(self, model: nn.Module, ins: FitIns):
        optimizer = optim.SGD(
            model.parameters(),
            lr=ins.config['lr'],
            weight_decay=ins.config['weight_decay']
        )
        return optimizer
    
    def fit(self, ins: FitIns) -> FitRes:

        # set parameters of probing head
        self._set_parameters(ins)
        
        epoch = ins.config['local_epoch']
        optimizer = self._set_optimizer(self.head, ins)
        

        # load pretrained backbones
        loss: Dict[str, List] = \
            self._probe_impl(
                self.backbone,
                self.head,
                optimizer,
                self.loss_fn,
                epoch,
                self.trainloader
            )

        status = Status(code=Code.OK, message='Success')
        payload = self._export_payload()

        metrics = {
            'cid' : int(self.cid),
        }

        # upload local loss to server
        for i, v in enumerate(loss):
            metrics[f'epoch_{i}'] = v

        fit_res = FitRes(
            parameters=payload,
            status=status,
            num_examples=self.num_examples,
            metrics=metrics
        )
        return fit_res

    def to(self, device):
        components = [self.backbone, self.head]
        for module in components:
            module = module.to(device)
        return self

    def _probe_impl(self,
                    backbone: nn.Module,
                    head: nn.Module,
                    optimizer,
                    criterion: Callable,
                    num_epoch: int,
                    dataloader):
        # toggle training states of feature extractor, i.e backbone
        backbone.eval()
        round_loss = []

        for _ in range(num_epoch):
            for inputs in dataloader:
                x, label = inputs
                x = x.to(self.device); label = label.to(self.device)

                with torch.no_grad():
                    feature = backbone(x)
                pred = head(feature)
                
                loss = criterion(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                round_loss.append(loss.item())
        
        return round_loss

    def _export_payload(self):
        payload = self.head.state_dict()
        return serialize(payload)
    

class fedRecClient(Client):
    def __init__(self,
                cid: int,
                mod_state: STATE,
                loss_mode: LossMode,
                model: nn.ModuleDict,
                loss_fn: Callable,
                trainloader,
                recnet_type: str='MLP',
                rec_penalty: float=10,
                testloader=None,
                device='cuda',
                sim_repel_fraction: float=None,
                aggregatable_components: List[str]=None,
                loadable_components: List[str]=None,
                use_rec_scaffold: bool=True,
                drawn_negative_ratio: float=None,
                ema_rec_embed: bool=False,
                moco_style: bool=False,
                embed_dim: int=None,
                use_different_proj_head: bool=False,
                add_fedprox_penalty: bool=False,
                augmentation: Callable=None,
                hparam_list: List[str]=None):
        '''
        Protocol for model dict here
        For the default experiment(MNIST) in particular
        contains the following components
        - 'audio': backbone for audio
        - 'image': backbone for image
        - 'audio_proj_head': projection head mapping embedding to another 
                             subspace where the loss is computed
        - 'image_proj_head': dido
        - 'i2a': reconstruction MLP which maps image `embed` to audio's
        - 'a2i': reconstruction MLP which maps audio `embed` to image's 
        '''
        super(fedRecClient, self).__init__(
            cid,
            mod_state,
            loss_mode,
            model,
            loss_fn,
            trainloader,
            testloader,
            negative_aug_ratio=None,
            mmvae_config=None,
            device=device,
            augmentation=augmentation,
            hparam_list=hparam_list
        )
        self.rec_penalty = rec_penalty
        self.rec_loss_fn = nn.MSELoss()
        
        self._ema_rec_embed = ema_rec_embed
        if ema_rec_embed:
            assert embed_dim is not None, 'To pre-allocate disk buffer, dimension of embed should be informed' 
            self._embed_dim = embed_dim
        self._embed_pool_path = f'./.{cid}_embed_pool'
        self._use_rec_scaffold = use_rec_scaffold

        self._recnet_type = recnet_type
        self._moco_style = moco_style
        
        self._cached_ckp_path = f'./.{cid}_cached_ckp.pt'
        self._aggregatable_components = aggregatable_components
        self._loadable_components = aggregatable_components if loadable_components is None \
                                    else loadable_components
        
        self._inject_negative = drawn_negative_ratio is not None
        self._drawn_negative_ratio = drawn_negative_ratio
        
        self._filter_drawn_negative = sim_repel_fraction is not None
        if sim_repel_fraction is not None:
            assert recnet_type == 'MMVAE', 'when sim_repel_fraction is turned on, MMVAE like recnet should be used'
            assert sim_repel_fraction > 0. and sim_repel_fraction < 1., \
            'the portion of discarded drawn negative sample should be in the range from 0. to 1., exclusive'
            self._preserved_nega_fraction = 1 - sim_repel_fraction
        
        self._fedprox_like = add_fedprox_penalty
        
        
        self._use_different_proj_head = use_different_proj_head
    
    
    def assemble_mod_iterator(self, loss_mode: str='self'):
        '''
        Generate model iterator which contains model that produces
        embedding to contrast against.
        The order of this iterator should be consistent with inputs dict
        '''
        
        assert loss_mode in ['self', 'cross'], 'only self and cross keywords are supported'
        
        extractors = [] 
        if self.mod_state == STATE.BOTH:
            audio_extractor = StandardPipeline(
                backbone=self.model['audio'],
                proj_head=self.model[f'audio_{loss_mode}_proj_head']
            ).to(self.device)

            image_extractor = StandardPipeline(
                backbone=self.model['image'],
                proj_head=self.model[f'image_{loss_mode}_proj_head']
            ).to(self.device)
            extractors = [audio_extractor, image_extractor]
        elif self.mod_state == STATE.AUDIO:
            extractors = [
                StandardPipeline(
                    backbone=self.model['audio'],
                    proj_head=self.model[f'audio_{loss_mode}_proj_head']
                ).to(self.device)
            ]
        else: # image 
            extractors = [
                StandardPipeline(
                    backbone=self.model['image'],
                    proj_head=self.model[f'image_{loss_mode}_proj_head']
                ).to(self.device)
            ]
        return extractors
    
    
    def _cache_personal_ckp(self):
        cached_data = {}
        if self.mod_state == STATE.IMAGE:
            components_to_cache = ['image_self_proj_head', 'image_cross_proj_head']
        elif self.mod_state == STATE.AUDIO:
            components_to_cache = ['audio_self_proj_head', 'audio_cross_proj_head']
        else: # both
            components_to_cache = [
                'image_self_proj_head', 'image_cross_proj_head',
                'audio_self_proj_head', 'audio_cross_proj_head'
            ]
        for name in components_to_cache:
            cached_data[name] = self.model[name].state_dict()
        save_path = f'./.{self.cid}_cached_ckp.pt'
        torch.save(cached_data, save_path)
    
    
    def _load_personal_ckp(self):
        if not os.path.exists(self._cached_ckp_path):
            # skip loading at first round
            return 
        cached_data = torch.load(self._cached_ckp_path)
        if self.mod_state == STATE.IMAGE:
            components_to_cache = ['image_self_proj_head', 'image_cross_proj_head']
        elif self.mod_state == STATE.AUDIO:
            components_to_cache = ['audio_self_proj_head', 'audio_cross_proj_head']
        else: # both
            components_to_cache = [
                'image_self_proj_head', 'image_cross_proj_head',
                'audio_self_proj_head', 'audio_cross_proj_head'
            ]
        for name in components_to_cache:
            self.model[name].load_state_dict(cached_data[name])
        print('success load')

    # overload main loop in train if ema embed option is turned on 
    
    # @register_method_hook(exit=_cache_personal_ckp)
    
    def _train_impl(self,
                    epoch, model,
                    optimizer,
                    scheduler=None,
                    **kwargs):
        if self.mod_state == STATE.BOTH or (not self._ema_rec_embed):
            # do not overload
            return super()._train_impl(epoch, model, optimizer, scheduler, kwargs['ins'])
        else:
            # overload train main loop, passing batch idcs to fwd branching function

            # initialize embed pool 

            raw_dataset = self.trainloader.dataset
            self._generate_cached_embed_pool(
                model['audio'] if self.mod_state == STATE.AUDIO else model['image'],
                raw_dataset
            )
            batch_sampler = CustomBatchSampler(
                len(raw_dataset), batch_size=self.trainloader.batch_size,
                drop_last=True
            )
            hooked_dl = DataLoader(raw_dataset, batch_sampler=batch_sampler)
            round_loss = []

            for _ in range(epoch):
                running_loss = .0
                drawed_batch_idcs = batch_sampler.drawed_batch
                for batch_idcs, inputs in zip(drawed_batch_idcs, hooked_dl):
                    loss = self._fwd_branching(_zippable(inputs), batch_idcs=batch_idcs)
                    
                    if self._fedprox_like:
                        assert 'omega' in self.hparam_list, \
                        'when using fedprox l2 penalty, omega should be specified'
                        assert kwargs['ins'] is not None, 'fit ins should be propagated'
                        fedprox_loss = self.add_fedprox_penalty(kwargs['ins'])
                        loss += self.omega * fedprox_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                batch_sampler.shuffle()

                if scheduler is not None:
                    scheduler.step()
                
                avg_loss = running_loss / len(self.trainloader)
                round_loss.append(avg_loss)

            return round_loss



    @torch.no_grad()
    def _generate_missing_mod(self, embed: torch.Tensor, batch_idcs: List[int]=None) -> torch.Tensor:
        if self.mod_state == STATE.AUDIO:
            if self._recnet_type == 'MLP':
                generated = self.model['a2i'](embed)
            elif self._recnet_type == 'MMVAE':
                generated = self.model['mmvae'].reconstruct({'0':embed})['1']
        
        elif self.mod_state == STATE.IMAGE:
            if self._recnet_type == 'MLP':
                generated = self.model['i2a'](embed)
            elif self._recnet_type == 'MMVAE':
                generated = self.model['mmvae'].reconstruct({'1':embed})['0']
        else:
            raise NotImplementedError
        
        if self._ema_rec_embed:
            assert batch_idcs is not None
            with h5py.File(self._embed_pool_path, 'r+') as handler:
                pool = handler['embed_pool']
                # when accessing hd5 file, sorted index is required
                sorted_idcs = np.argsort(batch_idcs)
                revert_idcs = np.argsort(sorted_idcs)

                cached_embed = torch.tensor(
                    np.take(pool[batch_idcs[sorted_idcs], :], revert_idcs, axis=0)
                ).to(self.device)
                
                if self.gamma != 1.:
                    ema_generated = self.gamma * cached_embed + (1-self.gamma) * generated
                    # inplace update tensor on disk
                    handler['embed_pool'][batch_idcs[sorted_idcs], :] = _to_numpy(ema_generated)[sorted_idcs, :]
                if self._moco_style:
                    return generated, ema_generated
                else:
                    return generated
        return generated
    
    @staticmethod
    def _negative_filter(grounded_embed: torch.Tensor,
                         negatives: torch.Tensor,
                         topk: int):
        '''
        bundle operations of selecting negatives which have top k largest average distance to grounded embeddings
        '''
        sim_mat = pairwise_sim_metrics(grounded_embed, negatives, metrics='Euclidean')
        avg_sim_mat = torch.mean(sim_mat, dim=0) # take average across the first dim, i.e online embedding dim
        _, topk_idcs = torch.topk(avg_sim_mat, k=topk, largest=True)
        return negatives[topk_idcs]
        
    
    def _compute_cross_mod_loss(self,
                                mod_iterator: List[StandardPipeline],
                                inputs: Tuple[torch.Tensor],
                                **kwargs):
        '''
        Overwrite computation function for cross modality loss
        '''
        projs = []
        embeds = []
        for model, x in zip(mod_iterator, inputs):
            proj, embed = model(x.to(self.device), return_embed=True)
            projs.append(proj); embeds.append(embed)

        if self.mod_state != STATE.BOTH:
            # generate missing modalty when required
            possessed_embed = embeds[0]
            generated_embed = self._generate_missing_mod(
                possessed_embed,
                kwargs.get('batch_idcs', None)
            )
            
            mod_proj_head = self.model['image_cross_proj_head'] if self.mod_state == STATE.AUDIO \
                            else self.model['audio_cross_proj_head']
            
            # symmetric loss
            
            if self._moco_style:
                on_the_fly, moved = generated_embed
                proj_view1, proj_view2, ema_proj = projs[0], mod_proj_head(on_the_fly).detach(), mod_proj_head(moved).detach()
                # loss = self.loss_fn['cross-contra']['single'](
                #     proj_view1, ema_proj, proj_view2, drop_diag=True 
                # )
                query_rec_with_real_loss = self.loss_fn['cross-contra']['single'](
                    proj_view1, ema_proj, ema_proj, drop_diag=True 
                )
                query_real_with_rec_loss = self.loss_fn['cross-contra']['single'](
                    ema_proj, proj_view1, proj_view1, drop_diag=True
                )
                loss = (query_rec_with_real_loss + query_real_with_rec_loss) / 2
                
            else:
                # not moco style
                proj_view1, proj_view2 = projs[0], mod_proj_head(generated_embed).detach()
                loss = self.loss_fn['cross-contra']['single'](proj_view1, proj_view2, cross_view_only=True)

        else: # mod_state == STATE.BOTH
            # compute two way reconstruction loss
            embed_view1, embed_view2 = embeds
            proj_view1, proj_view2 = projs
            
            if self._inject_negative:
                batch_size = proj_view1.shape[0]
                num_drawn_sample = int(batch_size * self._drawn_negative_ratio)
                drawn_negatives = self._draw_negative_sample(num_drawn_sample, prior=None)
                
                nega_aug_embed_view1, nega_aug_embed_view2 = drawn_negatives['0'], drawn_negatives['1']
                
                if self._filter_drawn_negative:
                    # if negative filtering option is turned on, only a portion of generated negative sample will be preserved
                    # this method is proposed to mitigate the negative impact of non-iid distribution across different clients
                    # pipeline: draw negative samples from MMVAE pool --> only preserve first k, which is determined by sim_repel_fraction parameter, samples
                    # detailed fiter process is: compute average distance between each drawn negative sample and on-the-fly computed embeds
                    # select negatives which largest top k average distance
                    num_selected_nega_sample = int(self._preserved_nega_fraction * num_drawn_sample)
                    nega_aug_embed_view1 = fedRecClient._negative_filter(embed_view2, nega_aug_embed_view1, topk=num_selected_nega_sample)
                    nega_aug_embed_view2 = fedRecClient._negative_filter(embed_view1, nega_aug_embed_view2, topk=num_selected_nega_sample)
                with torch.no_grad():
                    nega_aug_proj_view1 = self.model['audio_cross_proj_head'](nega_aug_embed_view1)
                    nega_aug_proj_view2 = self.model['image_cross_proj_head'](nega_aug_embed_view2)
                    
                contrast_loss = self.loss_fn['cross-contra']['multi'](
                    proj_view1, proj_view2, 
                    nega_aug_proj_view1, 
                    nega_aug_proj_view2,
                    cross_view_only=True
                )
            else:
                contrast_loss = self.loss_fn['cross-contra']['multi'](proj_view1, proj_view2, cross_view_only=True)

            if self._use_rec_scaffold:
                rec_0_to_1 = self.rec_loss_fn(
                    self.model['a2i'](embed_view1), embed_view2.detach()
                )
                rec_1_to_0 = self.rec_loss_fn(
                    self.model['i2a'](embed_view2), embed_view1.detach()
                )
                rec_loss = (rec_0_to_1 + rec_1_to_0) / 2
                loss = (1-self.beta) * contrast_loss + self.beta * self.rec_penalty * rec_loss
            else:
                loss = contrast_loss
                
            # if self._use_damping_penalty:
            #     damping_embed_view1, damping_embed_view2 = self._produce_damping_embed(embed_view1, embed_view2)
            #     damping_penalty = self.loss_fn['damp'](damping_embed_view1, embed_view1) + \
            #                       self.loss_fn['damp'](damping_embed_view2, embed_view2)
            #     loss = loss + damping_penalty
    

        return loss
    
    def _produce_damping_embed(self, view1_embed, view2_embed):
        assert 'mmvae' in self.model.keys(), 'mmvae reconstruction mode should be used'
        query_embed = {'0':view1_embed, '1':view2_embed}
        purified_embed = self.model['mmvae'].reconstruct(query_embed, joint_posterior='PoE')
        
        drawed_view1_embed, drawed_view2_embed = purified_embed['0'], purified_embed['1']
        return drawed_view1_embed.detach(), drawed_view2_embed.detach()
        
    def _draw_negative_sample(self, num_sample: int,
                              prior: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        assert 'mmvae' in self.model.keys(), 'mmvae reconstruction mode should be used'
        if prior is None:
            # directly draw latents from isotropic gaussian
            generated = self.model['mmvae'].generate(num_sample)
        return generated
            
    
    # @register_method_hook(exit=_load_personal_ckp)
    
    def set_parameters(self, fit_ins: FitIns):
        ckp = deserialize(fit_ins.parameters)
        # alter the logic, using keys in self.model to query distributed checkpoint
        for module_name in self._loadable_components:
            self.model[module_name].load_state_dict(ckp[module_name])
    

    def _set_optimizer(self, ins: FitIns):
        
        base_lr = ins.config['lr'].get('base', 1e-2)
        
        param_group = []
        # three groups of parameters to be configured
        for component_name, sub_model in self.model.items():
            if '2' in component_name:
                lr = ins.config['lr']['rec']
            else: # including proj-head and backbone
                
                # use seperate lr for different component if provided
                # default base lr otherwise
                lr = ins.config['lr'].get(component_name, base_lr)
                # # use seperate lr for proj heads
                # if 'audio' == component_name:
                #     lr = ins.config['lr']['audio']
                # elif: # image 
                #     lr = ins.config['lr']['image']
            param_group.append({
                'params' : sub_model.parameters(),
                'lr' : lr
            })
        optimizer = optim.SGD(
            param_group,
            lr=ins.config['lr']['base'],
            weight_decay=ins.config['weight_decay']
        )
        return optimizer
    
    def _generate_cached_embed_pool(self, model: nn.Module, dataset):
        """
        save current embeds into disk. This pool acts like a proxy for the underlying
        tensor. It allows large tensor to be that retrieved using index

        This pool is used for updating embeds on a single-modal client in EMA manner. 
        Smooth averaged embed will be fed into reconstruction network instead of embeds
        directly produced by feature extractor. This approach can produce a more stable anchor 
        for single modality to contrast with
        """
        
        batch_size = 64
        in_ordered_dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        # toggle states of model
        # model.eval()
        
        
        with h5py.File(self._embed_pool_path, 'w') as handler:
            # pre-allocate disk space
            buffer = handler.create_dataset(
                name='embed_pool', shape=(len(dataset), self._embed_dim),
                dtype='float32'
            )

            for idx, x in enumerate(in_ordered_dataloader):
                if isinstance(x, list):
                    x = random.choice(x)
                embed = _to_numpy(model(x.to(self.device)))
                buffer[idx * batch_size : (idx+1) * batch_size] = embed
        # model.train()




    @property
    def _available_mod(self):
        if self._aggregatable_components is not None:
            return self._aggregatable_components
    
        shared_component = [
            'audio_proj_head',
            'image_proj_head'
        ]
        if self.mod_state == STATE.AUDIO:
            component = [
                *shared_component,
                'audio',
            ]
            if self._recnet_type == 'MLP':
                component = component + ['a2i']
        elif self.mod_state == STATE.IMAGE:
            component = [
                *shared_component,
                'image', 
            ]
            if self._recnet_type == 'MLP':
                component = component + ['i2a']
            
        else:
            component = [
                *shared_component,
                'audio', 
                'image'
            ]
            if self._use_rec_scaffold:
                component = component + ['a2i', 'i2a']
        return component
                
    def add_fedprox_penalty(self, fit_ins: FitIns):
        # pair up aggregatable components
        ckp = deserialize(fit_ins.parameters)

        reg_loss = .0
        for module_name in self._aggregatable_components:
            if 'head' in module_name:
                continue
            local_state_dict = dict(self.model[module_name].named_parameters())
            global_state_dict = ckp[module_name]
            reg_loss += aggregate.add_fedprox_regularizer(local_state_dict, global_state_dict)
            
        return reg_loss
    
    # hook destructor function to clear up file buffer
    def __del__(self):
        if os.path.exists(self._embed_pool_path):
            os.remove(self._embed_pool_path)
        # if os.path.exists(self._cached_ckp_path):
        #     os.remove(self._cached_ckp_path)


        




