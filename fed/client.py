import os
import multiprocessing as MP
from typing import (
    Dict, Tuple, List
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
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torchvision
import copy

from .communicate import serialize, deserialize
from .utils.sampling import dirichelet_sampling

from experiments.ssl.model import get_backbone
from experiments.ssl.dataset import get_simclr_transform

from utils.augmentation import Augmentation
from model import BYOL


from .utils.bind import register_method_hook
import time



def _attempted_process_close(p, error_msg):
    try:
        p.close()
    except:
        print(error_msg)



# def _tensor2np(t: torch.Tensor):
#     return t.detach().cpu().numpy()

def _get_default_aug():
    train_transform, test_transform = get_simclr_transform()
    train_aug = Augmentation(train_transform, n_view=2)
    test_aug = Augmentation(test_transform, n_view=2)
    return train_aug, test_aug

'''
    Try FedBYOL on Cifar10
'''

class Client(fl.client.Client):

    def __init__(self, 
                cid: int,
                model: nn.Module,
                trainloader, testloader=None,
                augmentaion=None,
                state_dir='.client'):
        super(Client, self).__init__()

        self.cid = cid
        self.model = model

        self._device = 'cpu'
        if augmentaion is None:
            self.augmentation = _get_default_aug()
        else:
            self.augmentation = augmentaion

        self.trainloader, self.testloader = trainloader, testloader
        self.num_examples = len(self.trainloader.dataset)

        self.state_dir = state_dir
        if not os.path.isdir(state_dir):
            os.makedirs(state_dir, exist_ok=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        payload = {
            'online_encoder' : self.model.online_network.state_dict(),
            'predictor' : self.model.predictor.state_dict(),
        }
        status = Status(code=Code.OK, message='Success')
        return GetParametersRes(
            status=status,
            parameters=serialize(payload)
        )
    
    def _load_target_network(self):
        filename = f'{self.state_dir}/{self.cid}.pt'
        if not os.path.exists(filename):
            return
        ckp = torch.load(filename)
        self.model.target_network.load_state_dict(ckp['target_net'])
        
    def _save_target_network(self):
        filename = f'{self.state_dir}/{self.cid}.pt'
        torch.save(
            {'target_net' : self.model.target_network.state_dict()},
            filename
        )

    
    @register_method_hook(_load_target_network, _save_target_network)
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
        payload = serialize({
            'online_encoder' : self.model.online_network.state_dict(),
            'predictor' : self.model.predictor.state_dict(),
        })
        metrics = {
            'cid' : int(self.cid),
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
        self.model.online_network.load_state_dict(ckp['online_encoder'])
        self.model.predictor.load_state_dict(ckp['predictor'])

        if fit_ins.config['init']:
            self.model.target_network = copy.deepcopy(self.model.online_network)
    
    @property
    def device(self):
        return self._device
    
    def to(self, device):
        self._device = device
        components = [self.model, self.augmentation]
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
        round_loss = []
        for _ in range(epoch):
            running_loss = .0
            for inputs in self.trainloader:
                x, _ = inputs
                x = x.to(self.device)
                view1, view2 = self.augmentation(x)
                loss = model(view1, view2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if scheduler is not None:
                scheduler.step()
            
            avg_loss = running_loss / len(self.trainloader)
            round_loss.append(avg_loss)

        return round_loss

