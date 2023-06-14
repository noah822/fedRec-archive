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



def _tensor2np(t: torch.Tensor):
    return t.detach().cpu().numpy()

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
                augmentaion=None):
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

    def fit(self, ins: FitIns) -> FitRes:
        # load parameters 
        self.set_parameters(ins)
        epoch = ins.config['local_epoch']
        optimizer = self._set_optimizer(ins)

        # self._train_impl(
        #     epoch,
        #     self.model,
        #     optimizer
        # )

        status = Status(code=Code.OK, message='Success')
        payload = serialize({
            'online_encoder' : self.model.online_network.state_dict(),
            'predictor' : self.model.predictor.state_dict()
        })

        fit_res = FitRes(
            parameters=payload,
            status=status,
            num_examples=self.num_examples,
            metrics={}
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
        for _ in range(epoch):
            for inputs in self.trainloader:
                x, _ = inputs
                x = x.to(self.device)
                view1, view2 = self.augmentation(x)
                loss = model(view1, view2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()



def _get_criterion(dataset):
    criterion = []
    for item in dataset:
        _, label = item
        criterion.append(label)
    return criterion

def _get_client_loader(datasets: List[Dataset], config):
    loaders = []
    for dataset in datasets:
        loader = DataLoader(dataset, **config)
        loaders.append(loader)
    return loaders


NUM_CLIENTS = 10
dataset = torchvision.datasets.CIFAR10(
    './data', train=True, download=True,
    transform=lambda x: T.ToTensor()(x)
)
labels = _get_criterion(dataset)
client_datasets = dirichelet_sampling(
    dataset,
    labels,
    NUM_CLIENTS,
    alpha=5
)

dataloader_config = {
    'batch_size' : 32,
    'shuffle' : True
}

client_dataloaders = _get_client_loader(
    client_datasets,
    dataloader_config
)


def setup_client(cid: str):
    dataloader = client_dataloaders[int(cid)]
    augmentation, _ = _get_default_aug()
    backbone = get_backbone('resnet18')
    model = BYOL(
        backbone,
        512,
        1024,
        0.98,
        256,
        1024
    )
    client = Client(
        cid, model,
        dataloader,
        augmentaion=augmentation
    )
    return client
