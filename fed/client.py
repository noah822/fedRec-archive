from typing import Dict, Tuple
import flwr as fl
from flwr.common import Config, NDArrays, Scalar
import torch
import torch.nn as nn
import numpy as np
import torchvision



model = torchvision.models.resnet18(pretrained=False)


def _tensor2np(t: torch.Tensor):
    return t.detach().cpu().numpy()

class Client(fl.client.NumPyClient):

    def __init__(self, trainloader, testloader=None):
        super(Client, self).__init__()
        self.trainloader, self.testloader = trainloader, testloader

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        np_params_list = [_tensor2np(v) for (_, v) in model.state_dict().items()]
        return np_params_list 
    

    def _train_impl(self):
        pass

def setup_client(id):
    pass
