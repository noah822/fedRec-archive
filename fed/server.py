from typing import (
    Dict, Optional,
    Tuple, Union,
    List
)
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from .communicate import serialize, deserialize
from .aggregate import *

from experiments.ssl.model import get_backbone
from model import BYOL

class Server(fl.server.strategy.Strategy):
    def __init__(self):
        super(Server, self).__init__()

    def initialize_parameters(self, client_manager: ClientManager):
        backbone = get_backbone('resnet18')
        model = BYOL(
            backbone,
            inplanes=512,
            pred_hidden_dim=1024,
            epsilon=0.98,
            proj_dim=256,
            proj_hidden_dim=1024
        )
        payload = {
            'online_encoder' : model.online_network.state_dict(),
            'predictor' : model.predictor.state_dict()
        }
        return serialize(payload)
    
    def aggregate_evaluate(self,
            server_round: int, 
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures):
        pass
    def evaluate(self,
            server_round: int,
            parameters: Parameters):
        pass
    def configure_evaluate(self,
                server_round: int,
                parameters: Parameters,
                client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        pass
    
    def configure_fit(self, 
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.all()

        optimizer_config = {
            'lr' : 1e-3,
            'weight_decay' : 1e-5
        }
        config = {
            'init' : True if server_round == 0 else False,
            'local_epoch' : 5,
            **optimizer_config
        }

        fit_configs = []
        for _, client in clients.items():
            fit_configs.append(
                (client, FitIns(parameters, config))
            )
        return fit_configs

    
    def aggregate_fit(self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures              
        ) -> Parameters:
        online_encoders = []
        predictors = []

        for _, result in results:
            online_encoder, predictor = Server._unpack_client_params(result)
            online_encoders.append(
                (online_encoder, result.num_examples)
            )
            predictors.append(
                (predictor, result.num_examples)
            )

        exclude_list = ['batch']

        aggregated_online_encoder = fedAvg(
            online_encoders, exclude_list
        )
        aggregated_predictor = fedAvg(
            predictors, exclude_list
        )

        payload = {
            'online_encoder' : aggregated_online_encoder,
            'predictor' : aggregated_predictor
        }
        return serialize(payload), {}



    @staticmethod
    def _unpack_client_params(fit_res: FitRes):
         ckp = deserialize(fit_res.parameters)
         return ckp['online_encoder'], ckp['predictor']
    