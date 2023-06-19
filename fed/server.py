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
    def __init__(self,
            num_client=10,
            local_epoch=5,
            tensorboard_writer=None,
            cluster_size: int=5,    
        ):
        super(Server, self).__init__()

        self.num_client = num_client
        self.local_epoch = local_epoch
        self._writer = tensorboard_writer
        self._use_tb = False
        self.cluster_size = cluster_size
        if self._writer is not None:
            self._use_tb = True
    
    @property
    def writer(self):
        assert self._writer is not None
        return self._writer

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
        clients = client_manager.sample(self.num_client)

        optimizer_config = {
            'lr' : 1e-3,
            'weight_decay' : 1e-5
        }
        config = {
            'init' : True if server_round == 0 else False,
            'local_epoch' : self.local_epoch,
            **optimizer_config
        }

        fit_configs = []
        for client in clients:
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
            online_encoder, predictor = self._unpack_client_params(result)
            cid, loss = self._unpack_client_loss(result)
            online_encoders.append(
                (online_encoder, result.num_examples)
            )
            predictors.append(
                (predictor, result.num_examples)
            )

            if self._use_tb:
                graph_idx, client_idx = self._route_client_index(cid)
                start_epoch = (server_round-1) * self.local_epoch 
                self._write_tensorboard(
                    graph_idx, 
                    client_idx,
                    loss,
                    start_epoch
                )

                self.writer.add_scalars(
                    graph_idx,
                    {client_idx : loss},
                    (server_round-1) * self.local_epoch
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
    

    
        

    def _write_tensorboard(self,
        parent_index: str,
        child_index: str,
        data: List[float],
        start_epoch: int
    ):
        for i, value in enumerate(data):
            self.writer.add_scalars(
                parent_index,
                {child_index : value},
                start_epoch + i
            )
        
    def _route_client_index(self, cid: int) -> Tuple[str, str]:
        parent_idx = 'Client/Loss'
        cluster_idx = int(cid / self.cluster_size)
        return f'{parent_idx}/{cluster_idx}', f'client_{cid}'
        
    

    def _unpack_client_loss(self, fit_res: FitRes) -> Tuple[int, List[float]]:
        loss = [fit_res.metrics[f'epoch_{i}'] for i in range(self.local_epoch)]
        cid = fit_res.metrics['cid']
        return (cid, loss)

    def _unpack_client_params(self, fit_res: FitRes):
         ckp = deserialize(fit_res.parameters)
         return ckp['online_encoder'], ckp['predictor']
    