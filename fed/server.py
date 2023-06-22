from typing import (
    Dict, Optional,
    Tuple, Union,
    List,
    Callable
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

import torch
import torch.nn as nn

from reconstruct.mmvae import DecoupledMMVAE
from .utils.pipeline import (
    MMVAETrainer, 
    PairedFeatureBank
)

from experiments.mmvae.mnist.model import (
    get_mnist_image_encoder,
    get_mnist_audio_encoder
)

from .utils._internal import _transpose_list, _check_keys
from .config import STATE




class Server(fl.server.strategy.Strategy):
    def __init__(self,
            num_client=10,
            local_epoch=5,
            public_dataset=None,
            use_mmvae=False,
            mmvae_config: Dict=None,
            device='cuda',
            tensorboard_writer=None,
            cluster_size: int=5,  
            save_ckp_interval=None,
            save_path=None  
        ):
        super(Server, self).__init__()


        self.num_client = num_client
        self.local_epoch = local_epoch
        self._writer = tensorboard_writer
        self._use_tb = False
        self.cluster_size = cluster_size

        # if use mmvae, public dataset to train it should be provided
        self.use_mmvae = use_mmvae
    
        if use_mmvae:
            assert public_dataset is not None
            assert mmvae_config is not None

            assert _check_keys(
                mmvae_config,
                ['encoders', 'decoders', 'latent_dim', 'score_fns']
            )
            self.mmvae_config = mmvae_config


        if self._writer is not None:
            self._use_tb = True

        self.save_ckp_interval = save_ckp_interval
        self._save_path = save_path

        self.public_dataset = public_dataset
        self.cached_state_dict = None

        self.device = device

    def _save_ckp(self, payload):
        torch.save(payload, self.save_path)
        

    @property
    def save_path(self):
        if self._save_path is None:
            return './server.pt'
        else:
            return self._save_path

    @property
    def writer(self):
        assert self._writer is not None
        return self._writer

    def initialize_parameters(self, client_manager: ClientManager):
        audio_encoder = get_mnist_audio_encoder()
        image_encoder = get_mnist_image_encoder()
        mmvae = self._load_mmvae()
        payload = {
            'audio' : audio_encoder.state_dict(),
            'image' : image_encoder.state_dict(),
            'mmvae' : mmvae.state_dict()
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

        client_param_pool = ParamQueue(num_module=2)

        for _, result in results:
            client_param = self._unpack_client_params(result)
            client_param_pool.push(client_param)

            cid, loss = self._unpack_client_loss(result)
            state = self._unpack_client_state(result)

            if self._use_tb:
                graph_idx, client_idx = self._route_client_index(cid)
                start_epoch = (server_round-1) * self.local_epoch 
                self._write_tensorboard(
                    graph_idx, 
                    client_idx,
                    loss,
                    start_epoch
                )

        exclude_list = ['batch']

        aggregated_param = []
        for submodel in client_param_pool:
            aggregated_param.append(fedAvg(submodel, exclude_list))

        self.cached_state_dict = aggregated_param

        fitted_mmvae = None
        if self.use_mmvae:
            fitted_mmvae = self._get_fitted_mmvae()

        payload = {}
        for key, param in zip(self.distributed_key, aggregated_param):
            payload[key] = param

        # distribute trained MMVAE
        payload['mmvae'] = fitted_mmvae if state != STATE.BOTH.value else None
        
        if  self.save_ckp_interval is not None and \
            server_round % self.save_ckp_interval == 0:
            self._save_ckp(payload)


        return serialize(payload), {}
    
    @property
    def distributed_key(self):
        _default = ['audio', 'image']
        return _default    

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
        return f'{parent_idx}/cluster_{cluster_idx}', f'client_{cid}'
        

    def _unpack_client_loss(self, fit_res: FitRes) -> Tuple[int, List[float]]:
        loss = [fit_res.metrics[f'epoch_{i}'] for i in range(self.local_epoch)]
        cid = fit_res.metrics['cid']
        return (cid, loss)

    def _unpack_client_params(self, fit_res: FitRes):
        ckp = deserialize(fit_res.parameters)
        module_param = [None, None]
        if fit_res.metrics['state'] == STATE.AUDIO.value:
            module_param[0] = ckp['audio']
        elif fit_res.metrics['state'] == STATE.IMAGE.value:
            module_param[1] = ckp['image']
        else:
            module_param[0] = ckp['audio']
            module_param[1] = ckp['image']
        return module_param

        
 
    def _unpack_client_state(self, fit_res: FitRes):
         return fit_res.metrics['state']
    
    '''
        Extract embeddings of available public dataset using the aggregated 
        encoder model, which will be later used to train MMVAE on server end. 
        The trained MMVAE will be distributed to clients in the following rounds for 
        missing modality reconstruction
    '''
    def _get_fitted_mmvae(self):
        extractors = self._load_extractors()
        embeds_dataset = PairedFeatureBank(
            self.public_dataset,
            extractors
        )
        trainer = MMVAETrainer(
            self._load_mmvae(),
            embeds_dataset
        )
        trainer.train()
        fitted_mmvae = trainer.export_mmvae()

        return fitted_mmvae
    

    @property
    def extractors(self) -> Tuple[nn.Module]:
        models = [
            get_mnist_audio_encoder(),
            get_mnist_image_encoder()
        ]
        return models

    def _load_extractors(self):
        model = nn.ModuleDict()
        zipped_mod_net_state = zip(
            self.distributed_key,
            self.extractors,
            self.cached_state_dict
        )
        for mod, extractor, state_dict in zipped_mod_net_state:
            extractor.load_state_dict(state_dict)
            model[mod] = extractor
        return model 

    def _load_mmvae(self):
        mmave_param_list = {
            'encoders' : self.mmvae_config['encoders']().to(self.device),
            'decoders' : self.mmvae_config['decoders']().to(self.device),
            'latent_dim' : self.mmvae_config['latent_dim'],
            'score_fns' : self.mmvae_config['score_fns']().to(self.device)
        }
        mmvae = DecoupledMMVAE(**mmave_param_list, device=self.device)
        return mmvae
    



class ParamQueue:
    def __init__(self, num_module):
        self.num_module = num_module

        self._param_queues = [[] for _ in range(num_module)]
    def push(self, param_list: List):
        '''
           push param_list produced by a client to the query
           if client misses some submodule, it should be explicitly notified by
           passing None
        '''
        for param in param_list:
            for idx, submodule in enumerate(param):
                if submodule is None:
                    continue
                else:
                    self._param_queues[idx].append(submodule)
                
    def pop(self):
        '''
            pop array of params of submodules collected from available clients,
            the return value can be in type of skewed two dimensional list
        '''
        return self._param_queues
    def __getitem__(self, idx):
        assert idx < self.num_module
        return self._param_queues[idx]
    def __iter__(self):
        return self._param_queues

        