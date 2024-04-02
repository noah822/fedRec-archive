from collections import OrderedDict
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
from .facility import MLPRecNetTrainer, MMVAERecNetTrainer

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

from .utils._internal import _transpose_list, _check_keys, _iter_dict
from .config import STATE

from utils.scheduler import HparamScheduler




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
            save_path=None,
            hparam_scheduler: Dict[str, HparamScheduler] = None,
            from_ckp: str=None
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

        self._use_hparam_sche = hparam_scheduler is not None
        self.hparam_scheduler = hparam_scheduler

        self._resume = from_ckp is not None
        self._pretrained_ckp = from_ckp

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
        if self._resume:
            payload = torch.load(self._pretrained_ckp)
            print('Successfully load checkpoint at server!')
        else:
            audio_encoder = get_mnist_audio_encoder()
            image_encoder = get_mnist_image_encoder()
            if self.use_mmvae:
                mmvae = self._load_mmvae()
            else:
                mmvae = None
            payload = {
                'audio' : audio_encoder.state_dict(),
                'image' : image_encoder.state_dict(),
                'mmvae' : mmvae.state_dict() if mmvae is not None else None
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
            'lr' : {'audio':1e-3, 'image':1e-1, 'base':1e-3},
            'weight_decay' : 1e-5
        }
        config = {
            'init' : True if server_round == 0 else False,
            'local_epoch' : self.local_epoch,
            **optimizer_config
        }
        
        # if hpara scheduler is used, config hparams for client in next round
        if self._use_hparam_sche:
            for hparam_name, scheduler in self.hparam_scheduler.items():
                config[hparam_name] = scheduler.get_value(server_round)

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
            client_param_pool.push((client_param, result.num_examples))

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
        payload['mmvae'] = fitted_mmvae
        
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
            _iter_dict(extractors),
            self.device
        )
        dataloader_config = {
            'batch_size' : 32,
            'shuffle' : True,
        }
        trainer = MMVAETrainer(
            self._load_mmvae(),
            embeds_dataset,
            dataloader_config
        )
        trainer.train()
        fitted_mmvae = trainer.export_mmvae()

        return fitted_mmvae
    

    @property
    def extractors(self) -> Tuple[nn.Module]:
        models = [
            get_mnist_audio_encoder().to(self.device),
            get_mnist_image_encoder().to(self.device)
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
    def push(self, client_res: List):
        '''
           push param_list produced by a client to the query
           if client misses some submodule, it should be explicitly notified by
           passing None
        '''
        param_list, num_examples = client_res
        for idx, submodule in enumerate(param_list):
            if submodule is None:
                continue
            else:
                self._param_queues[idx].append((submodule, num_examples))
                
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
        return iter(self._param_queues)
    

class BaseServer(fl.server.strategy.Strategy):
    def __init__(self,
            num_client,
            optim_config: Dict=None,
            local_epoch=5,
            device='cuda',
            tensorboard_writer=None,
            save_ckp_interval:int=None,
            hparam_scheduler: Dict[str, HparamScheduler]=None,
            save_path='./linear_probe.pt',
            ):
        super(BaseServer, self).__init__()
        self.num_client = num_client

        self.local_epoch = local_epoch

        self.device = device
        self.writer = tensorboard_writer
        self._use_tb = self.writer is not None

        self.save_ckp_interval = save_ckp_interval
        self.save_path = save_path

        self._optim_config = optim_config
        self._use_hparam_sche = hparam_scheduler is not None
        self.hparam_scheduler = hparam_scheduler
    
    @property
    def optim_config(self):
        _default = {
            'lr' : 1e-3,
            'weight_decay' : 1e-5
        }
        if self._optim_config is None:
            return _default
        else:
            return self._optim_config


    def log_client_res(self,
                       server_round: int,
                       results: List[Tuple[ClientProxy, EvaluateRes]]):
        for _, result in results:
            cid, loss = self._unpack_client_loss(result)

            if self._use_tb:
                graph_idx = f'Client/'
                client_idx = f'{cid}'

                start_epoch = (server_round-1) * self.local_epoch 
                self._write_tensorboard(
                    graph_idx, 
                    client_idx,
                    loss,
                    start_epoch
                )

    def aggregate_params(self,
                        server_round: int, 
                        results: List[Tuple[ClientProxy, EvaluateRes]],
                        num_module: int=2,
                        exclude_list=['batch']):
        client_param_pool = ParamQueue(num_module=num_module)

        for _, result in results:
            client_param = self._unpack_client_params(result)
            client_param_pool.push((client_param, result.num_examples))
            
        aggregated_param = []
        for submodel in client_param_pool:
            aggregated_param.append(fedAvg(submodel, exclude_list))
        return aggregated_param

    

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

    def aggregate_fit(self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures              
        ) -> Parameters:
        exclude_list = ['batch']

        aggregated_param = self.aggregate_params(
            server_round,
            results,
            exclude_list
        )

        # log client result in previous round
        self.log_client_res(server_round, results)
        payload = aggregated_param
        
        if  self.save_ckp_interval is not None and \
            server_round % self.save_ckp_interval == 0:
            self._save_ckp(payload)


        return serialize(payload), {}
    
    
    def initialize_parameters(self, client_manager: ClientManager):
        raise NotImplementedError
    
    
    def configure_fit(self, 
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(self.num_client)
        config = {
            'init' : True if server_round == 0 else False,
            'local_epoch' : self.local_epoch,
            **self.optim_config
        }

        if self._use_hparam_sche:
            for hparam_name, scheduler in self.hparam_scheduler.items():
                config[hparam_name] = scheduler.get_value(server_round)
        
        fit_configs = []
        for client in clients:
            fit_configs.append(
                (client, FitIns(parameters, config))
            )
        return fit_configs

    
    def _unpack_client_params(self, fit_res: FitRes):
        ckp = deserialize(fit_res.parameters)
        return ckp

    def _unpack_client_loss(self,
                            fit_res: FitRes
                            ) -> Tuple[int, List]:

        cid = fit_res.metrics['cid']
        client_loss = []
        
        for i in range(self.local_epoch):
            client_loss.append(fit_res.metrics[f'epoch_{i}'])
        
        return cid, client_loss
    

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
        
    def _save_ckp(self, payload):
        torch.save(payload, self.save_path)
        
        


import copy
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from .utils.pipeline import fit_cross_mod_rec

class fedRecServer(BaseServer):
    def __init__(self,
            num_client,
            init_model: nn.Module,
            public_dataset: Dataset,
            optim_config: Dict=None,
            local_epoch=5,
            device='cuda',
            recnet_type: str='MLP',
            extra_rec_kwargs: Dict=None,
            aggregatable_components: List[str]=None,
            server_train_epoch: int=5,
            mmvae_config: Dict[str, Callable[..., nn.Module]]=None,
            tensorboard_writer=None,
            save_ckp_interval:int=None,
            hparam_scheduler: Dict=None,
            save_path: str='./fed.pt',
            single_modal_discard_round: int=0,
            from_ckp: str=None,
            verbose=False
            ):
        super(fedRecServer, self).__init__(
            num_client,
            optim_config,
            local_epoch,
            device,
            tensorboard_writer,
            save_ckp_interval,
            hparam_scheduler,
            save_path
        )
        self.init_model = init_model
        self.public_dataset = public_dataset
        
        
        self.components = self.init_model.keys() if aggregatable_components is None else aggregatable_components
        self.verbose = verbose
        self._resume = from_ckp is not None
        self._pretrained_ckp_path = from_ckp
        assert recnet_type in ['MLP', 'MMVAE', 'None'], \
            """
            current implementation only supports two types of reconstruction network:
            MLP & MMVAE, other key words are invalid
            """
        self._recnet_type = recnet_type
        self._mmvae_config = mmvae_config
        self._extra_rec_kwargs = extra_rec_kwargs
        
        self._server_train_epoch = server_train_epoch
        self._single_modal_discard_round = single_modal_discard_round
    
    def initialize_parameters(self, client_manager: ClientManager):
        payload = {}
        if self._resume:
            payload = torch.load(self._pretrained_ckp_path)
            print('Successfully load pretrained checkpoint at server!')
        else:
            for name in self.init_model.keys():
                payload[name] = self.init_model[name].state_dict()
        return serialize(payload)
    
    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures) -> Parameters:
        
        # log client result
        self._cur_server_round = server_round  #debug hack
        
        self.log_client_res(server_round, results)
        # list of aggregated params
        aggregated_param = self.aggregate_params(server_round, results, num_module=len(self.components))
        
        # wrap list of parameters into dict
        aggregated_param_dict = {}
        for param, name in zip(aggregated_param, self.components):
            aggregated_param_dict[name] = param

        if self._recnet_type != 'None':
            assert self._extra_rec_kwargs is not None, \
            'Architecture specific key word arguments for reconstruction network should be provided'
            fitted_recnet = self.train_recnet(
                self._recnet_type,
                self.build_extractors(aggregated_param_dict),
                epoch=self._server_train_epoch,
                optim_config={'lr' : 1e-3, 'weight_decay' : 1e-5},
                verbose=True,
                **self._extra_rec_kwargs
            )
            for k, state_dict in fitted_recnet.items():
                aggregated_param_dict[k] = state_dict
            
#             mlp_recnet = self.train_recnet(
#                 'MLP',
#                 self.build_extractors(aggregated_param_dict),
#                 epoch=self._server_train_epoch,
#                 optim_config={'lr' : 1e-3, 'weight_decay' : 1e-5},
#                 verbose=True,
#                 embed_dim=64,
#                 hidden_dim=128
#             )
#             mmvae_recnet = self.train_recnet(
#                 'MMVAE',
#                 self.build_extractors(aggregated_param_dict),
#                 epoch=self._server_train_epoch,
#                 optim_config={'lr' : 1e-3, 'weight_decay' : 1e-5},
#                 verbose=True,
#                 embed_dim=64,
#                 hidden_dim=32,
#                 bottleneck_dim=2,
#                 alpha=0.001
#             )
            
#             for k, state_dict in mlp_recnet.items():
#                 aggregated_param_dict[k] = state_dict
#             for k, state_dict in mmvae_recnet.items():
#                 aggregated_param_dict[k] = state_dict
        
        payload = aggregated_param_dict
        
        if  self.save_ckp_interval is not None and \
            server_round % self.save_ckp_interval == 0:
            self._save_ckp(payload)

        return serialize(payload), {}

    # we fine tune the mapping between embeds from different modalities
    def build_extractors(self, aggregated_param_dict):
        audio_backbone = copy.deepcopy(self.init_model['audio']).to(self.device)
        audio_backbone.load_state_dict(aggregated_param_dict['audio'])

        image_backbone = copy.deepcopy(self.init_model['image']).to(self.device)
        image_backbone.load_state_dict(aggregated_param_dict['image'])
        return [audio_backbone, image_backbone]

    def train_recnet(self,
                     option: str,
                     extractors: List[nn.Module],
                     epoch: int,
                     dl_config: Dict=None,
                     optim_config: Dict=None,
                     verbose: bool=True,
                     **kwargs):
        if option == 'MLP':
            executor = MLPRecNetTrainer(
                extractors=extractors, raw_dataset=self.public_dataset,
                optim_config=optim_config,
                **kwargs
            )
        elif option == 'MMVAE':
            executor = MMVAERecNetTrainer(
                extractors=extractors, raw_dataset=self.public_dataset,
                optim_config=optim_config,
                **kwargs
            )
        executor.run(epoch, dl_config, verbose)
        return executor.export_model()


    def _unpack_client_params(self, fit_res: FitRes):
        ckp = deserialize(fit_res.parameters)
        param_group = OrderedDict({
            group_name : None for group_name in self.components
        })
        client_state = fit_res.metrics['state']
        
        if self._cur_server_round <= self._single_modal_discard_round \
           and client_state != STATE.BOTH.value:
            print('skip single-modal client at starting stage')
            return param_group.values()

        # aggregatable_components = []
        
#         if client_state == STATE.AUDIO.value:
#            aggregatable_components = [
#                'audio', 'audio_proj_head',
#                'a2i', 'audio_self_proj_head'
#            ]
#            # print('skip audio client')
#            # return param_group.values()
    
#         elif client_state == STATE.IMAGE.value:
#             aggregatable_components = [
#                 'image', 'image_proj_head',
#                 'i2a'
#             ]
#             # print('skip image client')
#             # return param_group.values()
#         else: # STATE.BOTH
#             aggregatable_components = [
#                 'audio', 'audio_proj_head',
#                 'image', 'image_proj_head',
#                 'a2i', 'i2a'
#             ]
            
        for module_name in self.components:
            if module_name not in ckp.keys():
                continue

            param_group[module_name] = ckp[module_name]

        return param_group.values()


    