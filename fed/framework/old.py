from threading import Thread
from multiprocessing import Process
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter


from collections.abc import Iterable


import os
from pathlib import Path

from abc import ABC, abstractmethod
from itertools import chain
import pandas as pd
import numpy as np

def _return_type_check(check, err_message=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            if isinstance(res, check):
                return res
            else:
                if err_message is not None:
                    raise err_message
                else:
                    raise 'return type of {}() should be {}'.format(
                        func.__name__, check.__name__
                    )
        return wrapper
    return decorator


class _meta_checker:
    def __new__(cls, classname, bases, attrs):
        if classname != 'Client':
            _train = attrs['train']
            attrs['train'] = _return_type_check(Iterable)(_train)
            
        return type(classname, (ABC, ), attrs)
        


class Client(meta_class=_meta_checker):
    
    
    
    def __init__(self):
        super(Client, self).__init__()
    @abstractmethod
    def train(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def load_model(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def get_model(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def state(self):
        pass
    @abstractmethod
    def load_state(self):
        pass
class Server(ABC):
    def __init__(self):
        super(ABC, self).__init__()
        self.logging = True
    
    
    @abstractmethod
    def validate(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def aggregate(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def get_model(self):
        pass
    
    @abstractmethod
    def load_model(self, param):
        pass
    
    @abstractmethod
    def upload_model(self, params):
        pass

    
    @abstractmethod
    def state(self):
        pass
    
    @abstractmethod
    def load_state(self):
        pass


class fed_framework:
    
    

    
    def __init__(self,
                 clients, server,
                 global_round,
                 num_client=10, num_thread=10,
                 mode='new',
                 resume_mode='load_last', resume_overwrite=True,
                 device=None,
                 use_tensorboard=True, tensorboard_path='./runs', tensorboard_sp=0,
                 ckpt_path='./fed_ckpt'):
        '''

            ckpt_path: directory where the federated framework is saved, which have the following structure
                .
                ├── server.pt
                └── clients           ... client *.pt files are 0 indexed
                    ├── 0.pt
                    ├──  ⋮
                    └── {num_client-1}.pt
                    
                *.pt file should contain a dictionary with at least the following key-value pairs:
                `round`
                `model`     -> torch state_dict
                `optimizer` -> torch state_dict
                `scheduler` -> torch state_dict
                
            by default, in resume mode
            newly saved model will overwrite previously saved record
                
        '''
        
        '''
            to use fed_framework API
            model, optimizer, scheduler, aggregator parameters should be of callable type
            
            train, test should be passed as functions
            
            parameters of those callable objected as passed via {}_config parameter
            
            inside the framework:
            
            '_' prefix notes that the parameter is instance
            train(_model, _optimizer, _scheduler, **train_config)
            test(_model, **test_config)
            

        
        '''
        self.clients = clients
        self.server = server
        
        self.global_round = global_round
        self.cur_round = 0
        
        self.num_thread = num_thread
        self.num_client = num_client
        
        
        self.ckpt_path = ckpt_path
        self.writer = None
        if use_tensorboard:
            self.wrtier = SummaryWriter(tensorboard_path)
            self.cur_round = tensorboard_sp
            
        # if mode == 'new':
            
        #     for _label in range(num_client):
        #         _model = self.model(**model_config)
        #         _optimizer = self.optimizer(_model.parameters(), **optimizer_config)
        #         _scheduler = torch.optim.lr_scheduler.StepLR(
        #             _optimizer,
        #             **scheduler_config
        #         )
                
        #         self.clients.append(
        #             Client(label=_label, model=_model,
        #                 optimizer=_optimizer, scheduler=_scheduler,
        #                 train_config=local_train_config
        #             )
        #         )
        #         self.server = Server()
                    
        # elif args.mode == 'resume':
        #     self.clients, self.server = self.load_framework_state()
            
        
    
    def run(self):

        
        _clients_loss = [None for _ in range(self.num_client)]
        if self.num_thread > 0:
            
            
            def _thread_train(clients, _base):
                for i, client in enumerate(clients):
                    _clients_loss[_base+i] = client.train()    
            
            
            _threads = []
            clients_per_thread = int(len(self.clients)/self.num_thread)
            
            
            
            for i in range(self.num_thread):
                _threads.append(
                    Process(target=_thread_train,
                            args=(
                                self.clients[i*clients_per_thread : (i+1)*clients_per_thread],
                                i*clients_per_thread
                            )
                        )
                )
                
            
        for round in range(self.global_round):
            
            
            '''
                clients load aggreated model from server
            '''
            if round > 0: 
                for client in self.clients:
                    client.load_model(self.server.get_model())
            
            if self.num_thread > 1:        
                for _thread in _threads: _thread.start()
                for _thread in _threads: _thread.join()
            else:
                for i, client in enumerate(self.clients):
                    _clients_loss[i] = client.train()
                    

            '''
                collect parameters of client model, which will be then aggregated at server end
            '''
            
            _upload = [
                {
                    'model_param' : client.get_model()
                    # 'hyper_param' : client.get_hyper_param()  
                } for client in self.clients
            ]
            
            
            self.server.upload_model(_upload)
            self.server.aggregate()
            _val_res = self.server.validate()
            
            self.cur_round += 1
            
            
            if self.server.logging and self.writer is not None:
                self.__server_tb_logging(_val_res)
                
            if self.clients[0].logging and self.writer is not None:
                self.__clients_tb_logging(_clients_loss)
            
    
    
    def save_framework_state(self):
        if not os.path.isdir(self.ckpt_path):
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        _clients_save_path = os.path.join(self.ckpt_path, 'clients')
        if not os.path.isdir(_clients_save_path):
            os.mkdir(_clients_save_path)
        
        
        torch.save(
            self.server.state(),
            os.path.join(self.ckpt_path, 'server.pt')
        )
        
        
        for i, client in enumerate(self.clients):
            torch.save(
                client.state(),
                os.path.join(_clients_save_path, f'{i}.pt')
            )
            
            
    def __reset__attribute(self, **kwargs):
        for (k, v) in kwargs:
            if k not in self.__dict__:
                raise 'wrong fed_framework attribute name, please double-check'
            else:
                self.__dict__[k] = v
            
    
    def load_framework_state(self, ckpt_path, **kwargs):
        if not os.path.isdir(ckpt_path):
            raise 'checkpoint directory to load the model from does not exist'
        _server_ckpt_path = os.path.join(ckpt_path, 'server.pt')
        
        self.server.load_state(
            torch.load(_server_ckpt_path)
        )
        
        
        _client_ckpt_path = os.path.join(ckpt_path, 'clients')
        for idx in range(self.num_client):
            _path = os.path.join(_client_ckpt_path, f'{idx}.pt')
            self.clients[idx].load_state(
                torch.load(_path)
            )
        
        self.__reset__attribute(**kwargs)
            
        
    def __server_tb_logging(self, data):
        graph_name = 'server/Accuracy'
        self.writer.add_scalar(
            graph_name, data, self.cur_round
        )
    
    def __clients_tb_logging(self, data, cluster_size=5):
        print(data)
        graph_name = 'client/Loss'
        num_sub_graph = int((self.num_client-1)/cluster_size) + 1
        
        
        for k in range(num_sub_graph):
            _graph_name = '{}/{}-{}'.format(
                graph_name,
                k*cluster_size,
                min(self.num_client-1, (k+1)*cluster_size-1)          
            )
            _step_base = self.cur_round * data[0].shape[0] + 1
            _client_base = k * cluster_size
            _data = np.array(data[k*cluster_size: (k+1)*cluster_size]).T
            for i, v in enumerate(_data):
                self.writer.add_scalars(_graph_name, 
                    {'{}'.format(_client_base + c) : v[c] for c in range(len(v))},
                    _step_base + i
                )
            
        
    @staticmethod
    def __train_test_split(csv_path, dst_dir, split_ratio, save=True):
        
        df = pd.read_csv(csv_path, header=None)
        
        _mask = np.random.rand(df.shape[0]) <= split_ratio
        
        test_df = df[~_mask]
        train_df = df[_mask]
        
        if save:
            test_df.to_csv(
                os.path.join(dst_dir, 'test.csv'), index=None, header=None
            )
            
            train_df.to_csv(
                os.path.join(dst_dir, 'train.csv')
            )
        
        return train_df, test_df

               
               
    @staticmethod
    def iid_sampling(csv_path, criterion,
                     client_num,
                     dst_path='./clients', train_test_split_ratio=None):
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path, exist_ok=True)
        
        
        
        
        if train_test_split_ratio is not None:
            _parent_dir = Path(dst_path).resolve().parents[0]
            df, _ = fed_framework.__train_test_split(csv_path, _parent_dir, train_test_split_ratio)
        else:
            df = pd.read_csv(csv_path, header=None)
    
        
        criterion = np.array(criterion)
        
        label_num = len(np.unique(criterion))
        data_dict = [np.where(criterion == i)[0] for i in range(label_num)]
        
        selected_data = [np.random.choice(i, size=(client_num, int(len(i)/client_num)), replace=False) for i in data_dict]
        
        for i in range(client_num):
            f_name = os.path.join(dst_path, f'{i}.csv')
            data_per_client = [data_per_label[i] for data_per_label in selected_data]
            
            iid_df = df.iloc[
                np.array(list(chain.from_iterable(data_per_client))).reshape(-1),:
            ]
            iid_df.to_csv(f_name, index=False, header=False)
        
        
        
        
    
    @staticmethod
    def dirichelet_sampling(csv_path, criterion,
                            alpha,
                            client_num, 
                            dst_path='./clients', train_test_split_ratio=None):
        
        
        if train_test_split_ratio is not None:
            _parent_dir = Path(dst_path).resolve().parents[0]
            df, _ = fed_framework.__train_test_split(csv_path, _parent_dir, train_test_split_ratio)
        else:
            df = pd.read_csv(csv_path, header=None)
        
        
        criterion = np.array(criterion)
        data_num = len(criterion)
        label_num = len(np.unique(criterion))
        
        data_dict = [np.where(criterion == i)[0] for i in range(label_num)]
        
        
        # alpha parameter controls the variance of dirichlet distribution
        # the smaller the alpha, the larger the variance, i.e a larger degree of non-iid
        non_iid_sample = np.random.dirichlet(np.ones(label_num) * alpha, client_num)
        
        simplex = np.ones(client_num); simplex = simplex/np.sum(simplex)
        
        client_data_num = simplex * data_num; client_data_num = client_data_num.astype(int)
        
        
        
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path, exist_ok=True)
        
        
        for i in range(client_num):
            updated_data_dict = []
            selected_data = []
            
            
            f_name = os.path.join(dst_path, f'{i}.csv')
            num_ = client_data_num[i]
            num_per_label = [int(num_ * ratio) for ratio in non_iid_sample[i]]
            
            for idx, data_per_label in enumerate(data_dict):
                sampling = np.random.randint(0, len(data_dict), num_per_label[idx])
                selected_data.append(data_per_label[sampling])
                
                updated_data_dict.append(np.delete(data_per_label, sampling))
                
            # write to csv file
            
            non_iid_df = df.iloc[
                np.array(list(chain.from_iterable(selected_data))).reshape(-1),:
            ]
            non_iid_df.to_csv(f_name, index=False, header=False)
        
            data_dict = updated_data_dict

