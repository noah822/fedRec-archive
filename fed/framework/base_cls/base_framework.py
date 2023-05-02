import os
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain

class Framework:
    def __init__(self):
        pass
        
    def __server_tb_logging(self, data):
        graph_name = 'server/Accuracy'
        self.writer.add_scalar(
            graph_name, data, self.cur_round
        )
    
    def __clients_tb_logging(self, data, cluster_size=5):
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
            df, _ = Framework.__train_test_split(csv_path, _parent_dir, train_test_split_ratio)
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
            df, _ = Framework.__train_test_split(csv_path, _parent_dir, train_test_split_ratio)
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

