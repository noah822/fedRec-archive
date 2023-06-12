import numpy as np
import pandas as pd
from typing import List
import os
from pathlib import Path
from itertools import chain

def _pipeline(csv_path,
            header, index_col,
            criterion,
            num_client,
            sampler,
            train_test_split_ratio=None,
            dst_path='./clients'):
    '''
    sample from rows of csv file specified by `csv_path` without replacement
    Args:
    - csv_path: path of csv file for the dataset to be splitted
    - criterion: List of label for each row of the csv file
      the function will then sample from dataset according to the labels
    - num_client: number of clients, i.e the number of chunk dataset will be splitted into
    - seed: random seed for reproductivity
    - train_test_split_ratio: if set to None, only train set will be generated
    - dst_path: name of the folder the generated csv file storing the sample result will be stored into
      the sampling result will be stored in the following structure by default
      
    $dst_path        
    └── train
        ├── 0.csv
        ├──  ⋮
        └── {num_client-1}.csv
    └── test         if `train_test_split_ratio` is specified
        ├── 0.csv
        ├──  ⋮
        └── {num_client-1}.csv
    '''

    df = pd.read_csv(csv_path, header=header, index_col=index_col)

    # save_paths = None
    datasets = _unified_sample_interface(
        df, 
        num_client,
        sampler,
        criterion
    )
    
    res = []
    save_paths = None
    if train_test_split_ratio is not None:
        for client_ds in datasets:
            train_df, test_df = _split_df(client_ds, train_test_split_ratio)
            res.append([train_df, test_df])

        res = _transpose_2d_list(res)
        save_paths = ['train', 'test']
    else:
        res.append(datasets)
        save_paths = ['train']
    
    # create folders if not exist
    for path in save_paths:
        folder_path = f'{dst_path}/{path}'
        Path(folder_path).mkdir(parents=True, exist_ok=False)
    
    for datasets, path in zip(res, save_paths):
        full_path = os.path.join(dst_path, path)
        _saver(datasets, full_path)


def _transpose_2d_list(x: List):
    transposed = [list(v) for v in zip(*x)]
    return transposed

def _split_df(df: pd.DataFrame, split_ratio):
    N = df.shape[0]
    chunk_a = int(split_ratio * N)
    row_idx = np.arange(N)
    mask_a = row_idx[:chunk_a]; mask_b = row_idx[chunk_a:]

    return df.iloc[mask_a,:], df.iloc[mask_b,:]

def _saver(df_array: List[pd.DataFrame], prefix):
    for i, df in enumerate(df_array):
        path = f'{prefix}/{i}.csv'
        df.to_csv(path, index=False, header=None)

def _unified_sample_interface(
        df:pd.DataFrame,
        num_split: int,
        sampler: callable,
        criterion: np.ndarray
    ) -> List[pd.DataFrame]:
    sampled = sampler(num_split, criterion)
    dfs = []
    for row_idx in sampled:
        dfs.append(df.iloc[row_idx,:])
    return dfs

'''
    IID sampling
'''

def _iid_sampler(num_split: int, arr: np.ndarray) -> List[np.ndarray]:
    labels = np.unique(arr)
    num_class = len(labels)
    data_dict = [np.where(arr == i)[0] for i in labels]
    selected_data = [
        np.random.choice(i, size=(num_split, int(len(i)/num_split)), replace=False) for i in data_dict
    ]

    res = []
    for i in range(num_class):
        data_per_client = [data_per_label[i] for data_per_label in selected_data]
        res.append(
            np.array(list(chain.from_iterable(data_per_client))).reshape(-1)
        )

    return res
        



def iid_sampling(csv_path, criterion,
                 num_client,
                 header=0,
                 index_col=0,
                 seed=None,
                 train_test_split_ratio=None,
                 dst_path='./clients'):

    if seed is not None:
        np.random.seed(seed)

    _pipeline(
        csv_path,
        header,
        index_col,
        criterion,
        num_client,
        _iid_sampler,
        train_test_split_ratio,
        dst_path
    )



def _dirichelet_sampler(num_split: int, arr: np.ndarray, alpha):
    N = arr.shape[0]
    labels = np.unique(arr)

    data_dict = [np.where(arr == i)[0] for i in labels]

    non_iid_sample = np.random.dirichlet(np.ones(num_split) * alpha, num_split)
    
    simplex = np.ones(num_split); simplex = simplex/np.sum(simplex)
    
    client_data_num = simplex * N; client_data_num = client_data_num.astype(int)

    res = []
    for i in range(num_split):
        updated_data_dict = []
        selected_data = []
        
        num_ = client_data_num[i]
        num_per_label = [int(num_ * ratio) for ratio in non_iid_sample[i]]
        
        for idx, data_per_label in enumerate(data_dict):
            sampling = np.random.randint(0, len(data_dict), num_per_label[idx])
            selected_data.append(data_per_label[sampling])
            
            updated_data_dict.append(np.delete(data_per_label, sampling))
        
        res.append(
            np.array(list(chain.from_iterable(selected_data))).reshape(-1)
        )
    
        data_dict = updated_data_dict
    return res


def non_iid_sampling(*args, **kwargs):
    return dirichelet_sampling(*args, **kwargs)

def dirichelet_sampling(csv_path, criterion,
                 num_client,
                 alpha=1.,
                 header=0,
                 index_col=0,
                 seed=None,
                 train_test_split_ratio=None,
                 dst_path='./clients'):

    def wrapped_sampler(num_split, arr):
        return _dirichelet_sampler(
            num_split, arr, 
            alpha=alpha
        )
    
    if seed is not None:
        np.random.seed(seed)

    _pipeline(
        csv_path,
        header,
        index_col,
        criterion,
        num_client,
        wrapped_sampler,
        train_test_split_ratio,
        dst_path
    )