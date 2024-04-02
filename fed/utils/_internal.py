from functools import partial
import torch
import torch.nn as nn

from typing import (
    List,
    Dict
)

def _infer_device(x):
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).device
    else:
        raise NotImplementedError
    
# iterate elements in dict by ascending order of keys
def _iter_dict(x: dict, exclude_list: List=None):
    if exclude_list is None:
        exclude_list = []
    ordered_list = []
    for k in sorted(x.keys()):
        if k in exclude_list:
            continue
        ordered_list.append(x[k])
    return ordered_list
    
def _zippable(x):
    if isinstance(x, tuple) or \
       isinstance(x, list):
        return x
    else:
        return (x, )

def _transpose_list(x):
    return [list(y) for y in zip(*x)]

'''
    check whehter expected keys are provided in the dict instance passed in 
'''
def _check_keys(x: Dict, expected: List):
    return all([key in x.keys() for key in expected])

def _to_numpy(t):
    return t.contiguous().detach().cpu().numpy()


def pairwise_sim_metrics(x: torch.Tensor,
                         y: torch.Tensor,
                         metrics: str='Euclidean'):
    '''
    Args:
    - x: of shape (n, d)
    - y: of shape (m, d)

    Return:
    - pairwise distance of shape (n, m)
    '''
    broadcasted_pairwise_diff = x.unsqueeze(-2) - y.unsqueeze(0)
    if metrics == 'Euclidean':
        raise_fn = lambda x: x**2
        aggre_fn = partial(torch.sum, dim=-1)
    else:
        raise NotImplementedError(f'{metrics} aggregation scheme is not supported')
    pairwise_sim = aggre_fn(raise_fn(broadcasted_pairwise_diff))
    return pairwise_sim