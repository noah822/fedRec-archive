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
    