import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Tuple
from functools import reduce

__all__ = ['fedAvg']


def _test_whether_exclude(param_name: str, exclude_list: List[str]) -> bool:
    return any([name in param_name for name in exclude_list])


def _mul_state_dict(
        x: OrderedDict, w: float,
        exclude_from_mul: List[str]=[]
) -> OrderedDict:
    for param_name, weight in x.items():
        if _test_whether_exclude(param_name, exclude_from_mul):
            continue
        else:
            weight.mul_(w)
    return x

def _add_state_dict(
        x: OrderedDict, y: OrderedDict,
        exclude_from_add: List[str]=[],
) -> OrderedDict:
    for param_name, weight in x.items():
        if _test_whether_exclude(param_name, exclude_from_add):
            continue
        else:
            weight.add_(y[param_name])
    return x


def fedAvg(
        results: List[Tuple[OrderedDict, float]],
        exclude_from_aggregate: List[str]=[]
) -> OrderedDict:
    num_examples_total = sum([num_examples for _, num_examples in results])
    weighted_weights = [
        _mul_state_dict(state_dict, w/num_examples_total, exclude_from_aggregate) for state_dict, w in results 
    ]

    def _wrapped_aggregate(x, y):
        return _add_state_dict(x, y, exclude_from_aggregate)

    aggregated_weights = reduce(_wrapped_aggregate, weighted_weights)

    return aggregated_weights


