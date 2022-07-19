from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Union

import numpy as np
import numpy.random as rnd
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from absl import logging

logging.set_verbosity(logging.INFO)

ListType = Union[List, np.ndarray]
TensorType = Union[pt.Tensor, pt.LongTensor]
TensorDict = MutableMapping[str, TensorType]


def activation_fn(act_fn: str) -> Callable[[TensorType], TensorType]:
    if act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'gelu':
        return nn.GELU()
    elif act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'tanh':
        return nn.Tanh()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softmax':
        return nn.Softmax(-1)
    else:
        raise NotImplementedError(act_fn)

def fix_seed(args: MutableMapping[str, Any]) -> None:
    if args.seed:
        rnd.seed(0)
        pt.manual_seed(0)

def get_len_block(config: Mapping[str, Any]) -> int:
    n_modalities = 0
    for key, value in config.module_specs.items():
        if value not in ['local_position', 'global_position']:
            n_modalities += 1
    len_block = int(config.len_context*n_modalities)
    return len_block

def loss_fn(loss_fn: str) -> Callable[[TensorType], TensorType]:
    if loss_fn == 'mse_loss':
        return F.mse_loss
    if loss_fn == 'binary_cross_entropy':
        return F.binary_cross_entropy
    if loss_fn == 'kl_div':
        return F.kl_div
    else:
        raise NotImplementedError(loss_fn)

def join_tensor(*x: TensorType, dim: Optional[int] = -1, mode: Optional[str] = 'cat'):
    '''
    Combine list of tensors along dimension dim, default mode = concatenate
    '''
    if mode == 'cat':
        tokens = pt.cat([*x], dim)
    elif mode == 'stack':
        tokens = pt.stack([*x], dim)
    else:
        raise NotImplementedError()
    return tokens