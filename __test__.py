from typing import Any, Mapping
import numpy.random as rnd
import torch
import torch as pt
import torch.nn as nn

from arguments import get_args
from model import DecisionTransformer
from utils import activation_fn, get_len_block, TensorDict, TensorType

config = get_args()
config.batch_size = 2
config.len_context = 4
config.vocab_size = 3
config.n_actions = config.vocab_size
config.n_heads = 2
config.d = 4
config.p_dropout = 0.2
config.causal = True
config.core_act_fn = 'tanh'
config.n_core_layers = 4
config.device = 'cpu'
config.conv_act_fn = 'relu'
config.output_act_fn = 'softmax'
config.ebd_act_fn = 'tanh'
config.core_type = 'tf_encoder'
config.len_trajectory = 5
config.module_specs = {'actions': 'continuous'}

transformer = DecisionTransformer(config)

x = {'actions': pt.rand(config.batch_size, config.len_context, config.n_actions), 'local_position': pt.randint(0, config.len_context, (config.batch_size, config.len_context))}



pip3 install torch-tensorrt -f https://github.com/pytorch/TensorRT/releases
