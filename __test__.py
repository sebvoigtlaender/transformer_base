from typing import Any, Mapping
import numpy.random as rnd
import torch
import torch as pt
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from arguments import get_args
from vision_transformer import VisionTransformer
from utils import activation_fn, get_len_block, join_tensor, TensorDict, TensorType


train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)



config = get_args()
config.batch_size = 5
config.n_heads = 4
config.d = 8
config.p_dropout = 0.2
config.causal = False
config.core_act_fn = 'gelu'
config.n_core_layers = 4
config.device = 'cpu'
config.conv_act_fn = 'relu'
config.output_act_fn = 'softmax'
config.ebd_act_fn = 'gelu'
config.core_type = 'tf_encoder'
config.len_trajectory = 5
config.module_specs = {'patches': 'continuous'}


data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=config.batch_size,
                                          shuffle=True)
x, label = next(iter(data_loader))

h_image = 28
h_patch = 4
len_context = h_image*h_image//(h_patch*h_patch) + 1

config.vocab_size = h_patch*h_patch
config.len_context = len_context

dictionary = {0: pt.ones(2, 3, 4), 1: pt.ones(2, 1, 4)}
#all(value.shape[1] in dictionary.values)

# class_token = pt.ones(config.batch_size, 1, )
# patches = x.flatten(1, -1).reshape(config.batch_size, config.len_context, h_patch*h_patch)
# print(patches.shape)

# vision_transformer = VisionTransformer(config)
# x = {'patches': patches, }

# print(vision_transformer.embedder_bank)
# tf_output = vision_transformer(x)
# print(tf_output)