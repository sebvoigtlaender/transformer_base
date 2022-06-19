from typing import Any, Mapping, MutableMapping, Optional, Tuple, Union

import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from absl import logging

from utils import activation_fn, get_len_block, TensorDict, TensorType 

logging.set_verbosity(logging.INFO)


class SelfAttention(nn.Module): 

    def __init__(self, 
                 d: int,
                 n_heads: int, 
                 len_block: Optional[int] = None, 
                 p_dropout: Optional[float] = 0.0, 
                 bias: Optional[bool] = True, 
                 batch_first: Optional[bool] = True, 
                 causal: Optional[bool] = False, 
                 norm: Optional[bool] = True) -> None:

        super().__init__()
        self.causal = causal
        self.norm = norm
        self.self_attention = nn.MultiheadAttention(d, n_heads, dropout=p_dropout, bias=bias, batch_first=batch_first)
        if norm:
            self.layer_norm = nn.LayerNorm(d)
        if len_block is not None:
            self.causal = True
            self.causal_mask = pt.triu(pt.ones(len_block, len_block, dtype=pt.bool), diagonal=1)

    def forward(self, x: TensorType, causal_mask: Optional[TensorType] = None) -> TensorType: # maybe more dropout here

        if causal_mask is not None: 
            self.causal = True
            x = x + self.self_attention(x, x, x, attn_mask=causal_mask)[0]
        else:
            if self.causal:
                x = x + self.self_attention(x, x, x, attn_mask=self.causal_mask)[0]
            elif not self.causal:
                x = x + self.self_attention(x, x, x)[0]
        if self.norm:
            x = self.layer_norm(x)
        return x


class Core(nn.Module):

    def __init__(self,
                 config: Mapping[str, Any], 
                 ) -> None: 
        super().__init__()
        self.config = config
        self.core_type = config.core_type
        len_block = get_len_block(config)

        if self.core_type == 'tf_encoder':
            self.core_layers = nn.Sequential(SelfAttention(config.d, config.n_heads, len_block, p_dropout=config.p_dropout, causal = False))
        elif self.core_type == 'tf_decoder':
            self.core_layer = SelfAttention(config.d, config.n_heads, len_block, p_dropout=config.p_dropout, causal = False)
            self.causal_core_layer = SelfAttention(config.d, config.n_heads, len_block, p_dropout=config.p_dropout, causal = True)
        elif self.core_type == 'custom_tf_core':
            core_layers = []
            for n in range(config.n_core_layers):
                assert hasattr(config, 'causal'), 'config.causal must be in {True, False}'
                core_layers.append(SelfAttention(config.d, config.n_heads, len_block, p_dropout=config.p_dropout, causal = config.causal))
            self.core_layers = nn.Sequential(*core_layers)
        else:
            raise NotImplementedError('core_type must be in {tf_encoder, tf_decoder, custom_tf_core}')

        self.fully_connected = nn.Sequential(
            nn.Linear(self.config.d, 4*self.config.d),
            activation_fn(self.config.core_act_fn),
            nn.Dropout(self.config.p_dropout),
            nn.Linear(4*self.config.d, self.config.d),
            nn.Dropout(self.config.p_dropout))

    def forward(self, x: TensorType, causal_mask: Optional[TensorType] = None) -> TensorType:
        if not self.core_type == 'tf_decoder':
            for core_layer in self.core_layers:
                x = core_layer(x, causal_mask)
        elif self.core_type == 'tf_decoder':
            x = self.core_layer(x, causal_mask)
            x = self.causal_core_layer(x,causal_mask)
        x = x + self.fully_connected(x)
        return x