from collections import OrderedDict
from typing import Any, Mapping, MutableMapping, Optional, Tuple, Union

import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from absl import logging

from core import Core, SelfAttention
from ops import EmbedderBank, PredictorBank
from utils import activation_fn, get_len_block, TensorDict, TensorType

logging.set_verbosity(logging.INFO)


def get_vision_transformer_config(args: MutableMapping[str, Any]) -> Mapping[str, Any]:

    parser = argparse.ArgumentParser()
    config = parser.parse_args(args=[])

    config.batch_size = args.batch_size
    config.device = args.device


class VisionTransformer(nn.Module):

    def __init__(self, config: Mapping[str, Any],
                 output_dir: Optional[str] = None) -> None:
        super().__init__()

        len_block = get_len_block(config)
        config.len_block = len_block

        self.embedder_bank = EmbedderBank(config, config.module_specs)
        transformer_core = [Core(config) for n_layers in range(config.n_core_layers)]
        self.transformer_core = nn.Sequential(*transformer_core,  nn.LayerNorm(config.d), nn.Linear(config.d, config.d, bias = False))
        self.predictor_bank = PredictorBank(config, config.module_specs)

        self.config = config

    def forward(self, x: TensorDict) -> TensorDict:
        token_dict, position_dict = self.embedder_bank.tokenize(x)
        tf_input = self.embedder_bank.join_tokens(token_dict, position_dict)
        tf_output = self.transformer_core(tf_input)
        prediction_dict = self.predictor_bank.predict(tf_output)
        return prediction_dict

    def tokenize(self, x: TensorDict) -> Tuple[TensorDict]:
        token_dict, position_dict = self.embedder_bank.tokenize(x)
        return token_dict, position_dict

    def tf_output(self, token_dict: TensorDict, position_dict: TensorDict) -> TensorType:
        tf_input = self.embedder_bank.join_tokens(token_dict, position_dict)
        tf_output = self.transformer_core(tf_input)
        return tf_output

    def predict(self, tf_output: TensorType) -> TensorDict:
        '''
        can be used to override self.predictor_bank.predict -
        default:
        prediction_dict = self.predictor_bank.predict(tf_output)

        in that case, the forward function also has to be rewritten, e.g.:

        token_dict, position_dict = transformer.embedder_bank.tokenize(x)
        tf_input = transformer.embedder_bank.join_tokens(token_dict, position_dict)
        tf_output = transformer.transformer_core(tf_input)
        prediction_dict = transformer.predict(tf_output)

        alternatively, the ops module has to be customized slightly 
        '''

        prediction_dict = OrderedDict()
        n_modalities = int(tf_output.shape[1]//self.config.len_context)
        tf_output = tf_output.reshape((n_modalities, self.config.batch_size, self.config.len_context, self.config.d))
        assert len(tf_output) == len(self.predictor_bank.predictor_bank.keys())
        for i in range(len(tf_output)):
            key = list(self.predictor_bank.predictor_bank.keys())[i]
            value = tf_output[i]
            prediction_dict[key] = self.predictor_bank.predictor_bank[key](value)

        return prediction_dict

    def compute_loss(self, x: TensorType, targets: TensorType = None) -> TensorType:
        loss = None
        if targets is not None:
            pass
        return x, loss

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)