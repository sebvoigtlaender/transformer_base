'''
The EmbedderCore is a module that tokenizes different types of data into a numerical embedding flexibly, as defined by the EmbedderCore configuration.
The EmbedderBank produces an dictionary of vectors that can be flexibly queried by the downstream models, as defined by the transformer configuration or by the model configuration
of the model that contains the transformer core. This flexibility will allow to tokenize various data and feed them selectively to specialized downstream models.
The hull which wraps the transformer core is responsible for all computations outside of the transformer, i.e. preprocessing such as tokenize, join, etc.,
as well as post processing. Another class will be responsible for the control flow that is independent of the core computations, that is, it is responsible for
training, evaluation, logging, saving state dictionaries, parameter optimization, etc.
'''

from absl import logging
from collections import OrderedDict
from typing import Any, Mapping, MutableMapping, Optional, Tuple, Union

import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

logging.set_verbosity(logging.INFO)

from utils import activation_fn, TensorDict, TensorType


class ConvEncoder(nn.Module):

    def __init__(self, config: Mapping[str, Any], 
                 state_channels: int, 
                 output_channels: int) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(nn.Conv2d(state_channels, 32, kernel_size = 8, stride = 4), activation_fn(self.config.conv_act_fn),
                                     nn.Conv2d(32, 64, kernel_size = 4, stride = 2), activation_fn(self.config.conv_act_fn),
                                     nn.Conv2d(64, 64, kernel_size = 3, stride = 1), activation_fn(self.config.conv_act_fn),
                                     nn.Flatten(), nn.Linear(3136, output_channels), activation_fn(self.config.ebd_act_fn))

    def forward(self, x: TensorType) -> TensorType:
        x = self.encoder(x)
        return x


class EmbedderBank(nn.Module):

    def __init__(self, config: Mapping[str, Any],
                 module_specs: Mapping[str, Union[str, Mapping[str, Any]]]) -> None:
        super().__init__()
        self.config = config
        self.embedder_bank = nn.ModuleDict()
        self.module_specs = module_specs
        if not 'local_position' in module_specs.keys():
            module_specs['local_position'] = 'local_position'

        module_dict = OrderedDict()
        for key, spec in module_specs.items():
            if spec == 'global_position' or spec == 'local_position':
                assert key == spec, f'for {spec} it is required that key = {spec} for later retrieval'
            module_dict[key] = self._make_layer(spec)
        self.embedder_bank.update(module_dict)

    def _make_layer(self, input_type: str) -> nn.Module:
        if input_type == 'continuous':
            return nn.Sequential(nn.Linear(self.config.vocab_size, self.config.d), activation_fn(self.config.ebd_act_fn))
        elif input_type == 'discrete':
            return nn.Sequential(nn.Embedding(self.config.vocab_size, self.config.d), activation_fn(self.config.ebd_act_fn))
        elif input_type == 'global_position':
            return nn.Embedding(self.config.len_trajectory, self.config.d)
        elif input_type == 'local_position':
            return nn.Embedding(self.config.len_context, self.config.d)
        elif input_type == 'scalar':
            return nn.Sequential(nn.Linear(1, self.config.d), activation_fn(self.config.ebd_act_fn))
        elif input_type == 'visual':
            return ConvEncoder(self.config, self.config.state_channels, self.config.d)
        else:
            raise NotImplementedError('input_type must be in {continuous, discrete, global_position, local_position, scalar, visual}')

    def _register(self, module_key: str, module: nn.Module) -> None:
        if not module_key in self.embedder_bank:
            assert type(module_key) == str
            self.embedder_bank[key] = module
        else:
            logging.info(f'{module_key} already exists')

    def tokenize(self, x: TensorDict) -> TensorDict:
        '''
        tokenize() returns a token_dict that only contains values that are not None
        '''
        token_dict = OrderedDict()
        position_dict = OrderedDict()
        for key, value in x.items():
            assert key in self.embedder_bank.keys(), f'the input keys must be contained in {self.embedder_bank.keys()}'
            if value is not None:
                if key in ['global_position', 'local_position']:
                    position_dict[key] = self.embedder_bank[key](value)
                else:
                    token_dict[key] = self.embedder_bank[key](value)
        return token_dict, position_dict

    def join_tokens(self, token_dict: TensorDict,
                    position_dict: TensorDict, 
                    targets: Optional[TensorType] = None) -> TensorType:
        '''
        self.model_type == 'reward_conditioned' means that the token embeddings contain return to go -
        in this implementation that simply means that return to go is a key as specified by the config

        join_tokens() then concatenates the non-None tokens along the second dimension
        '''

        assert position_dict is not None, 'position_dict cannot be None'
        n_inputs_not_null = len(token_dict)
        token_embeddings = []

        for key in token_dict.keys():

            assert token_dict[key].shape[-1] == self.config.d

            if token_dict[key].shape == (self.config.batch_size*self.config.len_context, self.config.d): 
                token_dict[key] = token_dict[key].view(self.config.batch_size, self.config.len_context, self.config.d)

            for position_key in position_dict.keys():
                token_embeddings.append(token_dict[key] + position_dict[position_key])
                
        if not hasattr(self.config, 'join_mode'):
            self.config.join_mode = 'interleave'
            logging.info(f'self.config.join_mode not given, set to {self.config.join_mode}')

        if self.config.join_mode == 'interleave':
            token_embeddings = pt.reshape(pt.stack(token_embeddings, -1).flatten(),
                                          (self.config.batch_size, self.config.len_context*n_inputs_not_null, self.config.d))

        elif self.config.join_mode == 'block':
            token_embeddings = pt.reshape(pt.stack(token_embeddings, 1).flatten(),
                              (self.config.batch_size, self.config.len_context*n_inputs_not_null, self.config.d))

        return token_embeddings


class PredictorBank(nn.Module):

    def __init__(self, config: Mapping[str, Any],
                 module_specs: Mapping[str, Union[str, Mapping[str, Any]]]) -> None:
        super().__init__()
        self.config = config
        self.predictor_bank = nn.ModuleDict()
        self.module_specs = module_specs

        module_dict = OrderedDict()
        for key, spec in module_specs.items():
            if not spec == 'global_position' and not spec == 'local_position':
                module_dict[key] = self._make_layer(spec)
        self.predictor_bank.update(module_dict)

    def _make_layer(self, output_type: str) -> nn.Module:
        if not output_type == 'local_position' or output_type == 'global_position':
            if output_type == 'continuous':
                return nn.Sequential(nn.Linear(self.config.d, self.config.vocab_size), activation_fn('softmax'))
            elif output_type == 'discrete':
                return nn.Sequential(nn.Linear(self.config.d, 1), activation_fn('softmax'))
            elif output_type == 'scalar':
                return nn.Sequential(nn.Linear(self.config.d, 1), activation_fn(self.config.output_act_fn))
            elif output_type == 'visual':
                raise NotImplementedError()
                # return ConvDecoder(self.config, self.config.d, self.config.state_channels)
            else:
                raise NotImplementedError('output_type must be in {continuous, discrete, scalar, visual}')

    def _register(self, module_key: str, module: nn.Module) -> None:
        if not module_key in self.predictor_bank:
            assert type(module_key) == str
            self.predictor_bank[key] = module
        else:
            logging.info(f'{module_key} already exists')

    def predict(self, x: TensorType) -> TensorDict:
        '''
        predict() returns a prediction_dict that contains one prediction for every modality
        the loop is too restrictive for certain applications since the value, the predictor,
        for a certain modality is only constructed from the embedding of the same modality
        '''
        prediction_dict = OrderedDict()
        n_modalities = int(x.shape[1]//self.config.len_context)
        x = x.reshape((n_modalities, self.config.batch_size, self.config.len_context, self.config.d))
        assert len(x) == len(self.predictor_bank.keys())
        for i in range(len(x)):
            key = list(self.predictor_bank.keys())[i]
            value = x[i]
            prediction_dict[key] = self.predictor_bank[key](value)
        return prediction_dict