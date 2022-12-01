
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from custom_architectures import get_architecture
from models.siamese.base_encoder import BaseEncoderModel
from models.interfaces.base_audio_model import BaseAudioModel
from utils.audio import load_audio

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

MIN_AUDIO_TIME      = 0.1 # below 0.1sec the encoding is not really relevant

DEFAULT_AUDIO_RATE      = 16000
DEFAULT_MAX_AUDIO_TIME  = 3

class AudioEncoder(BaseAudioModel, BaseEncoderModel):
    def __init__(self,
                 audio_rate     = DEFAULT_AUDIO_RATE,
                 
                 max_audio_time     = DEFAULT_MAX_AUDIO_TIME,
                 use_fixed_length_input = False,
                 
                 ** kwargs
                ):
        self._init_audio(audio_rate = audio_rate, ** kwargs)
        
        self.max_audio_time     = max_audio_time
        self.use_fixed_length_input = use_fixed_length_input
        
        super().__init__(** kwargs)
    
    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.identification_dir, exist_ok = True)

    def build_encoder(self, 
                      depth             = 128,
                      encoder_type      = 'conv1d',
                      flatten_type      = 'max',
                      embedding_dim     = 128, 
                      flatten_kwargs    = {},
                      normalize         = None,
                      **kwargs
                     ):
        flatten_kwargs.setdefault('units', embedding_dim)
        audio_encoder_config = {
            'architecture_name'     : 'simple_cnn', 
            'input_shape'   : self.encoder_input_shape[1:],
            'output_shape'  : embedding_dim,
            'n_conv'    : 5,
            'filters'   : [
                depth, 
                depth * 2,
                depth * 3, 
                [depth * 3, depth * 4],
                [depth * 4, depth * 4],
            ],
            'strides'   : [4, 4, 2, 2, 1],
            'kernel_size'   : [
                32, 32, 3, 3, [3, 1]
            ],
            'residual'      : True,
            'drop_rate'     : 0.05,
            'conv_type'     : encoder_type,
            'flatten'       : True,
            'dense_as_final'    : True,
            'name'  : 'Encoder'
        }
        voicemap_encoder_config = {
            'architecture_name'     : 'simple_cnn', 
            'input_shape'   : self.encoder_input_shape[1:],
            'output_shape'  : embedding_dim,
            'n_conv'    : 4,
            'filters'   : [
                depth, 
                depth * 2,
                depth * 3, 
                depth * 4
            ],
            'strides'   : [4, 1, 1, 1],
            'kernel_size'   : [
                32, 3, 3, 3
            ],
            'pooling'       : 'max',
            'pool_size'     : [4, 2, 2, 2],
            'pool_strides'  : [4, 2, 2, 2],
            'residual'      : False,
            'activation'    : 'relu',
            'drop_rate'     : 0.05,
            'conv_type'     : 'conv1d',
            'flatten'       : True,
            'flatten_type'  : flatten_type,
            'flatten_kwargs'    : flatten_kwargs,
            'dense_as_final'    : True,
            'name'  : 'Encoder',
            ** kwargs
        }
        mel_encoder_config = {
            'architecture_name'     : 'simple_cnn', 
            'input_shape'   : self.encoder_input_shape[1:],
            'output_shape'  : embedding_dim,
            'n_conv'    : 4,
            'filters'   : [
                depth, 
                depth * 2, 
                [depth * 2, depth * 4],
                [depth * 4, depth * 4],
            ],
            'strides'   : [2, 2, 1, 1],
            'kernel_size'   : [
                7, 5, 3, [3, 1]
            ],
            'residual'      : False,
            'conv_type'     : encoder_type,
            'flatten'       : True,
            'flatten_type'  : flatten_type,
            'flatten_kwargs'    : flatten_kwargs,
            'dense_as_final'    : flatten_type not in ('lstm', 'gru'),
            'name'  : 'Encoder',
            ** kwargs
        }
        
        encoder_config = mel_encoder_config if self.use_mel_fn else voicemap_encoder_config
        
        return get_architecture(** encoder_config)

    @property
    def identification_dir(self):
        return os.path.join(self.folder, 'identification')
    
    @property
    def encoder_input_shape(self):
        length = None if not self.use_fixed_length_input else self.max_input_length
        
        if not self.use_mel_fn:
            shape = (None, length, 1)
        elif not self.mel_as_image:
            shape = (None, length, self.n_mel_channels)
        else:
            shape = (None, length, self.n_mel_channels, 1)
        return shape
    
    @property
    def min_input_length(self):
        if self.use_mel_fn:
            return self.mel_fn.get_length(int(MIN_AUDIO_TIME * self.audio_rate))
        else:
            return int(MIN_AUDIO_TIME * self.audio_rate)
        
    @property
    def max_input_length(self):
        if self.use_mel_fn:
            return self.mel_fn.get_length(int(self.max_audio_time * self.audio_rate))
        else:
            return int(self.max_audio_time * self.audio_rate)
                
    @property
    def training_hparams(self):
        return super().training_hparams(** self.training_hparams_audio)
        
    def __str__(self):
        return super().__str__() + self._str_audio()
    
    def get_input(self, data, ** kwargs):
        if isinstance(data, list):
            return [self.get_input(data_i, ** kwargs) for data_i in data]
        elif isinstance(data, pd.DataFrame):
            return [self.get_input(row, ** kwargs) for idx, row in data.iterrows()]
        
        input_data = self.get_audio(data)
        
        if tf.shape(input_data)[0] > self.max_input_length:
            start = tf.random.uniform(
                (), minval = 0, 
                maxval = tf.shape(input_data)[0] - self.max_input_length,
                dtype = tf.int32
            )
            input_data = input_data[start : start + self.max_input_length]
        
        return input_data
    
    
    def augment_input(self, inp):
        return self.augment_audio(inp)
    
    def get_dataset_config(self, ** kwargs):
        kwargs.update({'pad_kwargs' : {}, 'padded_batch' : True})
        if self.use_fixed_length_input:
            input_shape = self.encoder_input_shape
            kwargs['pad_kwargs'] = {
                'padded_shapes' : (input_shape[1:], ())
            }
        
        return super().get_dataset_config(** kwargs)
                        
    def get_config(self, *args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_audio(),
            'max_audio_time'    : self.max_audio_time,
            'use_fixed_length_input'    : self.use_fixed_length_input
        })
            
        return config
