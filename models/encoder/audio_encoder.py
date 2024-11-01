# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base_encoder import BaseEncoderModel
from utils.keras_utils import TensorSpec, ops
from models.interfaces.base_audio_model import BaseAudioModel

MIN_AUDIO_TIME      = 0.1 # below 0.1sec the encoding is not really relevant

DEFAULT_AUDIO_RATE      = 16000
DEFAULT_MAX_AUDIO_TIME  = 3

class AudioEncoder(BaseAudioModel, BaseEncoderModel):
    prepare_input   = BaseAudioModel.get_audio
    augment_input   = BaseAudioModel.augment_audio
    
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
    
    def build(self, 
              depth             = 128,
              encoder_type      = 'conv1d',
              flatten_type      = 'max',
              final_activation  = None,
              
              embedding_dim     = 128, 
              flatten_kwargs    = {},
              normalize         = None,
              
              model = None,
              
              ** kwargs
             ):
        if model is not None: return super().build(model = model)
        
        if normalize is None: normalize = self.distance_metric == 'euclidian'
        if normalize:
            final_activation = 'l2' if not final_activation else [final_activation, 'l2']
        
        flatten_kwargs.setdefault('units', embedding_dim)
        voicemap_encoder_config = {
            'architecture'  : 'simple_cnn', 
            'input_shape'   : self.input_signature.shape[1:],
            'output_shape'  : embedding_dim,
            'final_activation'  : final_activation,
            
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
            'drop_rate'     : 0.1,
            'conv_type'     : 'conv1d',
            'flatten'       : True,
            'flatten_type'  : flatten_type,
            'flatten_kwargs'    : flatten_kwargs,
            'dense_as_final'    : True,
            'name'  : 'encoder',
            ** kwargs
        }
        mel_encoder_config = {
            'architecture'  : 'simple_cnn', 
            'input_shape'   : self.input_signature.shape[1:],
            'output_shape'  : embedding_dim,
            'final_activation'  : final_activation,

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
            'bnorm' : ['after', 'after', 'after', 'after'],
            'residual'      : False,
            'conv_type'     : encoder_type,
            'flatten'       : True,
            'flatten_type'  : flatten_type,
            'flatten_kwargs'    : flatten_kwargs,
            'dense_as_final'    : flatten_type not in ('lstm', 'gru'),
            'name'  : 'encoder',
            ** kwargs
        }
        
        config = mel_encoder_config if self.use_mel_fn else voicemap_encoder_config
        
        return super().build(model = config)

    @property
    def pad_value(self):
        return self.pad_mel_value if self.use_mel_fn else 0.
    
    @property
    def input_signature(self):
        length = None if not self.use_fixed_length_input else self.max_input_length
        
        if not self.use_mel_fn:
            shape = (None, length, 1)
        elif not self.mel_as_image:
            shape = (None, length, self.n_mel_channels)
        else:
            shape = (None, length, self.n_mel_channels, 1)
        
        return TensorSpec(shape = shape, dtype = 'float')
    
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
    
    def process_input(self, input_data, ** kwargs):
        if isinstance(input_data, list):
            return [self.prepare_input(data_i, ** kwargs) for data_i in input_data]

        if ops.shape(input_data)[0] > self.max_input_length:
            start = ops.random.randint(
                (),
                minval = 0, 
                maxval = ops.shape(input_data)[0] - self.max_input_length,
            )
            input_data = input_data[start : start + self.max_input_length]
        
        return input_data
    
    def get_dataset_config(self, mode, ** kwargs):
        if self.use_fixed_length_input:
            kwargs['pad_kwargs'] = {
                'padded_shapes' : (self.input_signature.shape[1:], ())
            }
        
        return super().get_dataset_config(mode, ** kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            ** self.get_config_audio(),
            'max_audio_time'    : self.max_audio_time,
            'use_fixed_length_input'    : self.use_fixed_length_input
        })
            
        return config
