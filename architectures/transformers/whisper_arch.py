# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
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
import keras
import numpy as np
import keras.ops as K

from ..simple_models import HParamsConvBN, simple_cnn
from .transformer_arch import *
from .text_transformer_arch import TextTransformer
from .gpt2_arch import GPT2, HParamsBaseGPT2

WHISPER_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
}

_whisper_loaded    = {}


HParamsWhisperEncoder = HParamsTransformerBlock(
    ** HParamsConvBN(
        use_mask    = False,
        n_conv  = 2,
        kernel_size = 3,
        padding = 'same',
        strides = [1, 2],
        bnorm   = 'never',
        activation  = 'gelu',
        final_activation = 'gelu'
    ),
    n_mel_channels  = 80,
    max_input_length    = 1500,
    normalize_output    = True,
    
    epsilon = 1e-5,
    mha_epsilon = 1e-5,
    mha_key_bias    = False,
    mha_mask_factor = -1.,
    
    normalize   = 'middle',
    mha_normalize   = False,
    mha_normalize_input = True,
    
    ffn_dim     = 2048,
    ffn_activation  = 'gelu'
)

HParamsWhisperDecoder   = HParamsBaseGPT2(
    use_encoder_attention   = True,
    use_causal_attention    = True,
    positional_offset   = 0,
    scale_embedding = False,
    
    normalize   = 'middle',
    mha_normalize   = False,
    enc_mha_normalize   = False,

    mha_mask_factor = -1.,
    enc_mha_mask_factor = -1.,
    
    mha_key_bias    = False,
    enc_mha_key_bias    = False,
    
    mha_normalize_input = True,
    enc_mha_normalize_input = True,
    enc_mha_epsilon = 1e-5,
    mha_epsilon = 1e-5,
    epsilon = 1e-5,
    
    ffn_dim = 2048,
    ffn_activation  = 'gelu'
)

def get_pos_embedding(max_length, embedding_dim, max_timescale = 10000, dtype = 'float32'):
    """ Returns sinusoids for positional embedding """
    assert embedding_dim % 2 == 0
    
    log_timescale_increment = K.convert_to_tensor(
        np.log(max_timescale) / (embedding_dim // 2 - 1), dtype
    )
    
    inv_timescales  = K.exp(-log_timescale_increment * K.arange(embedding_dim // 2, dtype = dtype))

    scaled_time = K.arange(max_length, dtype = dtype)[:, None] * inv_timescales[None, :]
    return K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis = 1)

@keras.saving.register_keras_serializable('whisper')
class WhisperEncoder(TransformerBlock):
    default_params  = HParamsWhisperEncoder
    _attr_to_set    = TransformerBlock._attr_to_set + [
        'n_mel_channels', 'max_input_length', 'use_mask'
    ]
    
    def __init__(self, n_mel_channels = 80, embedding_dim = 512, ** kwargs):
        super().__init__(
            n_mel_channels = n_mel_channels, embedding_dim = embedding_dim, ** kwargs
        )
        self.pos_encoding   = K.convert_to_tensor(get_pos_embedding(
            self.max_input_length, self.embedding_dim, dtype = 'float32'
        ), dtype = 'float32')

    def build(self, input_shape):
        self.feature_extractor  = simple_cnn(** self.hparams(
            use_sequential  = False,
            input_shape     = (None, self.n_mel_channels),
            output_shape    = self.embedding_dim,
            conv_type   = 'conv1d',
            use_mask    = self.hparams.use_mask,
            filters     = self.embedding_dim,
            strides     = [1, 2],
            padding     = 'same',
            activation  = self.hparams.activation,
            final_activation    = self.hparams.activation,
            flatten     = False,
            dense_as_final  = False,
            name    = 'feature_extractor'
        ))
        super().build((None, None, self.embedding_dim))


    def prepare_input(self, inputs, training = False, mask = None, ** kwargs):
        embedded = self.feature_extractor(inputs, training = training, mask = mask)
        if getattr(embedded, '_keras_mask', None) is not None:
            mask = embedded._keras_mask[:, None, None, :]
        else:
            mask = K.ones((K.shape(embedded)[0], 1, 1, K.shape(embedded)[1]), dtype = 'bool')

        embedded = embedded + K.cast(
            self.pos_encoding[None, :K.shape(embedded)[1]], embedded.dtype
        )
        try:
            embedded._keras_mask = mask
        except AttributeError:
            pass
        
        return embedded

    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import _transformer_patterns, name_based_partial_transfer_learning

        if isinstance(pretrained, dict) and any('encoder' in k for k in pretrained):
            pretrained = {k : v for k, v in pretrained.items() if 'encoder' in k}

        return name_based_partial_transfer_learning(
            self, pretrained, skip_root = False, patterns = {
                ** _transformer_patterns, 'blocks/' : 'layer_', '_ln' : '_norm',
                'ffn/0' : 'ffn/dense_1', 'ffn/2' : 'ffn/dense_2', 'mha_norm' : 'mha/norm_input',
                'final_layer_norm' : 'norm', 'layers/' : 'layer_'
            }, ** kwargs
        )
        
    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'medium',
                        pretrained      = None,
                        tqdm    = lambda x: x,
                        ** kwargs
                       ):
        pretrained = load_whisper(pretrained_name, pretrained = pretrained)
        pretrained, config = pretrained['model_state_dict'], pretrained['dims']
        
        config = cls.default_params(
            n_mel_channels  = config['n_mels'],
            embedding_dim   = config['n_audio_state'],
            max_input_length    = config['n_audio_ctx'],
            
            num_layers  = config['n_audio_layer'],
            mha_num_heads   = config['n_audio_head'],
            ffn_dim     = config['n_audio_state'] * 4
        )

        instance = cls(** config(** kwargs))
        instance.build((None, None, config['n_mel_channels']))

        instance.transfer_weights(pretrained, tqdm = tqdm, ** kwargs)

        return instance

class WhisperDecoder(GPT2):
    default_params  = HParamsWhisperDecoder

    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import _transformer_patterns, name_based_partial_transfer_learning

        if isinstance(pretrained, dict):
            pretrained = {k : v for k, v in pretrained.items() if 'decoder' in k}
        
        return name_based_partial_transfer_learning(
            self, pretrained, skip_root = False, patterns = {
                ** _transformer_patterns, 'blocks/' : 'layer_', '_ln' : '_norm',
                'ffn/0' : 'ffn/dense_1', 'ffn/2' : 'ffn/dense_2',
                '(attn|mha)_norm' : 'mha/norm_input', 'cross_' : 'enc_', 'enc_attn' : 'enc_mha',
                'final_layer_norm' : 'norm', 'layers/' : 'layer_'
            }, ** kwargs
        )
        
    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'medium',
                        pretrained      = None,
                        tqdm    = lambda x: x,
                        ** kwargs
                       ):
        pretrained = load_whisper(pretrained_name, pretrained = pretrained)
        pretrained, config = pretrained['model_state_dict'], pretrained['dims']
        
        config = cls.default_params(
            vocab_size  = config['n_vocab'],
            embedding_dim   = config['n_text_state'],
            max_input_length    = config['n_text_ctx'],
            
            num_layers  = config['n_text_layer'],
            ffn_dim     = config['n_text_state'] * 4,
            mha_num_heads   = config['n_text_head']
        )

        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(pretrained, tqdm = tqdm, ** kwargs)

        return instance

class Whisper(TextTransformer):
    encoder_class   = WhisperEncoder
    decoder_class   = WhisperDecoder
    
    _shared_keys    = TextTransformer._shared_keys + ['n_mel_channels']
    
    def __init__(self, * args, ** kwargs):
        super().__init__(* args, ** kwargs)
    
    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'medium',
                        pretrained      = None,
                        tqdm    = lambda x: x,
                        ** kwargs
                       ):
        pretrained = load_whisper(pretrained_name, pretrained = pretrained)
        pretrained, config = pretrained['model_state_dict'], pretrained['dims']
        
        config = cls.default_params(
            vocab_size      = config['n_vocab'],
            n_mel_channels  = config['n_mels'],
            
            encoder_embedding_dim   = config['n_audio_state'],
            encoder_max_input_length    = config['n_audio_ctx'],
            encoder_num_layers  = config['n_audio_layer'],
            encoder_mha_num_heads   = config['n_audio_head'],
            encoder_ffn_dim     = config['n_audio_state'] * 4,

            decoder_embedding_dim   = config['n_text_state'],
            decoder_max_input_length    = config['n_text_ctx'],
            decoder_num_layers  = config['n_text_layer'],
            decoder_mha_num_heads   = config['n_text_head'],
            decoder_ffn_dim     = config['n_text_state'] * 4
        )

        instance = cls(** config(** kwargs))
        instance.build([(None, None, config.n_mel_channels), (None, None)])

        instance.encoder.transfer_weights(pretrained, tqdm = tqdm, ** kwargs)
        instance.decoder.transfer_weights(pretrained, tqdm = tqdm, ** kwargs)

        return instance

def load_whisper(pretrained_name = 'medium', pretrained = None, ** kwargs):
    global _whisper_loaded
    
    if isinstance(pretrained, str): pretrained_name, pretrained = pretrained, None
    
    if pretrained is None:
        from utils.file_utils import download_file
        from models import get_pretrained_weights_dir
        
        if pretrained_name.startswith('whisper-'):
            pretrained_name = pretrained_name.replace('whisper-', '')
        
        if pretrained_name not in _whisper_loaded:
            import torch

            if pretrained_name not in WHISPER_MODELS:
                raise ValueError('Unknown pretrained Whisper model !\n  Accepted : {}\n  Got : {}'.format(
                    tuple(WHISPER_MODELS.keys()), pretrained_name
                ))

            filename = download_file(
                WHISPER_MODELS[pretrained_name], directory = get_pretrained_weights_dir()
            )

            if filename is None:
                raise RuntimeError('filename is None, an error has occured while loading it')

            pretrained  = torch.load(filename, map_location = 'cpu')
            if not isinstance(pretrained, dict): pretrained = pretrained.state_dict()
            _whisper_loaded[pretrained_name] = pretrained
        
        pretrained = _whisper_loaded[pretrained_name]
    
    if isinstance(pretrained, dict):
        state_dict = pretrained
    elif hasattr(pretrained, 'dims'):
        state_dict = {
            'dims' : pretrained.dims.__dict__, 'model_state_dict' : pretrained.state_dict()
        }
    elif hasattr(pretrained, 'config'):
        state_dict = {
            'model_state_dict' : pretrained.state_dict(),
            'dims'  : {
                'n_mels'    : pretrained.config.num_mel_bins,
                'n_vocab'   : pretrained.config.vocab_size,
                
                'n_audio_state' : pretrained.config.d_model,
                'n_audio_ctx'   : pretrained.config.max_source_positions,
                'n_audio_layer' : pretrained.config.encoder_layers,
                'n_audio_head'  : pretrained.config.encoder_attention_heads,

                'n_text_state' : pretrained.config.d_model,
                'n_text_ctx'   : pretrained.config.max_target_positions,
                'n_text_layer' : pretrained.config.decoder_layers,
                'n_text_head'  : pretrained.config.decoder_attention_heads
            }
        }
    
    return state_dict

