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

import os
import keras
import logging
import keras.ops as K

from custom_layers import get_activation, RotaryMultiHeadAttention, RMSLayerNormalization
from .text_transformer_arch import TextTransformerEncoder, HParamsTextTransformerEncoder

logger = logging.getLogger(__name__)

HParamsMistral  = HParamsTextTransformerEncoder(
    use_causal_attention    = True,
    normalize_embeddings    = False,
    scale_embeddings    = False,
    max_input_length    = -1,
    
    output_dim      = None,
    final_bias      = False,
    final_activation    = 'softmax',
    
    epsilon = 1e-5,
    normalize   = 'middle',
    normalize_output    = True,
    
    mha_normalize_input = True,
    mha_output_bias = False,
    mha_use_bias    = False,
    mha_mask_factor = -1e9,
    mha_normalize   = False,
    mha_epsilon = 1e-5,
    
    ffn_dim     = 4.,
    ffn_use_bias    = False,
    ffn_activation  = 'silu',
    ffn_use_up_proj = True
)

@keras.saving.register_keras_serializable('transformers')
class Mistral(TextTransformerEncoder):
    default_params  = HParamsMistral

    def __init__(self, * args, ** kwargs):
        kwargs.update({
            'mha_class' : RotaryMultiHeadAttention,
            'norm_class'    : RMSLayerNormalization
        })
        super().__init__(* args, ** kwargs)

        self.final_dense    = keras.layers.Dense(
            self.output_dim, use_bias = self.hparams.final_bias, name = 'final_dense'
        )
        self.final_act_layer    = get_activation(self.hparams.final_activation)
    
    def build(self, input_shape):
        super().build(input_shape)
        self.final_dense.build((None, None, self.embedding_dim))
    
    @property
    def output_dim(self):
        return self.hparams.output_dim if self.hparams.output_dim else self.hparams.vocab_size
    
    def compute_output(self, output, *, apply_softmax = True, ** kwargs):
        output = super().compute_output(output, ** kwargs)
        
        output = self.final_dense(output)
        if self.final_act_layer is not None and apply_softmax:
            output = self.final_act_layer(output)
        return output

    def prepare_input(self,
                      inputs,
                      *,
                      attention_kwargs,
                      lengths = None,
                      initial_state = None,
                      ** kwargs
                     ):
        if initial_state:
            assert lengths is not None, 'You must proide `lengths`'
            offset  = lengths - 1
        else:
            offset  = None

        if logger.isEnabledFor(logging.DEBUG) and keras.backend.backend() == 'tensorflow':
            import tensorflow as tf
            tf.print('inputs shape :', K.shape(inputs), 'offset :', offset)

        sin, cos = self[0].attention.get_rotary_embedding(
            K.shape(inputs)[1], offset, self.compute_dtype
        )
        attention_kwargs.update({'sin' : sin, 'cos' : cos})
        
        return super().prepare_input(
            inputs, lengths = lengths, initial_state = initial_state, ** kwargs
        )

    def transfer_weights(self, pretrained, ** kwargs):
        kwargs.setdefault('skip_root', False)
        kwargs.setdefault('patterns', {
            'layers/' : 'layer_', 'mlp' : 'ffn', 'gate_proj' : 'dense_1', 'down_proj' : 'dense_2',
            'self_attn' : 'mha',
            'input_layernorm' : 'mha/norm_input', 'post_attention_layernorm' : 'norm',
            'o_' : 'output_', 'q_' : 'query_', 'k_' : 'key_', 'v_' : 'value_', '_proj' : '_layer'
        })

        return super(Mistral, self).transfer_weights(pretrained, ** kwargs)
    
    @classmethod
    def from_pretrained(cls,
                        pretrained_name = None,
                        pretrained  = None,
                        partial     = False,
                        ** kwargs
                       ):
        from utils import load_json, dump_json
        from models import get_pretrained_weights_dir
        
        model_dir   = os.path.join(
            get_pretrained_weights_dir(), pretrained_name.replace('/', '--')
        )
        config_file = os.path.join(model_dir, 'config.json')
        weights_file    = os.path.join(model_dir, 'model.weights.h5')
        if not os.path.exists(config_file) or not os.path.exists(weights_file):
            if pretrained is None: pretrained = _get_pretrained_mistral(pretrained_name, ** kwargs)

            config = cls.default_params(
                vocab_size       = pretrained.config.vocab_size,
                embedding_dim    = pretrained.config.hidden_size,
                num_layers       = pretrained.config.num_hidden_layers,
                
                sos_token   = 1,
                eos_token   = 2,
                pad_token   = 2,
                
                mha_num_heads    = pretrained.config.num_attention_heads,
                mha_num_kv_heads = pretrained.config.num_key_value_heads,
        
                ffn_dim          = pretrained.config.intermediate_size,
                ffn_activation   = pretrained.config.hidden_act,
            )

            instance = cls(** config(** kwargs))
            instance.build((None, None))

            print(instance.transfer_weights(pretrained, ** kwargs))
            
            os.makedirs(model_dir, exist_ok = True)
            dump_json(config_file, instance.get_config(), indent = 4)
            instance.save_weights(weights_file)
        else:
            logger.info('Building model from config file {}'.format(config_file))
            instance = cls.from_config({** load_json(config_file), ** kwargs})
            instance.build((None, None))

            logger.info('Loading weights from {}'.format(weights_file))
            try:
                instance.load_weights(weights_file)
            except ValueError as e:
                if partial:
                    from models.weights_converter import name_based_partial_transfer_learning
                    
                    logger.info('Loading official pretrained model for partial transfer')
                    original = cls.from_pretrained(pretrained_name, pretrained)
                    
                    logger.info('Making partial transfer learning')
                    name_based_partial_transfer_learning(instance, original, ** kwargs)
                    del original
                else:
                    logger.warning(str(e))

        return instance

def _get_pretrained_mistral(model_name, torch_dtype = 'float16', device = 'cpu', ** _):
    import torch
    
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_name, device_map = device, torch_dtype = getattr(torch, torch_dtype)
    )
