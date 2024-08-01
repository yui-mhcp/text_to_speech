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
    
def _get_pretrained_mistral(model_name, torch_dtype = 'float16', device = 'cpu', ** _):
    import torch
    
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_name, device_map = device, torch_dtype = getattr(torch, torch_dtype)
    )
