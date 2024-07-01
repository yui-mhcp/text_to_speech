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

import keras

from .text_transformer_arch import TextTransformerEncoder, HParamsTextTransformerEncoder

HParamxLMRoberta = HParamsTextTransformerEncoder(
    scale_embeddings    = False,
    normalize_embeddings    = True,
    normalize_output    = False,
    max_token_types     = 1,
    
    epsilon = 1e-5,
    normalize   = 'after',
    
    mha_residual    = True,
    mha_normalize   = True,
    mha_mask_factor = -1e9,
    mha_normalize_input = False,
    mha_epsilon      = 1e-5,
    
    ffn_dim          = 4.
)

@keras.saving.register_keras_serializable('transformers')
class XLMRoberta(TextTransformerEncoder):
    default_params = HParamxLMRoberta

    def transfer_weights(self, * args, ** kwargs):
        kwargs['patterns'] = {
            'LayerNorm' : 'norm', 'layer/' : 'layer_', 'attention/output' : 'mha', 'mha/dense' : 'mha/output_layer',
            'attention/self' : 'mha', 'output/dense' : 'dense_2', 'intermediate/dense' : 'dense_1', 'output/norm' : 'norm'
        }
        return super().transfer_weights(* args, ** kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_name, pretrained = None, ** kwargs):
        if pretrained is None:
            from transformers import AutoModel
            pretrained = AutoModel.from_pretrained(pretrained_name)

        config = cls.default_params(
            vocab_size       = pretrained.config.vocab_size,
            embedding_dim    = pretrained.config.hidden_size,
            num_layers       = pretrained.config.num_hidden_layers,
            max_input_length = pretrained.config.max_position_embeddings - 2,
            mha_num_heads    = pretrained.config.num_attention_heads,
            ffn_activation   = pretrained.config.hidden_act,
            positional_offset   = 2,
            sos_oken    = 0,
            pad_token   = 1,
            eos_token   = 2
        )
        
        instance = cls(** config(** kwargs))
        instance.build((None, None))
        
        instance.transfer_weights(pretrained, ** kwargs)
        del pretrained
        return instance 

