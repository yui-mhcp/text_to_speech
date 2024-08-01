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
    positional_offset   = 2,
    
    normalize   = 'after',
    
    mha_residual    = True,
    mha_normalize   = True,
    mha_mask_factor = -1e9,
    mha_normalize_input = False,
    
    ffn_dim = 4.
)

@keras.saving.register_keras_serializable('transformers')
class XLMRoberta(TextTransformerEncoder):
    default_params = HParamxLMRoberta

    def transfer_weights(self, * args, ** kwargs):
        kwargs['patterns'] = {
            'LayerNorm' : 'norm', 'layer/' : 'layer_', 'attention/output' : 'mha', 'mha/dense' : 'mha/output_layer',
            'attention/self' : 'mha', 'output/dense' : 'dense_2', 'intermediate/dense' : 'dense_1', 'output/norm' : 'norm',
            'colbert' : '/pooler/colbert',
            'sparse' : '/pooler/sparse'
        }
        return super().transfer_weights(* args, ** kwargs)
    
    @classmethod
    def convert_config(cls, config, prefix = None, pretrained = None):
        from . import convert_hf_config
            
        hparams = convert_hf_config(config, cls.default_params, prefix)
        hparams.max_input_length -= 2
        if 'bge' in pretrained:
            hparams.update({
                'poolers'   : {'dense' : None, 'sparse' : 1, 'colbert' : -1},
                'pooler_mode'   : {'dense' : 0, 'sparse' : None, 'colbert' : '1:'},
                'pooler_activation' : {'dense' : 'l2', 'sparse' : 'relu', 'colbert' : 'l2'}
            })
        return hparams
