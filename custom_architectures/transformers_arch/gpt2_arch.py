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

HParamsBaseGPT2  = HParamsTextTransformerEncoder(
    use_causal_attention    = True,
    normalize_embeddings    = False,
    
    epsilon = 1e-5,
    normalize   = 'middle',
    normalize_output    = True,
    mha_normalize_input = True,
    mha_normalize   = False,
    mha_epsilon = 1e-5,
    
    ffn_dim     = 3072,
    ffn_activation  = 'gelu_new'
)

@keras.saving.register_keras_serializable('transformers')
class BaseGPT2(TextTransformerEncoder):
    default_params  = HParamsBaseGPT2

    def transfer_weights(self, pretrained, transpose = False, ** kwargs):
        from models.weights_converter import _attn_split

        kwargs.setdefault('transforms', _attn_split)
        if transpose:
            kwargs['transforms'] = {
                ** kwargs['transforms'], '.*' : lambda k, v: {k : [vi.T for vi in v]}
            }
            
        kwargs.setdefault('skip_root', True)

        return super().transfer_weights(pretrained, ** kwargs)
        
    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'gpt2',
                        pretrained_task = 'generation',
                        pretrained      = None,
                        ** kwargs
                       ):
        if pretrained is None:
            from transformers import GPT2Model
            pretrained = GPT2Model.from_pretrained(pretrained_name)

        if isinstance(pretrained, dict):
            pretrained  = {k : v for k, v in pretrained.items() if 'gpt' in k}
            n_layer     = len([k for k in pretrained if k.endswith('attn.weight')])
            
            config = HParamsBaseGPT2(
                vocab_size      = pretrained['gpt.transformer.wte.weight'].shape[0],
                embedding_dim   = pretrained['gpt.transformer.wte.weight'].shape[1],
                max_input_length    = pretrained['gpt.transformer.wpe.weight'].shape[0],
                sos_token   = 50256,
                eos_token   = 50256,

                num_layers  = n_layer,
                mha_num_heads   = 12
            )
        else:
            config = HParamsBaseGPT2(
                vocab_size      = pretrained.config.vocab_size,
                embedding_dim   = pretrained.config.n_embd,
                max_input_length    = pretrained.config.n_positions,
                sos_token   = 50256,
                eos_token   = 50256,

                num_layers  = pretrained.config.n_layer,
                mha_num_heads   = pretrained.config.n_head
            )

        instance = cls(** config(** kwargs))
        instance.build((None, None))

        instance.transfer_weights(pretrained, ** kwargs)

        return instance

@keras.saving.register_keras_serializable('transformers')
class GPT2(BaseGPT2):
    @property
    def output_dim(self):
        return self.vocab_size
    
    def compute_output(self, output, training = False, mask = None, ** kwargs):
        output = super().compute_output(output, training = training, mask = mask, ** kwargs)

        return self.embeddings.linear(output)
