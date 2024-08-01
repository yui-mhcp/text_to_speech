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

from tqdm import tqdm

from custom_layers import CustomEmbedding, get_activation
from .text_transformer_arch import *

HParamsBartEncoder      = HParamsTextTransformerEncoder(
    positional_offset = 2, scale_embedding = False
)

HParamsBartDecoder  = HParamsTextTransformerDecoder(
    positional_offset   = 2,
    scale_embedding = False,
    final_activation    = 'softmax'
)

@keras.saving.register_keras_serializable('transformers')
class BartEncoder(TextTransformerEncoder):
    default_params = HParamsBartEncoder

@keras.saving.register_keras_serializable('transformers')
class BartDecoder(TextTransformerDecoder):
    default_params = HParamsBartDecoder
    
    def __init__(self, vocab_size, embedding_dim, token_embedding = None, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim,
            token_embedding = token_embedding, ** kwargs
        )

        self.final_act_layer    = get_activation(self.hparams.final_activation)
    
    def build(self, input_shape):
        super().build(input_shape)
        self.final_bias = self.add_weight(
            shape = [1, self.vocab_size], name = "final_bias", trainable = False, initializer = "zeros"
        )
        
    def change_vocabulary(self, vocab, ** kwargs):
        raise NotImplementedError()
        
    @property
    def output_dim(self):
        return self.vocab_size
    
    def compute_output(self, output, apply_softmax = True, ** kwargs):
        output = super().compute_output(output, ** kwargs)
        
        output = self.embeddings.linear(output) + self.final_bias
        if self.final_act_layer is not None and apply_softmax:
            output = self.final_act_layer(output)
        return output

@keras.saving.register_keras_serializable('transformers')
class Bart(TextTransformer):
    encoder_class   = BartEncoder
    decoder_class   = BartDecoder
    
    def __init__(self, vocab_size, embedding_dim, max_input_length,
                 ** kwargs):
        shared_embedding = CustomEmbedding(
            vocab_size, embedding_dim, name = "token_embedding"
        )
        super().__init__(
            vocab_size      = vocab_size,
            embedding_dim   = embedding_dim,
            max_input_length    = max_input_length,
            shared_layers   = {'token_embedding' : shared_embedding},
            ** kwargs
        )


def transformers_bart(name = 'facebook/bart-large', task = 'generation'):
    import transformers
    if task == 'generation':
        if 'barthez' in name:
            return transformers.AutoModelForSeq2SeqLM.from_pretrained(name)
        return transformers.TFBartForConditionalGeneration.from_pretrained(name)
    else:
        raise ValueError("Unknown task !\n  Accepted : {}\n  Got : {}".format(
            tuple(_transformers_pretrained_task.keys()), task
        ))

