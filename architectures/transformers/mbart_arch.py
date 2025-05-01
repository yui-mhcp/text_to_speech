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

import keras

from .bart_arch import *

HParamsMBartEncoder     = HParamsBartEncoder(
    normalize   = 'middle',
    mha_normalize   = False,
    mha_normalize_input = True,
    normalize_output    = True,
    ffn_activation  = 'gelu'
)

HParamsMBartDecoder  = HParamsBartDecoder(
    normalize   = 'middle',
    normalize_output    = True,
    mha_normalize   = False,
    mha_normalize_input = True,
    enc_mha_normalize   = False,
    enc_mha_normalize_input = True,
    ffn_activation  = 'gelu'
)

@keras.saving.register_keras_serializable('transformers')
class MBartEncoder(BartEncoder):
    default_params = HParamsMBartEncoder

@keras.saving.register_keras_serializable('transformers')
class MBartDecoder(BartDecoder):
    default_params = HParamsMBartDecoder
    
@keras.saving.register_keras_serializable('transformers')
class MBart(Bart):
    encoder_class   = MBartEncoder
    decoder_class   = MBartDecoder

def transformers_mbart(name = 'moussaKam/barthez', task = 'generation'):
    import transformers
    if task == 'generation':
        return transformers.AutoModelForSeq2SeqLM.from_pretrained(name)
    else:
        raise ValueError("Unknown task !\n  Accepted : {}\n  Got : {}".format(
            tuple(_transformers_pretrained_task.keys()), task
        ))
