
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
import pandas as pd
import tensorflow as tf

from hparams import HParams
from utils.text import get_encoder, random_mask
from models.interfaces.base_model import BaseModel

TextTrainingHParams = HParams(
    nb_mask   = 1,
    min_mask_length   = 1,
    max_mask_length   = 1
)

class BaseTextModel(BaseModel):
    def _init_text(self, lang, text_encoder = None, text_encoder_config = {}, ** kwargs):
        self.lang   = lang
        
        # Initialization of Text Encoder
        self.text_encoder_config    = text_encoder_config
        self.text_encoder = get_encoder(
            text_encoder = text_encoder, lang = lang, ** text_encoder_config
        )
    
    @property
    def text_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),
            tf.TensorSpec(shape = (None,), dtype = tf.int32)
        )

    @property
    def training_hparams_text(self):
        return TextTrainingHParams()
    
    @property
    def text_encoder_file(self):
        return os.path.join(self.save_dir, 'text_encoder.json')
    
    @property
    def vocab(self):
        return self.text_encoder.vocab

    @property
    def vocab_size(self):
        return self.text_encoder.vocab_size

    @property
    def blank_token_idx(self):
        return self.text_encoder.blank_token_idx

    @property
    def sep_token(self):
        return self.text_encoder.sep_token

    @property
    def sep_token_idx(self):
        return self.text_encoder.sep_token_idx
    
    @property
    def mask_token_idx(self):
        return self.text_encoder.mask_token_idx
    
    @property
    def sos_token_idx(self):
        return self.text_encoder.sos_token_idx

    @property
    def eos_token_idx(self):
        return self.text_encoder.eos_token_idx

    def _str_text(self):
        des = "- Language : {}\n".format(self.lang)
        des += "- Vocabulary (size = {}) : {}\n".format(
            self.vocab_size,
            self.vocab if len(self.vocab) < 25 else '[{}, ...]'.format(str(self.vocab[:25])[1:-1])
        )
        return des
    
    def encode_text(self, text, * args, ** kwargs):
        return self.text_encoder.encode(text, * args, ** kwargs)
    
    def decode_text(self, encoded, ** kwargs):
        return self.text_encoder.decode(encoded, ** kwargs)
    
    def tf_encode_text(self, text, default_key = 'text'):
        if isinstance(text, (dict, pd.Series)): text = text[default_key]
        
        encoded_text = tf.py_function(
            self.encode_text, [text], Tout = tf.int32
        )
        encoded_text.set_shape([None])
        
        return encoded_text

    
    def augment_text(self, tokens, length, min_idx = 1, max_idx = -1, nb_mask = None,
                     min_mask_length = None, max_mask_length = None):
        if nb_mask is None: nb_mask = self.nb_mask
        if min_mask_length is None: min_mask_length = self.min_mask_length
        if max_mask_length is None: max_mask_length = self.max_mask_length
        
        tokens = tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: random_mask(
                tokens, self.mask_token_idx,
                min_idx = min_idx, max_idx = max_idx,
                nb_mask = nb_mask,
                min_mask_length = min_mask_length,
                max_mask_length = max_mask_length
            ),
            lambda: tokens
        )
        return tokens, len(tokens)

    def get_config_text(self, * args, ** kwargs):
        # Saving text encoder and mel fn (if needed)
        if not os.path.exists(self.text_encoder_file):
            self.text_encoder.save_to_file(self.text_encoder_file)

        return {
            'lang'      : self.lang,
            'text_encoder'  : self.text_encoder_file
        }
        
