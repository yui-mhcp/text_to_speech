
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
import tensorflow as tf

from hparams import HParams
from utils import copy_methods
from utils.text import TextEncoder, get_encoder, random_mask
from models.interfaces.base_model import BaseModel

_tokens         = ('sos_token', 'eos_token', 'blank_token', 'sep_token', 'mask_token')
_token_indexes  = [tok_name + '_idx' for tok_name in _tokens]
_methods        = ('encode', 'decode', 'format', 'split', 'join')
_methods        += tuple(['multi_' + m for m in _methods])

TextTrainingHParams = HParams(
    nb_mask   = 1,
    min_mask_length   = 1,
    max_mask_length   = 1
)

@copy_methods(
    'text_encoder', 'vocab', 'vocab_size', * _tokens, * _token_indexes, 'clean_text',
    ** {name + '_text' : name for name in _methods},
    ** {'tf_' + name + '_text' : name for name in _methods},
    attr_type = TextEncoder
)
class BaseTextModel(BaseModel):
    def _init_text(self, lang, text_encoder = None, text_encoder_config = {}, ** kwargs):
        """ Init variables for text-based models """
        self.lang   = lang
        
        # Initialization of Text Encoder
        self.text_encoder_config    = text_encoder_config
        self.text_encoder = get_encoder(
            text_encoder = text_encoder, lang = lang, ** text_encoder_config
        )
    
    @property
    def text_encoder_file(self):
        return os.path.join(self.save_dir, 'text_encoder.json')

    @property
    def is_encoder_decoder(self):
        return getattr(self.get_model(), 'decoder', None) is not None

    @property
    def text_signature(self):
        return tf.TensorSpec(shape = (None, None), dtype = tf.int32)

    @property
    def multi_text_signature(self):
        return tf.TensorSpec(shape = (None, None, None), dtype = tf.int32)

    @property
    def training_hparams_text(self):
        return TextTrainingHParams()
    
    @property
    def model_tokens(self):
        return {
            'sos_token' : self.sos_token_idx,
            'eos_token' : self.eos_token_idx,
            'pad_token' : self.blank_token_idx,
        }
    
    def _str_text(self):
        des = "- Language : {}\n".format(self.lang)
        des += "- Vocabulary (size = {}) : {}\n".format(
            self.vocab_size,
            self.vocab if len(self.vocab) < 25 else '[{}, ...]'.format(str(self.vocab[:25])[1:-1])
        )
        return des
    
    def set_text_encoder(self, new_encoder, lang = None, ** kwargs):
        """
            Change the current `text_encoder` to `new_encoder` and possibly update the model's vocabulary (if it has the mothod `change_vocabulary`)
            
            Arguments :
                - new_encoder   : the new text encoder
                    - filename (str)    : the text encoder's config filename
                    - model_name (str)  : the model's name from which to get the text encoder's config file
                    - TextEncoder       : the instance
                    - BaseTextModel's subclass  : the model instance
                - lang  : possibly update the language (if needed)
                - kwargs    : forwarded to `self.get_model().change_vocabulary(...)`
            Returns :
                - new_encoder   : the new TextEncoder instance initialized
        """
        from models import is_model_name, get_pretrained
        
        if lang is None: lang = self.lang
        old_vocab   = self.vocab
        
        if isinstance(new_encoder, str):
            if is_model_name(new_encoder):
                new_encoder = get_pretrained(new_encoder)
            else:
                new_encoder = get_encoder(lang = lang, text_encoder = new_encoder)
        
        if isinstance(new_encoder, BaseTextModel):
            self.lang   = new_encoder.lang
            self.text_encoder   = new_encoder.text_encoder
            self.text_encoder_config    = new_encoder.text_encoder_config
        elif isinstance(new_encoder, TextEncoder):
            self.lang   = lang
            self.text_encoder   = new_encoder
            self.text_encoder_config    = {}
        else:
            raise ValueError('Unsupported TextEncoder (type {}) : {}'.format(
                type(new_encoder), new_encoder
            ))
        
        self.save_text_encoder(force = True)
        
        self.update_model_vocab(old_vocab)
        if self.vocab != old_vocab: self.save()
        
        return self.text_encoder

    def update_model_vocab(self, old_vocab, ** kwargs):
        model = self.get_model()
        if self.vocab != old_vocab:
            if hasattr(model, 'change_vocabulary'):
                model.change_vocabulary(
                    self.vocab, old_vocab = old_vocab, ** self.model_tokens, ** kwargs
                )
            else:
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'change_vocabulary'):
                    model.encoder.change_vocabulary(
                        self.vocab, old_vocab = old_vocab, ** self.model_tokens, ** kwargs
                    )
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'change_vocabulary'):
                    model.decoder.change_vocabulary(
                        self.vocab, old_vocab = old_vocab, ** self.model_tokens, ** kwargs
                    )
    
    def augment_text(self, tokens, min_idx = 1, max_idx = -1, nb_mask = None,
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
        return tokens

    def save_text_encoder(self, filename = None, force = False):
        if filename is None: filename = self.text_encoder_file
        
        if not os.path.exists(filename) or force:
            self.text_encoder.save_to_file(filename)
        return filename
    
    def get_config_text(self, * args, ** kwargs):
        # Saving text encoder and mel fn (if needed)
        self.save_text_encoder()
        
        return {
            'lang'      : self.lang,
            'text_encoder'  : self.text_encoder_file
        }
        
