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

from .base_model import BaseModel
from utils import HParams, copy_methods, is_dataframe
from utils.keras_utils import TensorSpec, ops
from utils.text import TextEncoder, get_encoder, random_mask

_tokens         = ('sos_token', 'eos_token', 'blank_token', 'ukn_token', 'sep_token', 'mask_token')
_token_indexes  = [tok_name + '_idx' for tok_name in _tokens]
_methods        = ('encode', 'decode', 'format', 'split', 'join', 'split_and_format')
_methods        += tuple(['multi_' + m for m in _methods])
_methods        += ('ctc_decode', )

TextTrainingHParams = HParams(nb_mask = 0.1)

@copy_methods(
    'text_encoder', 'vocab', 'vocab_size', 'template', * _tokens, * _token_indexes, 'clean_text',
    ** {name + '_text' : name for name in _methods},
    chat_template = 'template', encode_chat = 'encode_template',
    attr_type = TextEncoder
)
class BaseTextModel(BaseModel):
    def _init_text(self, lang, text_encoder = None, ** kwargs):
        """ Init variables for text-based models """
        self.lang   = lang
        
        self.text_encoder = get_encoder(
            text_encoder = text_encoder, lang = lang
        )
    
    @property
    def text_encoder_file(self):
        return os.path.join(self.save_dir, 'text_encoder.json')

    @property
    def is_chat_model(self):
        return self.text_encoder.template is not None
    
    @property
    def is_encoder_decoder(self):
        if self.runtime == 'trt_llm': return self.model.is_enc_dec
        return getattr(self.model, 'decoder', None) is not None

    @property
    def text_signature(self):
        return TensorSpec(shape = (None, None), dtype = 'int32')

    @property
    def multi_text_signature(self):
        return TensorSpec(shape = (None, None, None), dtype = 'int32')

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
        if not lang: lang = self.lang
        old_vocab = self.vocab
        
        if isinstance(new_encoder, str):
            from models import is_model_name, get_pretrained
            if is_model_name(new_encoder):
                new_encoder = get_pretrained(new_encoder)
            else:
                new_encoder = get_encoder(lang = lang, text_encoder = new_encoder)
        
        if isinstance(new_encoder, BaseTextModel):
            self.lang   = new_encoder.lang
            self.text_encoder   = new_encoder.text_encoder
        elif isinstance(new_encoder, TextEncoder):
            self.lang   = lang
            self.text_encoder   = new_encoder
        else:
            raise ValueError('Unsupported `TextEncoder` (type {}) : {}'.format(
                type(new_encoder), new_encoder
            ))
        
        self.save_text_encoder(force = True)
        
        self.update_model_vocab(old_vocab)
        if self.vocab != old_vocab: self.save()
        
        return self.text_encoder

    def update_model_vocab(self, old_vocab, ** kwargs):
        model = self.model
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
    
    def prepare_text(self, data, *, format = None, max_length = None, use_template = True, ** kwargs):
        if is_dataframe(data): data = data.to_dict('records')
        if isinstance(data, list):
            return [
                self.prepare_text(
                    d, format = format, max_length = max_length, use_template = use_template, ** kwargs
                )
                for d in data
            ]
        
        # Most common case first
        if use_template and self.is_chat_model:
            if not isinstance(data, dict): data = {'text' : data}
            return self.encode_chat(format = format, ** {** kwargs, ** data})
        elif not format and not max_length:
            return self.encode_text(data, ** kwargs)
        elif format and max_length:
            if not isinstance(data, dict): data = {'text' : data}
            split_key = kwargs.pop('split_key', 'text' if len(data) != 1 else list(data.keys())[0])
            return self.split_and_format_text(format, split_key, max_length, ** data, ** kwargs)
        elif format:
            if not isinstance(data, dict): data = {'text' : data}
            return self.format_text(format, ** {** kwargs, ** data})
        else:
            return self.split_text(data, max_length, ** kwargs)
        
    def augment_text(self, tokens):
        return ops.cond(
            ops.random.uniform(()) < self.augment_prct,
            lambda: random_mask(tokens, self.blank_token_idx, self.nb_mask),
            lambda: tokens
        )

    def save_text_encoder(self, filename = None, force = False):
        if filename is None: filename = self.text_encoder_file
        
        if not os.path.exists(filename) or force:
            self.text_encoder.save_to_file(filename)
        return filename
    
    def get_config_text(self):
        # Saving text encoder and mel fn (if needed)
        self.save_text_encoder()
        
        return {
            'lang'  : self.lang,
            'text_encoder'  : self.text_encoder_file
        }
        
