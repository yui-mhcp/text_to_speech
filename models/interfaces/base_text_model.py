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

import os

from .base_model import BaseModel
from utils.keras import TensorSpec, ops
from utils.wrappers import copy_methods
from utils.text import Tokenizer, get_tokenizer, format_text

_tokens         = ('sos_token', 'eos_token', 'blank_token', 'ukn_token', 'sep_token', 'mask_token')
_token_indexes  = [tok_name + '_idx' for tok_name in _tokens]

@copy_methods(
    'tokenizer', 'vocab', 'vocab_size', 'template', * _tokens, * _token_indexes, 'clean_text',
    ** {name + '_text' : name for name in ('encode', 'decode', 'ctc_decode', 'format')},
    chat_template = 'template', encode_chat = 'encode_chat',
    type    = Tokenizer
)
class BaseTextModel(BaseModel):
    def _init_text(self, lang, tokenizer = None, text_encoder = None, ** kwargs):
        """ Init variables for text-based models """
        if text_encoder is not None: tokenizer = text_encoder
        
        self.lang   = lang
        
        self.tokenizer = get_tokenizer(tokenizer = tokenizer, lang = lang)
    
    @property
    def tokenizer_file(self):
        return os.path.join(self.save_dir, 'tokenizer.json')

    @property
    def is_chat_model(self):
        return self.chat_template is not None
    
    @property
    def is_encoder_decoder(self):
        if self.runtime == 'trt_llm': return self.model.is_enc_dec
        return getattr(self.model, 'decoder', None) is not None

    @property
    def text_signature(self):
        return TensorSpec(shape = (None, None), dtype = 'int32')
    
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
    
    def prepare_text(self, data, *, format = None, use_template = True, ** kwargs):
        if hasattr(data, 'to_dict'): data = data.to_dict('records')
        if isinstance(data, list):
            return [
                self.prepare_text(d, format = format, use_template = use_template, ** kwargs)
                for d in data
            ]
        
        # Most common case first
        if use_template and self.is_chat_model:
            assert not format, '`format` is not supported for chat models. Please apply the formatting before `prepare_text`'
            if not isinstance(data, dict): data = {'text' : data}
            return self.encode_chat(** {** kwargs, ** data})
        elif format:
            if not isinstance(data, dict): data = {'text' : data}
            data = format_text(format, ** data)
        return self.encode_text(data, ** kwargs)
    
    def save_tokenizer(self, filename = None, force = False):
        if filename is None: filename = self.tokenizer_file
        
        if not os.path.exists(filename) or force:
            self.tokenizer.save(filename)
        return filename
    
    def get_config_text(self):
        return {'lang' : self.lang, 'tokenizer' : self.save_tokenizer()}
        
