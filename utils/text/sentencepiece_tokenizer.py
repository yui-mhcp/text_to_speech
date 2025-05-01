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

from functools import cached_property

from ..file_utils import dump_json
from .tokenizer import Tokenizer

class SentencePieceTokenizer(Tokenizer):
    def __init__(self, vocab, tokenizer, *, offset = 0, ** kwargs):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, str):
            import sentencepiece

            self.tokenizer = sentencepiece.SentencePieceProcessor()
            with open(tokenizer, 'rb') as f:
                model_proto = f.read()
            self.tokenizer.Load(model_proto = model_proto)

        self.offset = offset
        kwargs['level'] = 'token'
        
        super().__init__(vocab = vocab, ** kwargs)
    
    @property
    def index_to_token(self):
        return {v : k for k, v in self.token_indexes.items()}
    
    @cached_property
    def space_replacement(self):
        return self.tokenizer.encode_as_pieces(' !')[0][0]
        
    def split_text(self, text, tokens = None):
        if not tokens: return [text]
        return super().split_text(text, tokens = tokens)

    def _tokenize(self, token):
        if isinstance(token, (list, tuple)): return super()._tokenize(token)
        return self.tokenizer.encode_as_pieces(token)

    def decode_ids(self, tokens):
        if self.offset == 0: return self.tokenizer.decode_ids(tokens)
        
        idx_to_token = self.index_to_token
        return ''.join([
            self.tokenizer.id_to_piece(idx - self.offset) if idx not in idx_to_token else idx_to_token[idx]
            for idx in tokens
        ]).replace(self.space_replacement, ' ').strip()

    def get_config(self):
        config = super().get_config()
        config['offset'] = self.offset
        return config
    
    def save(self, filename):
        model_path  = filename.replace('.json', '.model')
        
        with open(model_path, 'wb') as file:
            file.write(self.tokenizer.serialized_model_proto())
        
        config = self.get_config()
        config['tokenizer'] = model_path
        
        dump_json(filename, config, indent = 4)

        return filename
