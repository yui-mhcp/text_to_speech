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
import numpy as np

from functools import cached_property

from utils import dump_json
from utils.keras_utils import ops
from utils.text.text_encoder import TextEncoder

class SentencePieceTextEncoder(TextEncoder):
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
        
    def split_text(self, text, tokens = None, ** _):
        if tokens is None: return [text]
        return super().split_text(text, tokens = tokens)

    def _tokenize(self, text):
        return self.tokenizer.encode_as_pieces(text)

    def decode(self, sequence, skip_padding = True, attach_punctuation = True,
               remove_tokens = False):
        """ Decode a given np.ndarray by replacing each known id by its corresponding token """
        if hasattr(sequence, 'tokens'): sequence = sequence.tokens
        if ops.is_tensor(sequence):     sequence = ops.convert_to_numpy(sequence)
        if isinstance(sequence, np.ndarray):
            if np.issubdtype(sequence.dtype, np.floating) and all(s > 0 for s in sequence.shape):
                sequence = np.argmax(sequence, axis = -1)
            
            if len(sequence.shape) > 1:
                return [self.decode(
                    s, skip_padding = skip_padding, attach_punctuation = attach_punctuation,
                    remove_tokens = remove_tokens
                ) for s in sequence]
        
        if isinstance(sequence, (list, tuple)) and not isinstance(sequence[0], (int, np.integer)):
            return [self.decode(
                s, skip_padding = skip_padding, attach_punctuation = attach_punctuation,
                remove_tokens = remove_tokens
            ) for s in sequence]
        
        sequence = [int(tok) for tok in sequence if tok != self.blank_token_idx]
        if self.offset == 0: return self.tokenizer.decode_ids(sequence)
        idx_to_token = self.index_to_token
        return ''.join([
            self.tokenizer.id_to_piece(idx - self.offset) if idx not in idx_to_token else idx_to_token[idx]
            for idx in sequence
        ]).replace(self.space_replacement, ' ').strip()

    def save_to_file(self, filename):
        model_path  = filename.replace('.json', '.model')
        
        with open(model_path, 'wb') as file:
            file.write(self.tokenizer.serialized_model_proto())
        
        config = self.get_config()
        config['tokenizer'] = model_path
        
        dump_json(filename, config, indent = 4)

        return filename
