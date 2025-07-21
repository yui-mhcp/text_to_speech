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
import logging

from .web import *
from .parsers import *
from .metrics import *
from .cleaners import *
from .numbers import *
from .sentencepiece_tokenizer import SentencePieceTokenizer
from .text_processing import *
from .tokenizer import Tokenizer, TokenizerLevel, pretty_print_template
from .tokens_processing import *

logger = logging.getLogger(__name__)

_pad            = '_'
_punctuation    = '!\'(),.:;? '
_special    = '-'
_maj_letters    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
_min_letters    = 'abcdefghijklmnopqrstuvwxyz'
_letters    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_accents    = 'éèêîçô'
_maths      = '+*/%'
_numbers    = '0123456789'

_mini_punctuation   = ' \',.?!'
_mini_accents       = 'éèç'

#Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_cmudict_symbols = [
  'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

_arpabet = ['@' + s for s in _cmudict_symbols]
# Export all symbols:
en_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet
fr_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_accents)

_default_cleaners   = {
    'en'    : 'english_cleaners',
    'fr'    : 'french_cleaners',
    'multi' : 'french_cleaners' # toi avoid removing accents
}


def get_tokenizer(tokenizer = None, lang = None, ** kwargs):
    if tokenizer is None: tokenizer = kwargs
    
    if isinstance(tokenizer, Tokenizer):
        return tokenizer
    elif isinstance(tokenizer, str):
        model_tokenizer_file = os.path.join(
            'pretrained_models', tokenizer, 'saving', 'tokenizer.json'
        )
        if not os.path.exists(model_tokenizer_file):
            model_tokenizer_file = os.path.join(
                'pretrained_models', tokenizer, 'saving', 'text_encoder.json'
            )
        
        if os.path.isfile(tokenizer):
            return Tokenizer.load_from_file(tokenizer)
        elif os.path.isfile(model_tokenizer_file):
            return Tokenizer.load_from_file(model_tokenizer_file)
        elif tokenizer == 'clip':
            return Tokenizer.from_clip_pretrained()
        elif 'whisper' in tokenizer:
            return Tokenizer.from_whisper_pretrained(** kwargs)
        else:
            return Tokenizer.from_transformers_pretrained(tokenizer)

    elif isinstance(tokenizer, dict):
        if 'vocab' not in tokenizer:
            assert lang, 'You should provide either `vocab` either `lang` !'
            tokenizer['vocab'] = get_symbols(lang, arpabet = False)
            tokenizer['level'] = 'char'
        else:
            tokenizer.setdefault('level', 'char')
        
        tokenizer.setdefault('use_sos_and_eos', False)
        tokenizer.setdefault('cleaners', _default_cleaners.get(lang, 'basic_cleaners'))
        
        return Tokenizer(** tokenizer)
    else:
        raise ValueError("Unsupported `tokenizer` (type {}) : {}".format(
            type(tokenizer), tokenizer
        ))
    
    return tokenizer

def default_english_tokenizer(cleaners = ['english_cleaners'], level = 'char', ** kwargs):
    return Tokenizer(en_symbols, level = level, cleaners = cleaners, ** kwargs)

def default_french_tokenizer(cleaners = ['french_cleaners'], level = 'char', ** kwargs):
    return Tokenizer(fr_symbols, level = level, cleaners = cleaners, ** kwargs)

def get_symbols(lang,
                punctuation = 1,
                maj     = True,
                arpabet = True, 
                accents = True,
                numbers = False,
                maths   = False
               ):
    symbols = [_pad] + list(_special)
    
    if punctuation: 
        symbols += list(_punctuation) if punctuation == 1 else list(_mini_punctuation)
    else: symbols += [' ']
    
    symbols += list(_letters) if maj else list(_min_letters)
    
    if lang == 'en' and arpabet:            symbols += _arpabet
    if lang in ('fr', 'multi') and accents: symbols += list(_accents)
    
    if numbers: symbols += list(_numbers)
    if maths:   symbols += list(_maths)
    
    return symbols

