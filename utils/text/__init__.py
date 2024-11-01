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
import logging

from .byte_pair_encoding import bytes_to_unicode, bpe
from .text_encoder import TextEncoder
from .sentencepiece_encoder import SentencePieceTextEncoder
from .ctc_decoder import ctc_decode
from .text_processing import *
from .text_augmentation import random_mask

from .document_parser import _wiki_cleaner, parse_document

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

accent_replacement_matrix = {
    'a' : {'à' : 0, 'â' : 0}, 'à' : {'a' : 0, 'â' : 0}, 'â' : {'a' : 0, 'à' : 0},
    'u' : {'ù' : 0}, 'ù' : {'u' : 0},
    'o' : {'ô' : 0}, 'ô' : {'o' : 0},
    'i' : {'î' : 0}, 'î' : {'i' : 0}
}

def get_encoder(lang = None, text_encoder = None, ** kwargs):
    if text_encoder is None: text_encoder = kwargs.copy()
    
    if isinstance(text_encoder, dict):
        if 'vocab' not in text_encoder:
            assert lang, 'You should provide either `vocab` either `lang` !'
            text_encoder['vocab'] = get_symbols(lang, arpabet = False)
            text_encoder['level'] = 'char'
        else:
            text_encoder.setdefault('level', 'char')
        
        text_encoder.setdefault('use_sos_and_eos', False)
        text_encoder.setdefault('cleaners', _default_cleaners.get(lang, 'basic_cleaners'))
        
        encoder = TextEncoder(** text_encoder)
        
    elif isinstance(text_encoder, str):
        try:
            from models import get_model_dir
            model_encoder_file = get_model_dir(text_encoder, 'saving', 'text_encoder.json')
        except:
            model_encoder_file = None
        
        if os.path.isfile(text_encoder):
            encoder = TextEncoder.load_from_file(text_encoder)
        elif model_encoder_file and os.path.isfile(model_encoder_file):
            encoder = TextEncoder.load_from_file(model_encoder_file)
        elif text_encoder == 'clip':
            encoder = TextEncoder.from_clip_pretrained()
        elif 'whisper' in text_encoder:
            encoder = TextEncoder.from_whisper_pretrained(** kwargs)
        else:
            encoder = TextEncoder.from_transformers_pretrained(text_encoder)
    elif isinstance(text_encoder, TextEncoder):
        encoder = text_encoder
    else:
        raise ValueError("Unhandled `text_encoder` (type {}) : {}".format(
            type(text_encoder), text_encoder
        ))
    
    return encoder

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

def default_encoder(lang, ** kwargs):
    lang = lang.lower()
    if lang in ('fr', 'francais', 'français', 'french'):
        return default_french_encoder(** kwargs)
    elif lang in ('en', 'english', 'anglais'):
        return default_english_encoder(** kwargs)
    else:
        logger.warning("Unknown language : {} - return char-level encoder with default symbols".format(lang))
        return TextEncoder(get_symbols(lang), level = 'char', ** kwargs)

def default_english_encoder(cleaners = ['english_cleaners'], level = 'char', ** kwargs):
    return TextEncoder(en_symbols, level = level, cleaners = cleaners, ** kwargs)

def default_french_encoder(cleaners = ['french_cleaners'], level = 'char', ** kwargs):
    return TextEncoder(fr_symbols, level = level, cleaners = cleaners, ** kwargs)
