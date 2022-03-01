
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

""" Inspired from https://github.com/keithito/tacotron """

import re
import unicodedata

from unidecode import unidecode
from utils.text.numbers import normalize_numbers


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

_acronym_re     = re.compile(r'\b[A-Z]+\b')
_punctuation    = '_!?.,’“”‚‘—–()[]{}:;\'"`+-*/^=\\<>&#$%@¿′″·§~'
_left_punctuation   = '([{'
_right_punctuation  = ')]},.'
_accents    = "âéèêîç"


# List of (regular expression, replacement) pairs for abbreviations:
_english_abreviations = [(re.compile(r'\b{}\.'.format(x[0]), re.IGNORECASE), x[1])
    for x in [
        ('mrs', 'misess'),
        ('mr', 'mister'),
        ('dr', 'doctor'),
        ('st', 'saint'),
        ('co', 'company'),
        ('jr', 'junior'),
        ('maj', 'major'),
        ('gen', 'general'),
        ('drs', 'doctors'),
        ('rev', 'reverend'),
        ('lt', 'lieutenant'),
        ('hon', 'honorable'),
        ('sgt', 'sergeant'),
        ('capt', 'captain'),
        ('esq', 'esquire'),
        ('ltd', 'limited'),
        ('col', 'colonel'),
        ('ft', 'fort'),
    ]
]

_letter_pronounciation  = {
    'a' : {'fr' : 'ha', 'en' : 'ae'},
    'b' : {'fr' : 'bé', 'en' : 'be'},
    'c' : {'fr' : 'cé', 'en' : 'ce'},
    'd' : {'fr' : 'dé', 'en' : 'de'},
    'e' : {'fr' : 'euh', 'en' : 'e'},
    'f' : {'fr' : 'effe', 'en' : 'af'},
    'g' : {'fr' : 'gé', 'en' : 'ge'},
    'h' : {'fr' : 'hache', 'en' : 'aich'},
    'i' : {'fr' : 'ih', 'en' : 'eye'},
    'j' : {'fr' : 'ji', 'en' : 'jay'},
    'k' : {'fr' : 'ka', 'en' : 'kay'},
    'l' : {'fr' : 'elle', 'en' : 'el'},
    'm' : {'fr' : 'aime', 'en' : 'am'},
    'n' : {'fr' : 'aine', 'en' : 'an'},
    'o' : {'fr' : 'eau', 'en' : 'oo'},
    'p' : {'fr' : 'pé', 'en' : 'pe'},
    'q' : {'fr' : 'cu', 'en' : 'qu'},
    'r' : {'fr' : 'air', 'en' : 'ar'},
    's' : {'fr' : 'aisse', 'en' : 'as'},
    't' : {'fr' : 'thé', 'en' : 'tea'},
    'u' : {'fr' : 'eu', 'en' : 'yu'},
    'v' : {'fr' : 'vé', 'en' : 've'},
    'w' : {'fr' : 'double vé', 'en' : 'double yu'},
    'x' : {'fr' : 'ix', 'en' : 'ex'},
    'y' : {'fr' : 'i grec', 'en' : 'way'},
    'z' : {'fr' : 'zed', 'en' : 'ze'},
}

def strip(text, lstrip = True, rstrip = True, ** kwargs):
    if lstrip and rstrip: return text.strip()
    elif lstrip: return text.lstrip()
    elif rstrip: return text.rstrip()
    return text

def lstrip(text, ** kwargs):
    return text.lstrip()

def rstrip(text, ** kwargs):
    return text.rstrip()

def expand_abbreviations(text, abreviations = _english_abreviations, ** kwargs):
    for regex, replacement in abreviations:
        text = re.sub(regex, replacement, text)
    return text

def _expand_acronym(text, lang, extensions = _letter_pronounciation, ** kwargs):
    if len(text) > 4 or (text == 'I' and lang == 'en'): return text
    return ' '.join([extensions.get(c.lower(), {}).get(lang, c) for c in text])

def detach_punctuation(text, punctuation = _punctuation, ** kwargs):
    for punct in punctuation:
        text = text.replace(punct, ' {} '.format(punct))
    return text.strip()

def remove_punctuation(text, punctuation = _punctuation, ** kwargs):
    return ''.join(c for c in text if c not in punctuation)

def remove_tokens(text, tokens = None, ** kwargs):
    if not tokens: return text
    regex = re.compile(r'\b({})\b'.format('|'.join(tokens)))
    text = re.sub(regex, ' ', text)
    return text

def attach_punctuation(text, ** kwargs):
    for punct in _left_punctuation:
        text = text.replace('{} '.format(punct), punct)
    for punct in _right_punctuation:
        text = text.replace(' {}'.format(punct), punct)
    return text

def expand_acronym(text, lang, ** kwargs):
    return re.sub(_acronym_re, lambda m: _expand_acronym(m.group(0), lang), text)

def expand_numbers(text, lang = 'en', ** kwargs):
    return normalize_numbers(text, lang = lang, ** kwargs)

def lowercase(text, ** kwargs):
    return text.lower()

def collapse_whitespace(text, ** kwargs):
    return re.sub(_whitespace_re, ' ', text)

def remove_control(text, ** kwargs):
    return "".join([
        c for c in text if c in ('\t', '\n', '\r', ' ') or c.isalnum()
        or not unicodedata.category(c).startswith('C')
    ])

def remove_accents(text, ** kwargs):
    text = unicodedata.normalize("NFD", text)
    return ''.join([c for c in text if unicodedata.category(c) != "Mn"])

def convert_to_ascii(text, ** kwargs):
    return unidecode(text)

def fr_convert_to_ascii(text, accents_to_keep = _accents, ** kwargs):
    converted = []
    for c in text:
        converted.append(unidecode(c) if c not in accents_to_keep else c)
    return ''.join(converted)

def convert_to_alnum(text, allowed_char = '.,?! ', replace_char = ' ', ** kwargs):
    new_text = ''
    for c in text:
        if c.isalnum() or c in allowed_char: new_text += c
        else: new_text += replace_char
    return new_text

def basic_cleaners(text, ** kwargs):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text, ** kwargs)
    text = collapse_whitespace(text, ** kwargs)
    return text

def transliteration_cleaners(text, ** kwargs):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text, ** kwargs)
    text = lowercase(text, ** kwargs)
    text = collapse_whitespace(text, ** kwargs)
    return text

def english_cleaners(text, to_lowercase = True, to_expand = True, to_expand_acronyms = False,
                     ** kwargs):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text, ** kwargs)
    if to_expand_acronyms: text = expand_acronym(text, lang = 'en', ** kwargs)
    if to_lowercase: text = lowercase(text, ** kwargs)
    if to_expand: text = expand_numbers(text, lang = 'en', ** kwargs)
    text = expand_abbreviations(text, ** kwargs)
    text = collapse_whitespace(text, ** kwargs)
    return text

def french_cleaners(text, to_lowercase = True, to_expand = True, to_expand_acronyms = False,
                    ** kwargs):
    '''Pipeline for French text, including number expansion.'''
    text = fr_convert_to_ascii(text, ** kwargs)
    if to_expand_acronyms: text = expand_acronym(text, lang = 'fr', ** kwargs)
    if to_lowercase: text = lowercase(text, ** kwargs)
    if to_expand: text = expand_numbers(text, lang = 'fr', ** kwargs)
    text = collapse_whitespace(text, ** kwargs)
    return text
