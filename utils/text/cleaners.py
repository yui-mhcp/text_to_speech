
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

_special_symbols    = {
    '='     : {'fr' : 'égal',       'en' : 'equal'},
    '+'     : {'fr' : 'plus',       'en' : 'plus'},
    '/'     : {'fr' : 'slash',      'en' : 'slash'},
    '*'     : {'fr' : 'étoile',     'en' : 'star'},
    '^'     : {'fr' : 'chapeau',    'en' : 'hat'},
    '%'     : {'fr' : 'pourcent',   'en' : 'percent'},
    '§'     : {'fr' : 'paragraphe', 'en' : 'paragraph'},
    '&'     : {'fr' : 'et',         'en' : 'and'}
}

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

_acronym_re     = re.compile(r"\b[A-Z]+(?!')\b")
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

def replace_patterns(text, patterns, ** kwargs):
    """ `pattern` is a dict associating the word to replace (key) and its replacement (value) """
    regex = re.compile(r'(\b|\s)({})(\b|\s)'.format('|'.join(
        [re.escape(pat) for pat in patterns.keys()]
    )))
    return re.sub(regex, lambda w: ' {} '.format(patterns[w.group(0).strip()]), text)

def expand_abbreviations(text, abreviations = _english_abreviations, ** kwargs):
    return replace_patterns(text, abreviations, ** kwargs)

def expand_special_symbols(text, lang = None, symbols = None, ** kwargs):
    assert lang is not None or symbols is not None
    if symbols is None: symbols = {k : v[lang] for k, v in _special_symbols.items() if lang in v}
    return replace_patterns(text, symbols, ** kwargs)

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
    """ Replace all tokens in `tokens` (an iterable) by ' ' (space) """
    if not tokens: return text
    regex = re.compile(r'\b({})\b'.format('|'.join(tokens)))
    return re.sub(regex, ' ', text)

def attach_punctuation(text, ** kwargs):
    for punct in _left_punctuation:
        text = text.replace('{} '.format(punct), punct)
    for punct in _right_punctuation:
        text = text.replace(' {}'.format(punct), punct)
    return text

def expand_acronym(text, lang, ** kwargs):
    """ Expand all words composed of uppercases """
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
    """ Convert to ascii (with `unidecode`) while keeping some french accents """
    converted = ''
    idx = 0
    while idx < len(text):
        next_idx = min([
            text.index(a, idx) if a in text[idx:] else len(text) for a in accents_to_keep
        ])
        converted += unidecode(text[idx : next_idx])
        if next_idx < len(text): converted += text[next_idx]
        idx = next_idx + 1
    return converted

def convert_to_alnum(text, allowed_char = '.,?! ', replace_char = ' ', ** kwargs):
    """ Replace all non-alphanumeric charactes by `replace_char` """
    return ''.join([c if c.isalnum() or c in allowed_char else replace_char for c in text])

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

def complete_cleaners(text,
                      lang,
                      to_lowercase  = True,
                      to_expand     = True,
                      to_expand_symbols = True,
                      to_expand_acronyms    = False,
                      abbreviations = None,
                      replacements  = None,
                      ** kwargs
                     ):
    """
        Complete cleaners pipeline for a given language (the language is required for some processing). Note that some processing are optional (cf arguments).
        `to_expand` is for the 5th step (number + symbols expansion)
        1) Convert to ASCII
        2) Expand abbreviations
        3) Expand acronyms
        4) Lowercase
        5) Expand numbers + special symbols
        6) Collapse whitespace
    """
    if lang == 'fr':        text = fr_convert_to_ascii(text, ** kwargs)
    else:                   text = convert_to_ascii(text, ** kwargs)
    
    if abbreviations:       text = expand_abbreviations(text, ** kwargs)
    if to_expand_acronyms:  text = expand_acronym(text, lang = lang, ** kwargs)
    if to_lowercase:        text = lowercase(text, ** kwargs)
    if to_expand:
        text = expand_numbers(text, lang = lang, expand_symbols = to_expand_symbols, ** kwargs)
        if to_expand_symbols:
            text = expand_special_symbols(text, lang = lang, ** kwargs)
    text = collapse_whitespace(text, ** kwargs)
    return text

def english_cleaners(text, ** kwargs):
    return complete_cleaners(text, lang = 'en', ** kwargs)

def french_cleaners(text, ** kwargs):
    return complete_cleaners(text, lang = 'fr', ** kwargs)
