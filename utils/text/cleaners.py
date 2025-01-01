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
import re
import unicodedata

from unidecode import unidecode

from .numbers import normalize_numbers
from utils.wrapper_utils import partial


_special_symbols    = {
    '='     : {'fr' : 'égal',       'en' : 'equal'},
    '+'     : {'fr' : 'plus',       'en' : 'plus'},
    '/'     : {'fr' : 'slash',      'en' : 'slash'},
    '*'     : {'fr' : 'étoile',     'en' : 'star'},
    '^'     : {'fr' : 'chapeau',    'en' : 'hat'},
    '%'     : {'fr' : 'pourcent',   'en' : 'percent'},
    '§'     : {'fr' : 'paragraphe', 'en' : 'paragraph'},
    '&'     : {'fr' : 'et',         'en' : 'and'},
    '°C'    : {'fr' : 'degrés',     'en' : 'degrees'},
    '°'     : {'fr' : 'degrés',     'en' : 'degrees'}
}

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

_acronym_re     = re.compile(r"\b[A-Z]+(?!')\b")
_punctuation    = '_!?.,’“”‚‘—–()[]{}:;\'"`+-*/^=\\<>&#$%@¿′″·§~'
_left_punctuation   = '([{'
_right_punctuation  = ')]},.'
_accents    = "âéèêîç"


# List of (regular expression, replacement) pairs for abbreviations:
_abreviations   = {}

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

def get_cleaners_fn(cleaners):
    """
        Returns a `list` of `callable`, the functions associated to the given `cleaners`
        
        Arguments :
            - cleaners  : a list of cleaner
                - str   : cleaner name
                - dict  : cleaner config with a `name` for the name of the cleaner
                - tuple : `(name, kwargs)`
                - callable  : cleaning function
            Return :
                - cleaners_fn   : `list` of `callable`
    """
    if not isinstance(cleaners, (list, tuple)): cleaners = [cleaners]
    cleaners_fn    = []
    for name in cleaners:
        kwargs = None
        if isinstance(name, tuple):
            name, kwargs = name
        elif isinstance(name, dict):
            name, kwargs = name['name'], {k : v for k, v in name.items() if k != 'name'}
        
        cleaner = globals().get(name, None)
        
        if cleaner is None:
            raise ValueError("Unknown cleaner : {}".format(name))
        elif not callable(cleaner):
            raise ValueError("Cleaner must be callable : {}".format(name))
        
        cleaners_fn.append(cleaner if not kwargs else partial(cleaner, ** kwargs))
    
    return cleaners_fn

def clean_text(text, cleaners, tokens = {}, ** kwargs):
    """ Cleans `text` with the list of `cleaners` (see `get_cleaners_fn`) """
    if not cleaners: return text
    
    text = text
    for cleaner in cleaners:
        text = cleaner(text, ** kwargs)
    
    for cleaned, token in tokens.items():
        text = text.replace(cleaned, token)
    
    return text

def get_abreviations(lang):
    if lang not in _abreviations:
        from utils.file_utils import load_json
        
        filename = os.path.join(
            __package__.replace('.', '/'), 'abreviations', lang + '.json'
        )

        _abreviations[lang] = load_json(filename, default = {})
    return _abreviations[lang]

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
    """ Replaces a `dict` of `{pattern : replacement}` """
    for pattern, repl in patterns.items():
        text = re.sub(pattern, repl, text)
    return text

def replace_words(text,
                  words,
                  pattern_format = r'\b({})\b',
                  getter    = None,
                  flags = re.IGNORECASE,
                  ** kwargs
                 ):
    """ `pattern` is a dict associating the word to replace (key) and its replacement (value) """
    _text_lower = text.lower()
    _min_patterns   = {k.lower() : v for k, v in words.items()}
    _min_patterns   = {k : v for k, v in _min_patterns.items() if k in _text_lower}
    if not _min_patterns: return text
    
    regex = re.compile(pattern_format.format('|'.join([
        re.escape(pat) for pat in words.keys()
    ])), flags)
    
    if getter is None: getter = lambda w: _min_patterns[w.group(0).lower()]
    
    return re.sub(regex, getter, text)

def expand_abreviations(text, abreviations = None, lang = None, ** kwargs):
    assert abreviations is not None or lang is not None
    
    if abreviations is None: abreviations = get_abreviations(lang)
    
    return replace_words(
        text,
        abreviations,
        pattern_format = r'\b({})(\.|\b)',
        getter  = lambda ab: abreviations[ab.group(0).lower().rstrip('.')]
    )

def expand_special_symbols(text, lang = None, symbols = None, ** kwargs):
    assert lang is not None or symbols is not None
    
    if symbols is None: symbols = {k : v[lang] for k, v in _special_symbols.items() if lang in v}
    
    for symbol, repl in symbols.items():
        text = text.replace(symbol, ' ' + repl + ' ')
    
    return text

def remove_tokens(text, tokens = [], ** kwargs):
    """ Replace all tokens in `tokens` (an iterable) by ' ' (space) """
    if not tokens: return text
    return replace_words(text, {tok : '' for tok in tokens})

def remove_markdown(text):
    return re.sub(r'\*\*(.*)\*\*', r'\1', text)

def _expand_acronym(text, lang, extensions = _letter_pronounciation, ** kwargs):
    if len(text) > 4 or (text == 'I' and lang == 'en'): return text
    return ' '.join([extensions.get(c.lower(), {}).get(lang, c) for c in text])

def expand_acronym(text, lang, ** kwargs):
    """ Expand all words composed of uppercases """
    return re.sub(_acronym_re, lambda m: _expand_acronym(m.group(0), lang), text)

def detach_punctuation(text, punctuation = _punctuation, ** kwargs):
    for punct in punctuation:
        text = text.replace(punct, ' {} '.format(punct))
    return text.strip()

def remove_punctuation(text, punctuation = _punctuation, ** kwargs):
    return ''.join(c for c in text if c not in punctuation)

def attach_punctuation(text, ** kwargs):
    for punct in _left_punctuation:
        text = text.replace('{} '.format(punct), punct)
    for punct in _right_punctuation:
        text = text.replace(' {}'.format(punct), punct)
    return text

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

def fr_convert_to_ascii(text, accepted = _accents, ** kwargs):
    """ Convert to ascii (with `unidecode`) while keeping some french accents """
    converted = ''
    idx = 0
    while idx < len(text):
        next_idx = min([
            text.index(a, idx) if a in text[idx:] else len(text) for a in accepted
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
                      to_expand_abrev   = True,
                      to_expand_symbols = True,
                      to_expand_acronyms    = False,
                      replacements  = None,
                      patterns  = None,
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
    if patterns:            text = replace_patterns(text, patterns, ** kwargs)
    if replacements:        text = replace_words(text, replacements, ** kwargs)
    if to_lowercase:        text = lowercase(text, ** kwargs)
    if to_expand:
        text = remove_markdown(text)
        if to_expand_abrev:     text = expand_abreviations(text, lang = lang, ** kwargs)
        text = expand_numbers(text, lang = lang, expand_symbols = to_expand_symbols, ** kwargs)
        if to_expand_symbols:   text = expand_special_symbols(text, lang = lang, ** kwargs)
    
    if lang == 'fr':        text = fr_convert_to_ascii(text, ** kwargs)
    else:                   text = convert_to_ascii(text, ** kwargs)

    text = collapse_whitespace(text, ** kwargs)
    return text

english_cleaners = partial(complete_cleaners, lang = 'en')
french_cleaners  = partial(complete_cleaners, lang = 'fr')
