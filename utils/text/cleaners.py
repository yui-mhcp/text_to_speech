""" inspired from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
import unicodedata

from unidecode import unidecode
from utils.text.numbers import normalize_numbers


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

_punctuation    = '_!?.,’“”‚‘—–()[]{}:;\'"`+-*/^=\\<>&#$%@¿′″·§~'
_left_punctuation   = '([{'
_right_punctuation  = ')]},.'
_accents    = "âéèêîçô"

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

def strip(text, lstrip = True, rstrip = True):
    if lstrip and rstrip: return text.strip()
    elif lstrip: return text.lstrip()
    elif rstrip: return text.rstrip()
    return text

def lstrip(text):
    return text.lstrip()

def rstrip(text):
    return text.rstrip()

def expand_abbreviations(text, abreviations = _english_abreviations):
    for regex, replacement in abreviations:
        text = re.sub(regex, replacement, text)
    return text

def detach_punctuation(text, punctuation = _punctuation):
    for punct in punctuation:
        text = text.replace(punct, ' {} '.format(punct))
    return text.strip()

def remove_punctuation(text, punctuation = _punctuation):
    return ''.join(c for c in text if c not in punctuation)

def remove_tokens(text, tokens = None):
    if not tokens: return text
    regex = re.compile(r'\b({})\b'.format('|'.join(tokens)))
    text = re.sub(regex, ' ', text)
    return text

def attach_punctuation(text):
    for punct in _left_punctuation:
        text = text.replace('{} '.format(punct), punct)
    for punct in _right_punctuation:
        text = text.replace(' {}'.format(punct), punct)
    return text

def expand_numbers(text, langue = 'en'):
    return normalize_numbers(text, langue = langue)

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)
    
def remove_control(text):
    return "".join([
        c for c in text if c in ('\t', '\n', '\r', ' ') or c.isalnum()
        or not unicodedata.category(c).startswith('C')
    ])

def remove_accents(text):
    text = unicodedata.normalize("NFD", text)
    return ''.join([c for c in text if unicodedata.category(c) != "Mn"])

def convert_to_ascii(text):
    return unidecode(text)

def fr_convert_to_ascii(text, accents_to_keep = _accents):
    converted = []
    for c in text:
        converted.append(unidecode(c) if c not in accents_to_keep else c)
    return ''.join(converted)

def convert_to_alnum(text, allowed_char = '.,?! ', replace_char = ' '):
    new_text = ''
    for c in text:
        if c.isalnum() or c in allowed_char: new_text += c
        else: new_text += replace_char
    return new_text

def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def english_cleaners(text, to_lowercase = True, to_expand = True):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    if to_lowercase: text = lowercase(text)
    if to_expand: text = expand_numbers(text, langue='en')
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

def french_cleaners(text, to_lowercase = True, to_expand = True):
    '''Pipeline for French text, including number expansion.'''
    text = fr_convert_to_ascii(text)
    if to_lowercase: text = lowercase(text)
    if to_expand: text = expand_numbers(text, langue='fr')
    text = collapse_whitespace(text)
    return text
