""" from https://github.com/keithito/tacotron """

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
from unidecode import unidecode
from utils.text.numbers import normalize_numbers


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

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

def expand_abbreviations(text, abreviations = _english_abreviations):
    for regex, replacement in abreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_numbers(text, langue = 'en'):
    return normalize_numbers(text, langue = langue)

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
    return unidecode(text)

def fr_convert_to_ascii(text, accents_to_keep = 'éèêç'):
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

def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text, langue='en')
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

def french_cleaners(text):
    '''Pipeline for French text, including number expansion.'''
    text = fr_convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text, langue='fr')
    text = collapse_whitespace(text)
    return text
