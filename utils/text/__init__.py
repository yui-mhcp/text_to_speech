from utils.text import cmudict

from utils.text.text_encoder import TextEncoder, CHAR_LEVEL, TOKEN_LEVEL, WORD_LEVEL
from utils.text.text_processing import *

_pad            = '_'
_punctuation    = '!\'(),.:;? '
_special    = '-'
_maj_letters    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
_min_letters    = 'abcdefghijklmnopqrstuvwxyz'
_letters    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_accents    = 'àéèêîçôù'
_maths      = '+*/'
_numbers    = '0123456789'

_mini_punctuation   = ' \',.?!'
_mini_accents       = 'éèç'

#Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
en_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet
fr_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_accents)

accent_replacement_matrix = {
    'a' : {'à' : 0, 'â' : 0}, 'à' : {'a' : 0, 'â' : 0}, 'â' : {'a' : 0, 'à' : 0},
    'u' : {'ù' : 0}, 'ù' : {'u' : 0},
    'o' : {'ô' : 0}, 'ô' : {'o' : 0},
    'i' : {'î' : 0}, 'î' : {'i' : 0}
}

def get_symbols(lang,
                punctuation = 1,
                maj     = True,
                arpabet = True, 
                accents = True,
                numbers = False
               ):
    symbols = [_pad] + list(_special)
    
    if punctuation: 
        symbols += list(_punctuation) if punctuation == 1 else list(_mini_punctuation)
    else: symbols += [' ']
    
    symbols += list(_letters) if maj else list(_min_letters)
    
    if lang == 'en' and arpabet: symbols += _arpabet
    if lang == 'fr' and accents: symbols += list(_accents)
    
    if numbers: symbols += _numbers
    
    return symbols

def default_encoder(lang, ** kwargs):
    lang = lang.lower()
    if lang in ('fr', 'francais', 'français', 'french'):
        return default_french_encoder(** kwargs)
    elif lang in ('en', 'english', 'anglais'):
        return default_english_encoder(** kwargs)
    else:
        print("Unknown language : {} - return char-level encoder with default symbols".format(lang))
        return TextEncoder(get_symbols(lang), level = 'char', ** kwargs)

def default_english_encoder(cleaners = ['english_cleaners'], level = 'char', ** kwargs):
    return TextEncoder(en_symbols, level = level, cleaners = cleaners, ** kwargs)

def default_french_encoder(cleaners = ['french_cleaners'], level = 'char', ** kwargs):
    return TextEncoder(fr_symbols, level = level, cleaners = cleaners, ** kwargs)

