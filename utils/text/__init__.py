from utils.text import cmudict

from utils.text.text_encoder import TextEncoder
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

def get_symbols(langue, ponctuation = True, maj = True, arpabet = True, 
                accents = True, numbers = False):
    symbols = [_pad] + list(_special)
    if ponctuation: 
        symbols += list(_punctuation) if ponctuation == 1 else list(_mini_punctuation)
    else: symbols += [' ']
    symbols += list(_letters) if maj else list(_min_letters)
    if langue == 'en' and arpabet: symbols += _arpabet
    if langue == 'fr' and accents: 
        symbols += list(_accents) if accents == 1 else list(_mini_accents)
    if numbers: symbols += _numbers
    return symbols

def default_encoder(langue, ** kwargs):
    langue = langue.lower()
    if langue in ('fr', 'francais', 'français', 'frensch'):
        return default_french_encoder(** kwargs)
    elif langue in ('en', 'english', 'anglais'):
        return default_english_encoder(** kwargs)
    else:
        print("Langue d'encoder inconnue : {} - renvoi de l'encoder par défaut".format(langue))
        return TextEncoder(get_symbols(langue), word_level = False, **kwargs)

def default_english_encoder(cleaners = ['english_cleaners'], word_level = False, **kwargs):
    return TextEncoder(en_symbols, word_level = word_level, cleaners = cleaners, **kwargs)

def default_french_encoder(cleaners = ['french_cleaners'], word_level = False, **kwargs):
    return TextEncoder(fr_symbols, word_level = word_level, cleaners = cleaners, **kwargs)

