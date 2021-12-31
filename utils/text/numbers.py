""" inspired from https://github.com/keithito/tacotron """

import re

from num2words import num2words

_langue = 'en'

_time_re            = re.compile(r'([0-9]+h|[0-9]+min|[0-9]+sec|[0-9]*:[0-9]{1,2}:[0-9]{1,2})')
_comma_number_re    = re.compile(r'([0-9][0-9\,]+[0-9])')
_space_number_re    = re.compile(r'([0-9]+ [0-9]{3}!\d)')
_tiret_number_re    = re.compile(r'([0-9]+-[0-9])')
_decimal_number_re  = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re          = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re         = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re_en      = re.compile(r'[0-9]+(st|nd|rd|th)')
_ordinal_re_fr      = re.compile(r'[0-9]+(er|ère|ème|eme|ième|ieme)')
_number_re          = re.compile(r'[0-9]+')

_math_symbols   = {
    ' = '   : {'fr' : ' egal ',     'en' : ' equal '},
    ' + '   : ' plus ',
    ' - '   : {'fr' : ' moins ',    'en' : ' minus '},
    ' * '   : {'fr' : ' fois ',     'en' : ' times '},
    ' / '   : {'fr' : ' sur ',      'en' : ' divide by '},
    '% '    : {'fr' : ' pourcent ', 'en' : ' percent '},
}
_time_extended    = {
    'h' : {'fr' : 'heures', 'en' : 'hours'},
    'min'   : {'fr' : 'minutes', 'en' : 'minutes'},
    'sec'   : {'fr' : 'secondes', 'en' : 'seconds'}
}

def _expand_math_symbols(text):
    for s, extended in _math_symbols.items():
        if _langue in extended: extended = extended[_langue]
        text = text.replace(s, extended)
    return text

def _expand_time(m):
    txt = m.group(1)
    if ':' in txt:
        txt = txt.split(':')
        return ' '.join([t + ' ' + _time_extended[pat][_langue] for t, pat in zip(txt, ['h', 'min', 'sec'])])
    
    for pat, ext in _time_extended.items():
        if txt.endswith(pat):
            return txt.replace(pat, ' {}'.format(ext[_langue]))
    return txt
    

def _remove_commas(m):
    if _langue == 'fr':
        return m.group(1).replace(',', '.')
    return m.group(1).replace(',', '')

def _expand_tiret(m):
    return m.group(1).replace('-', ' - ')

def _remove_space(m):
    return m.group(1).replace(' ', '')

def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    num = m.group(0)
    num = _number_re.match(num).group(0)
    return num2words(num, lang = _langue, to = 'ordinal')

def _expand_number(m):
    num = m.group(0)
    return num2words(num, lang = _langue)


def normalize_numbers(text, langue = 'en'):
    global _langue
    _langue = langue
    
    text = _expand_math_symbols(text)
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_tiret_number_re, _expand_tiret, text)
    text = re.sub(_space_number_re, _remove_space, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_number, text)
    if langue == 'en':
        text = re.sub(_ordinal_re_en, _expand_ordinal, text)
    elif langue == 'fr':
        text = re.sub(_ordinal_re_fr, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
