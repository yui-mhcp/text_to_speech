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

import re

from functools import cache

_lang = 'en'

_comma_extended     = {
    'fr' : 'virgule', 'en' : 'punt'
}
_math_symbols   = {
    '=' : {'fr' : 'égal',       'be' : 'égal',       'en' : 'equal'},
    '+' : {'fr' : 'plus',       'be' : 'plus',       'en' : 'plus'},
    '-' : {'fr' : 'moins',      'be' : 'moins',      'en' : 'minus'},
    '*' : {'fr' : 'fois',       'be' : 'fois',       'en' : 'times'},
    '/' : {'fr' : 'divisé par', 'be' : 'divisé par', 'en' : 'divide by'},
    '^' : {'fr' : 'exposant',   'be' : 'exposant',   'en' : 'exponent'}
}
_join_symbols   = '|'.join([re.escape(symbol) for symbol in _math_symbols])
_time_extended  = {
    'h'     : {'fr' : 'heure',     'be' : 'heure',      'en' : 'hour'},
    'min'   : {'fr' : 'minute',    'be' : 'minute',     'en' : 'minute'},
    'sec'   : {'fr' : 'seconde',   'be' : 'seconde',    'en' : 'second'},
    's'     : {'fr' : 'seconde',    'be' : 'seconde',   'en' : 'second'},
    '_sep'  : {'fr' : ' et ', 'be' : ' et ', 'en' : ' and '}
}

_units  = {
    'l'     : {'fr' : 'litre',  'en'    : 'litre'},
    
    'g'     : {'fr' : 'gramme', 'en'    : 'gram'},
    't'     : {'fr' : 'tonne',  'en'    : 'tonne'},
    
    'm'     : {'fr' : 'mètre',  'en'    : 'meter'},
    'mi'    : {'fr' : 'mile',   'en'    : 'mile'},
    'o'     : {'fr' : 'octet',  'en'    : 'bytes'},
    'b'     : {'fr' : 'bit',    'en'    : 'bit'},
    
    'V'     : {'fr' : 'volt',   'en'    : 'volt'},
    'W'     : {'fr' : 'watt',   'en'    : 'watt'},
    'A'     : {'fr' : 'ampère', 'en'    : 'ampere'},
    'Hz'    : {'fr' : 'hertz',  'en'    : 'hertz'},
    
    'J'     : {'fr' : 'joule',  'en'    : 'joul'},
    'N'     : {'fr' : 'newton', 'en'    : 'newton'},
    'b'     : {'fr' : 'bar',    'en'    : 'bar'}

}
_unit_prefix    = {
    'n' : {'fr' : 'nano',   'en'    : 'nano'},
    'm' : {'fr' : 'mili',   'en'    : 'mili'},
    'c' : {'fr' : 'centi',  'en'    : 'centi'},
    'd' : {'fr' : 'déci',   'en'    : 'deci'},
    'k' : {'fr' : 'kilo',   'en'    : 'kilo'},
    'M' : {'fr' : 'méga',   'en'    : 'mega'},
    'G' : {'fr' : 'giga',   'en'    : 'giga'},
    'T' : {'fr' : 'tera',   'en'    : 'tera'}
}
_units_sep = {'fr' : 'par',    'en'    : 'per'}

_units_re   = re.compile(
    r'(\d+)\s*({})?({})(?:\/({}))\b'.format(
        '|'.join(_unit_prefix.keys()), '|'.join(_units.keys()), '|'.join(_time_extended)
    )
)

_math_symbol_re = re.compile(
    r'(?:(?<=\d)(\s*[\+\*\/\^\=]\s*(\+|\-\s*)?)(?=\d)|((?:^|\s+)(\-|\+)\s*(\+|\-\s*)?)(?=\d))'
)

_sec_pattern    = r'(\d+)\s*(?:sec|s)\b'
_min_pattern    = r'(\d+)\s*min(?:\s*{})?'.format(_sec_pattern)
_hours_pattern  = r'(\d+)\s*h\s*(?:{}|{})?'.format(_min_pattern, _sec_pattern)

_time_re    = re.compile(
    r'\b(?:{}|{}|{})\b'.format(_hours_pattern, _min_pattern, _sec_pattern)
)
_clock_re   = re.compile(r'(\d{1,2}):(\d{1,2}):(\d{1,2})')

_comma_number_re    = re.compile(r'([0-9][0-9\,]+[0-9])')
_space_number_re    = re.compile(r'[0-9]+( [0-9]{3,3})+(?!\d)')
_tiret_number_re    = re.compile(r'([0-9]+-[0-9])')

_pounds_re          = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re         = re.compile(r'\$([0-9\.\,]*[0-9]+)')

_decimal_number_re  = re.compile(r'([0-9]+\.[0-9]+)')
_number_re          = re.compile(r'[0-9]+')

_ordinal_re = re.compile(r'([0-9]+)(st|nd|rd|th|er|ère|ème|eme|ième|ieme)')

@cache
def num2words(number, lang, ordinal = False):
    from num2words import num2words as _num2words
    text = _num2words(number, ordinal = ordinal, lang = lang if lang != 'be' else 'fr')
    if lang == 'be':
        for prefix, new in (('soixante-', 'septante'), ('quatre-vingt-', 'nonante')):
            if prefix not in text: continue
            if ordinal:
                text = text \
                    .replace(f'{prefix}onz',    f'{new} et un')  \
                    .replace(f'{prefix}douz',   f'{new} deux')  \
                    .replace(f'{prefix}treiz',      f'{new} trois')    \
                    .replace(f'{prefix}quatorz',    f'{new} quatre') \
                    .replace(f'{prefix}quinz',      f'{new} cinqu') \
                    .replace(f'{prefix}seiz',       f'{new} six')   \
                    .replace(f'{prefix}dix-sept',   f'{new} sept')   \
                    .replace(f'{prefix}dix-huit',   f'{new} huit')   \
                    .replace(f'{prefix}dix-neuv',   f'{new} neuv')   \
                    .replace(f'{prefix}dix',    new[:-1])
            else:
                text = text \
                    .replace(f'{prefix}onze',   f'{new} et un')  \
                    .replace(f'{prefix}douze',  f'{new} deux')  \
                    .replace(f'{prefix}treize',     f'{new} trois')    \
                    .replace(f'{prefix}quatorze',   f'{new} quatre') \
                    .replace(f'{prefix}quinze',     f'{new} cinq') \
                    .replace(f'{prefix}seize',      f'{new} six')   \
                    .replace(f'{prefix}dix-sept',   f'{new} sept')   \
                    .replace(f'{prefix}dix-huit',   f'{new} huit')   \
                    .replace(f'{prefix}dix-neuf',   f'{new} neuf')   \
                    .replace(f'{prefix}dix',    new)
                
    return text

def _expand_units(m, lang = _lang):
    if lang == 'be': lang = 'fr'
    n, prefix, unit, per_time = m.groups()
    if n == '1' and lang == 'fr' and unit == 't': n = 'une'
    prefix = _unit_prefix[prefix][lang] if prefix else ''

    text = n + ' ' + prefix + _units[unit][lang]
    if n != 'une' and n > '1': text += 's'
    
    if per_time:    text += ' ' + _units_sep[lang] + ' ' + _time_extended[per_time][lang]
    return text

def _expand_math_symbols(m, lang = _lang):
    parts = [_math_symbols[symbol][lang] for symbol in m.group(0).split()]
    return ' ' + ' '.join(parts) + ' '

def _expand_time(m, lang = _lang):
    groups = m.groups()

    h = groups[0]
    m = groups[1] or groups[4]
    s = groups[2] or groups[3] or groups[5] or groups[6]
    parts = []
    for (t, unit) in ((h, 'h'), (m, 'min'), (s, 'sec')):
        if t is None: continue

        unit = _time_extended[unit][lang]
        if int(t) > 1: unit += 's'
        elif lang == 'fr' and int(t) == 1: t = 'une'
        parts.append('{} {}'.format(t, unit))
    
    return _time_extended['_sep'][lang].join(parts)

def _expand_clock(m, lang = _lang):
    h, m, s = m.groups()

    parts = []
    for (t, unit) in ((h, 'h'), (m, 'min'), (s, 'sec')):
        unit = _time_extended[unit][lang]
        if int(t) > 1: unit += 's'
        elif lang == 'fr' and int(t) == 1: t = 'une'
        parts.append('{} {}'.format(t, unit))
    
    return _time_extended['_sep'][lang].join(parts)

def _remove_commas(m, lang = _lang):
    """
        In French, decimal numbers are often noted with coma (e.g., 3,14) whereas in English, comma is used to delimitate groups of 3 numbers in large numbers (e.e., 3,000,000.14)
        In the first case, it is replaced by "." to be later handled as decimal
        In the second case, it is simply removed to be handled as large number
    """
    if lang == 'fr' and m.group(1).count(',') == 1:
        return m.group(1).replace(',', '.')
    return m.group(1).replace(',', '')

def _expand_tiret(m):
    return m.group(1).replace('-', ' - ')

def _remove_space(m):
    """ Remove space sometimes used in large numbers (e.g., 1 000 000) """
    return m.group(0).replace(' ', '')

def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2: return match + ' dollars' # unexpected

    dollars = int(parts[0]) if parts[0] else 0
    cents   = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        return '{} dollar{}, {} cent{}'.format(
            dollars, 's' if dollars != 1 else '', cents, 's' if cents != 1 else ''
        )
    elif dollars:
        return '{} dollar{}'.format(dollars, 's' if dollars != 1 else '')
    elif cents:
        return '{} cent{}'.format(cents, 's' if cents != 1 else '')
    else:
        return 'zero dollars'


def _expand_ordinal(m, lang = _lang):
    return num2words(m.group(1), lang = lang, ordinal = True)

def _extend_with_zeros(text, lang = _lang):
    n = 0
    while n < len(text) and text[n] == '0':
        n += 1
    
    to_text = num2words(text, lang = lang)
    if n == 0: return to_text
    elif n < 4: return '{} {}'.format(' '.join([num2words('0', lang = lang)] * n), to_text)
    return '{} {} {}'.format(
        num2words(str(n), lang = lang),
        _math_symbols['*'].get(lang, ''),
        num2words('0', lang = lang),
        to_text
    )

def _expand_number(m, lang = _lang, decimal_as_individual = None):
    if decimal_as_individual: decimal_as_individual = lang == 'en'
    
    num = m.group(0)
    if '.' not in num or decimal_as_individual:
        return num2words(num, lang = lang)
    else:
        ent, dec = num.split('.')

        if dec.count('0') == len(dec):
            return num2words(num, lang = lang)
        else:
            return '{} {} {}'.format(
                num2words(ent, lang = lang),
                _comma_extended.get(lang, ''),
                _extend_with_zeros(dec, lang = lang)
            )

def normalize_numbers(text, lang = _lang, expand_symbols = True, ** kwargs):
    if expand_symbols:
        text = re.sub(_units_re,        lambda m: _expand_units(m, lang), text)
        text = re.sub(_math_symbol_re,  lambda m: _expand_math_symbols(m, lang), text)
    text = re.sub(_time_re,         lambda m: _expand_time(m, lang), text)
    text = re.sub(_clock_re,        lambda m: _expand_clock(m, lang), text)
    
    text = re.sub(_comma_number_re, lambda m: _remove_commas(m, lang), text)
    text = re.sub(_tiret_number_re, _expand_tiret, text)
    text = re.sub(_space_number_re, _remove_space, text)
    
    text = re.sub(_pounds_re,       r'\1 pounds', text)
    text = re.sub(_dollars_re,      _expand_dollars, text)
    
    text = re.sub(_decimal_number_re,   lambda m: _expand_number(m, lang), text)
    text = re.sub(_ordinal_re,  lambda m: _expand_ordinal(m, lang), text)
    text = re.sub(_number_re,   lambda m: _expand_number(m, lang),  text)
    
    return text
