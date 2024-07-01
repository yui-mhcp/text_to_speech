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

import re

from functools import cache
from num2words import num2words as _num2words

_lang = 'en'

_comma_extended     = {
    'fr' : 'virgule', 'en' : 'punt'
}
_math_symbols   = {
    '=' : {'fr' : 'égal',       'en' : 'equal'},
    '+' : {'fr' : 'plus',       'en' : 'plus'},
    '-' : {'fr' : 'moins',      'en' : 'minus'},
    '*' : {'fr' : 'fois',       'en' : 'times'},
    '/' : {'fr' : 'sur',        'en' : 'divide by'},
    '^' : {'fr' : 'exposant',   'en' : 'exponent'}
}
_join_symbols   = '|'.join([re.escape(symbol) for symbol in _math_symbols])
_time_extended  = {
    'h'     : {'fr' : 'heure',     'en' : 'hour'},
    'min'   : {'fr' : 'minute',    'en' : 'minute'},
    'sec'   : {'fr' : 'seconde',   'en' : 'second'},
    '_sep'  : {'fr' : 'et', 'en' : 'and'}
}

_ordinal_re = {
    'en'    : re.compile(r'[0-9]+(st|nd|rd|th)'),
    'fr'    : re.compile(r'[0-9]+(er|ère|ème|eme|ième|ieme)')

}

_clock_pattern  = r'\d{1,2}:\d{1,2}:\d{1,2}'
_sec_pattern    = r'\d+\s*(sec|s)'
_min_pattern    = r'\d+\s*min(\s*{})?'.format(_sec_pattern)
_hours_pattern  = r'(\d+\s*h(\s*{})?|{})'.format(_min_pattern, _clock_pattern)

_time_re            = re.compile(
    r'\b({}|{}|{})\b'.format(_hours_pattern, _min_pattern, _sec_pattern)
)
_zero_re            = re.compile(r'^0*')
_comma_number_re    = re.compile(r'([0-9][0-9\,]+[0-9])')
_space_number_re    = re.compile(r'[0-9]+( [0-9]{3,3})+(?!\d)')
_tiret_number_re    = re.compile(r'([0-9]+-[0-9])')
_decimal_number_re  = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re          = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re         = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_number_re          = re.compile(r'[0-9]+')
_math_symbol_re     = re.compile(
    r'[0-9]+ *({}) *((\+|\-) *)?(?=[0-9]+)'.format(_join_symbols, _join_symbols)
)

@cache
def num2words(* args, ** kwargs):
    return _num2words(* args, ** kwargs)

def _expand_math_symbols(m, lang = None):
    if lang is None:
        global _lang
        lang = _lang

    text = m.group(0)
    for s, extended in _math_symbols.items():
        if lang not in extended: continue
        text = text.replace(s, ' {} '.format(extended[lang]))
    return text

def _expand_time(m, lang = None):
    if lang is None:
        global _lang
        lang = _lang

    text = m.group(0)
    if ':' in text:
        text = ['{} {}{}'.format(
            t if lang != 'fr' or t != '1' else 'une',
            _time_extended[t_unit][lang],
            's' if t > '1' else ''
        ) for t, t_unit in zip(text.split(':'), ('h', 'min', 'sec'))]
        
        if lang in _time_extended['_sep']: text.insert(-1, _time_extended['_sep'][lang])
        return ' '.join(text)

    for unit in ('h', 'min', 'sec'):
        text = text.replace(unit, ' {} '.format(_time_extended[unit][lang]))

    text = text.split()
    for i in range(0, len(text), 2):
        if text[i] > '1': text[i + 1] += 's'
        elif lang == 'fr' and text[i] == '1': text[i] = 'une'

    if len(text) > 2 and lang in _time_extended['_sep']:
        text.insert(-2, _time_extended['_sep'][lang])
    
    return ' '.join(text)

def _remove_commas(m, lang = None):
    if lang is None:
        global _lang
        lang = _lang

    if lang == 'fr':
        return m.group(1).replace(',', '.')
    return m.group(1).replace(',', '')

def _expand_tiret(m):
    return m.group(1).replace('-', ' - ')

def _remove_space(m):
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


def _expand_ordinal(m, lang = None):
    if lang is None:
        global _lang
        lang = _lang

    num = m.group(0)
    num = _number_re.match(num).group(0)
    return num2words(num, lang = lang, to = 'ordinal')

def _extend_with_zeros(text, lang):
    n = len(_zero_re.match(text).group(0))
    to_text = num2words(text, lang = lang)
    if n == 0: return to_text
    elif n < 4: return '{} {}'.format(' '.join([num2words('0', lang = lang)] * n), to_text)
    return '{} {} {}'.format(
        num2words(str(n), lang = lang), _math_symbols['*'].get(lang, ''), to_text
    )

def _expand_number(m, lang = None, decimal_as_individual = None):
    if lang is None:
        global _lang
        lang = _lang

    if decimal_as_individual: decimal_as_individual = lang == 'en'
    
    num = m.group(0)
    if '.' not in num or decimal_as_individual:
        words = num2words(num, lang = lang)
    else:
        ent, dec = num.split('.')

        if dec.count('0') == len(dec):
            words = num2words(num, lang = lang)
        else:
            words = '{} {} {}'.format(
                num2words(ent, lang = lang), _comma_extended.get(lang, ''), _extend_with_zeros(dec, lang = lang)
            )
    return words

def normalize_numbers(text, lang = None, expand_symbols = False, ** kwargs):
    global _lang
    if lang is None:
        lang = _lang
    else:
        _lang = lang
    
    if expand_symbols:
        text = re.sub(_math_symbol_re,  _expand_math_symbols, text)
    text = re.sub(_time_re,         _expand_time, text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_tiret_number_re, _expand_tiret, text)
    text = re.sub(_space_number_re, _remove_space, text)
    text = re.sub(_pounds_re,       r'\1 pounds', text)
    text = re.sub(_dollars_re,      _expand_dollars, text)
    text = re.sub(_decimal_number_re,   _expand_number, text)
    
    if lang in _ordinal_re:
        text = re.sub(_ordinal_re[lang],    _expand_ordinal, text)
    
    text = re.sub(_number_re,       _expand_number, text)
    return text
