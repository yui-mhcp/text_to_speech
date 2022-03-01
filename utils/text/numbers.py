
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

""" inspired from https://github.com/keithito/tacotron """

import re

from num2words import num2words

_lang = 'en'

_time_re            = re.compile(r'([0-9]+h|[0-9]+min|[0-9]+sec|[0-9]*:[0-9]{1,2}:[0-9]{1,2})')
_zero_re            = re.compile(r'^0*')
_comma_number_re    = re.compile(r'([0-9][0-9\,]+[0-9])')
_space_number_re    = re.compile(r'([0-9]+ [0-9]{3}!\d)')
_tiret_number_re    = re.compile(r'([0-9]+-[0-9])')
_decimal_number_re  = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re          = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re         = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re_en      = re.compile(r'[0-9]+(st|nd|rd|th)')
_ordinal_re_fr      = re.compile(r'[0-9]+(er|ère|ème|eme|ième|ieme)')
_number_re          = re.compile(r'[0-9]+')
_math_symbol_re     = re.compile(r'((([0-9]\s)?(\+|-|\*|/)\s?[0-9])|([0-9]\s?(\+|-|\*|/)\s?[0-9]?))')

_comma_extended     = {
    'fr' : 'virgule', 'en' : 'comma'
}
_special_symbols    = {
    '='     : {'fr' : ' egal ',     'en' : ' equal '},
    '+'     : ' plus ',
    '/'     : ' slash ',
    '%'     : {'fr' : ' pourcent ', 'en' : ' percent '},
    '§'     : {'fr' : ' paragraphe ', 'en' : ' paragraph '},
    '&'     : {'fr' : ' et ', 'en' : ' and '}
}
_math_symbols   = {
    '='   : {'fr' : ' egal ',     'en' : ' equal '},
    '+'   : {'fr' : ' plus ',     'en' : ' plus '},
    '-'   : {'fr' : ' moins ',    'en' : ' minus '},
    '*'   : {'fr' : ' fois ',     'en' : ' times '},
    '/'   : {'fr' : ' sur ',      'en' : ' divide by '},
}
_time_extended    = {
    'h' : {'fr' : 'heures', 'en' : 'hours'},
    'min'   : {'fr' : 'minutes', 'en' : 'minutes'},
    'sec'   : {'fr' : 'secondes', 'en' : 'seconds'}
}

def _expand_math_symbols(m, lang = None):
    if lang is None:
        global _lang
        lang = _lang
    
    text = m.group(0)
    for s, extended in _math_symbols.items():
        if lang in extended: extended = extended[lang]
        text = text.replace(s, extended)
    return text

def _expand_special_symbols(text, lang = None):
    if lang is None:
        global _lang
        lang = _lang
    
    for s, extended in _special_symbols.items():
        if lang in extended: extended = extended[lang]
        text = text.replace(s, extended)
    return text

def _expand_time(m, lang = None):
    if lang is None:
        global _lang
        lang = _lang

    txt = m.group(1)
    if ':' in txt:
        txt = txt.split(':')
        return ' '.join([
            t + ' ' + _time_extended[pat][lang] for t, pat in zip(txt, ['h', 'min', 'sec'])
        ])
    
    for pat, ext in _time_extended.items():
        if txt.endswith(pat):
            return txt.replace(pat, ' {}'.format(ext[lang]))
    return txt
    

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


def _expand_ordinal(m, lang = None):
    if lang is None:
        global _lang
        lang = _lang

    num = m.group(0)
    num = _number_re.match(num).group(0)
    return num2words(num, lang = lang, to = 'ordinal')

def _expand_number(m, lang = None, decimal_as_individual = False):
    def _extend_with_zeros(text):
        n = len(_zero_re.match(text).group(0))
        to_text = num2words(text, lang = lang)
        if n == 0: return to_text
        elif n < 4: return '{} {}'.format(' '.join([num2words('0', lang = lang)] * n), to_text)
        return '{} {} {}'.format(
            num2words(str(n), lang = lang), _math_symbols['*'].get(lang, ''), to_text
        )

    if lang is None:
        global _lang
        lang = _lang

    num = m.group(0)
    if '.' not in num or decimal_as_individual:
        words = num2words(num, lang = lang)
    else:
        ent, dec = num.split('.')

        if dec.count('0') == len(dec):
            words = num2words(num, lang = lang)
        else:
            words = '{} {} {}'.format(
                num2words(ent, lang = lang), _comma_extended.get(lang, ''), _extend_with_zeros(dec)
            )
    return '{}'.format(words)


def normalize_numbers(text, lang = None, ** kwargs):
    global _lang
    if lang is None:
        lang = _lang
    else:
        _lang = lang
    
    text = re.sub(_math_symbol_re, _expand_math_symbols, text)
    text = _expand_special_symbols(text, lang = lang)
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_tiret_number_re, _expand_tiret, text)
    text = re.sub(_space_number_re, _remove_space, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_number, text)
    
    if lang == 'en':
        text = re.sub(_ordinal_re_en, _expand_ordinal, text)
    elif lang == 'fr':
        text = re.sub(_ordinal_re_fr, _expand_ordinal, text)
    
    text = re.sub(_number_re, _expand_number, text)
    return text
