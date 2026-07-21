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

""" Pure-``pytest`` tests for :mod:`utils.text.numbers`.

Ported from the original ``absl.parameterized`` ``TestNumbersCleaners`` to
``@pytest.mark.parametrize``. ``normalize_numbers`` is pure-python (``re`` +
``num2words``) so all tests run unmarked. ``num2words`` is a hard dependency of the
module, so we do not guard against its absence.

Tests are grouped feature-by-feature (units / math / money / time / ordinals /
misc) and every parametrized case carries an explicit, readable ``id``.
"""

import pytest

from utils.text.numbers import normalize_numbers


# --- units (math) ------------------------------------------------------------------

@pytest.mark.parametrize('text,expected', [
    ('1g',  'one gram'),
    ('2g',  'two grams'),
    ('3m',  'three meters'),
    ('4l',  'four liters'),
    ('5mi', 'five miles'),
    ('6 t', 'six tons'),

    ('7 mm', 'seven milimeters'),
    ('8 kg', 'eight kilograms'),
    ('9 Mo', 'nine megaoctets'),
    ('10 Gb', 'ten gigabits'),

    ('5cm/s',  'five centimeters per second'),
    ('10km/h', 'ten kilometers per hour'),

    # multi-digit numbers exercise the integer-based pluralization
    ('100g', 'one hundred grams'),
], ids = [
    '1g', '2g', '3m', '4l', '5mi', '6t',
    '7mm', '8kg', '9Mo', '10Gb',
    '5cm/s', '10km/h',
    '100g',
])
def test_units(asserts, text, expected):
    asserts.assert_equal(expected, normalize_numbers(text))


# --- math operators ----------------------------------------------------------------

@pytest.mark.parametrize('text,expected', [
    ('-1',  ' minus one'),
    ('+1',  ' plus one'),
    ('1+1', 'one plus one'),
    ('1 + 1', 'one plus one'),
    ('1-1', 'one - one'),
    ('1 - 1', 'one minus one'),
    ('-1 - -1', ' minus one minus minus one'),
    ('-1 * -1', ' minus one times minus one'),
    ('-1.5 / - 2.5', ' minus one punt five divide by minus two punt five'),
], ids = [
    'neg', 'pos', 'add-tight', 'add-spaced', 'sub-tight', 'sub-spaced',
    'neg-sub-neg', 'neg-mul-neg', 'neg-div-neg-decimal',
])
def test_math_operators(asserts, text, expected):
    asserts.assert_equal(expected, normalize_numbers(text))


# --- expand_symbols toggle ---------------------------------------------------------

@pytest.mark.parametrize('text,expected', [
    # units and math symbols are left untouched; only plain numbers expand
    ('2g',  'twog'),
    ('1+1', 'one+one'),
], ids = ['unit-kept', 'operator-kept'])
def test_no_symbol_expansion(asserts, text, expected):
    asserts.assert_equal(expected, normalize_numbers(text, expand_symbols = False))


# --- money -------------------------------------------------------------------------

@pytest.mark.parametrize('text,lang,expected', [
    ('$10', 'en', 'ten dollars'),
    ('$1',  'fr', 'un dollar'),
    ('$1.50', 'en', 'one dollar, fifty cents'),
    ('$0.50', 'en', 'fifty cents'),
    ('$0',  'en', 'zero dollars'),
    ('£5',  'en', 'five pounds'),
], ids = [
    'dollars', 'dollar-fr', 'dollar-cents', 'cents-only', 'zero', 'pounds',
])
def test_money(asserts, text, lang, expected):
    asserts.assert_equal(expected, normalize_numbers(text, lang = lang))


# --- time --------------------------------------------------------------------------

@pytest.mark.parametrize('lang,text,expected', [
    ('en', '1 sec', 'one second'),
    ('en', '10sec', 'ten seconds'),
    ('en', '1min', 'one minute'),
    ('en', '2 min 1sec', 'two minutes and one second'),
    ('en', '1h', 'one hour'),
    ('en', '2 h 2min', 'two hours and two minutes'),
    ('en', '10 h 10 sec', 'ten hours and ten seconds'),
    ('en', '23h 59min 59sec', 'twenty-three hours and fifty-nine minutes and fifty-nine seconds'),

    ('fr', '1 sec', 'une seconde'),
    ('fr', '10sec', 'dix secondes'),
    ('fr', '1min', 'une minute'),
    ('fr', '2 min 1sec', 'deux minutes et une seconde'),
    ('fr', '1h', 'une heure'),
    ('fr', '2 h 2min', 'deux heures et deux minutes'),
    ('fr', '10 h 10 sec', 'dix heures et dix secondes'),
    ('fr', '23h 59min 59sec', 'vingt-trois heures et cinquante-neuf minutes et cinquante-neuf secondes'),
], ids = [
    'en-sec', 'en-secs', 'en-min', 'en-min-sec', 'en-h', 'en-h-min',
    'en-h-sec', 'en-full',
    'fr-sec', 'fr-secs', 'fr-min', 'fr-min-sec', 'fr-h', 'fr-h-min',
    'fr-h-sec', 'fr-full',
])
def test_time(asserts, lang, text, expected):
    asserts.assert_equal(expected, normalize_numbers(text, lang = lang))

def test_clock(asserts):
    asserts.assert_equal(
        'twelve hours and thirty minutes and forty-five seconds',
        normalize_numbers('12:30:45')
    )


# --- ordinals ----------------------------------------------------------------------

@pytest.mark.parametrize('lang,text,expected', [
    ('en', '3rd',  'third'),
    ('en', '2nd',  'second'),
    ('en', '10ème', 'tenth'),

    ('fr', '2nd',  'deuxième'),
    ('fr', '3rd',  'troisième'),
    ('fr', '10ième', 'dixième'),

    ('be', '1er',  'premier'),
    ('be', '3rd',  'troisième'),
    ('be', '70ème', 'septantième'),
    ('be', '91ème', 'nonante et unième'),
], ids = [
    'en-3rd', 'en-2nd', 'en-10', 'fr-2nd', 'fr-3rd', 'fr-10',
    'be-1er', 'be-3rd', 'be-70', 'be-91',
])
def test_ordinal(asserts, lang, text, expected):
    asserts.assert_equal(expected, normalize_numbers(text, lang = lang))


# --- misc --------------------------------------------------------------------------

@pytest.mark.parametrize('text,lang,expected', [
    ('1, 2, 3, 4 and 5 !', 'en', 'one, two, three, four and five !'),
    ('1 000', 'en', 'one thousand'),
    ('1 000 000', 'en', 'one million'),
    ('3,000', 'en', 'three thousand'),                  # comma = thousands separator (en)
    ('1.5', 'en', 'one punt five'),
    ('1,5', 'fr', 'un virgule cinq'),                   # comma = decimal separator (fr)
    ('put during 3-4 min', 'en', 'put during three - four minutes'),
], ids = [
    'enumeration', 'space-thousand', 'space-million', 'comma-thousand-en',
    'decimal-en', 'decimal-fr', 'range-with-unit',
])
def test_others(asserts, text, lang, expected):
    asserts.assert_equal(expected, normalize_numbers(text, lang = lang))
