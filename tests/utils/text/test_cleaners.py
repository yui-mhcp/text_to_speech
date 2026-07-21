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

""" Pure-``pytest`` tests for :mod:`utils.text.cleaners`.

Ported from the original ``unittest`` ``TestCleaners`` (``CustomTestCase``) to plain
module-level functions + ``@pytest.mark.parametrize`` (mirrors ``test_distances``).

``utils.text.cleaners`` is pure-python (``re`` / ``unicodedata`` / ``unidecode``) — no
deep-learning backend — so every test here runs keras-free and is left unmarked.
The string comparisons go through the tolerant ``asserts`` fixture (``tests.asserts``)
for consistency with the rest of the suite, even though they reduce to ``==``.
"""

import pytest

from utils.text.cleaners import (
    strip, lstrip, rstrip, collapse_whitespace, collapse_repetitions,
    detach_punctuation, remove_punctuation, attach_punctuation,
    replace_words, replace_patterns, remove_tokens, remove_markdown,
    expand_abreviations, expand_special_symbols,
    remove_urls, remove_files,
    remove_accents, convert_to_ascii, fr_convert_to_ascii, convert_to_alnum,
    basic_cleaners, get_cleaners_fn,
)


# --- stripping / whitespace --------------------------------------------------------

def test_strip(asserts):
    asserts.assert_equal('Hello  World !', strip(' Hello  World !  '))
    asserts.assert_equal('Hello  World ! ', lstrip(' Hello  World ! '))
    asserts.assert_equal(' Hello  World !', rstrip(' Hello  World ! '))

@pytest.mark.parametrize('lstrip_,rstrip_,expected', [
    (True,  True,  'Hello'),
    (True,  False, 'Hello  '),
    (False, True,  '  Hello'),
    (False, False, '  Hello  '),
])
def test_strip_flags(asserts, lstrip_, rstrip_, expected):
    asserts.assert_equal(expected, strip('  Hello  ', lstrip = lstrip_, rstrip = rstrip_))

def test_collapse_whitespace(asserts):
    asserts.assert_equal(' Hello World !', collapse_whitespace(' Hello  World   !'))

@pytest.mark.parametrize('text,max_rep,expected', [
    ('aaabbb', 2, 'aabb'),
    ('aaabbb', 1, 'ab'),
    ('abc',    1, 'abc'),
    ('aaa',    5, 'aaa'),
    ('',       2, ''),
])
def test_collapse_repetitions(asserts, text, max_rep, expected):
    asserts.assert_equal(expected, collapse_repetitions(text, max_rep))


# --- punctuation -------------------------------------------------------------------

def test_detach_punctuation(asserts):
    asserts.assert_equal(
        'Bonjour ,  comment ça va ?', detach_punctuation('Bonjour, comment ça va?')
    )

def test_remove_punctuation(asserts):
    asserts.assert_equal(
        'Bonjour comment ça va', remove_punctuation('Bonjour, comment ça va?')
    )

@pytest.mark.parametrize('text,expected', [
    ('( a )',            '(a)'),
    ('Hello , World .',  'Hello, World.'),
    ('[ x ]',            '[x]'),
])
def test_attach_punctuation(asserts, text, expected):
    asserts.assert_equal(expected, attach_punctuation(text))


# --- replacements ------------------------------------------------------------------

def test_replace_words(asserts):
    asserts.assert_equal('Ceci est un test', replace_words('Ceci es un test', {'es' : 'est'}))
    # 'est' is not a standalone word in 'Ceci es un test' -> no change
    asserts.assert_equal('Ceci es un test', replace_words('Ceci es un test', {'est' : ''}))
    # case-insensitive by default : both 'C' and 'c' match
    asserts.assert_equal("'est un  test", replace_words("C'est un c test", {'c' : ''}))

def test_replace_words_custom_pattern(asserts):
    # the `(?!')` lookahead protects "C'est" (the apostrophe right after the word)
    asserts.assert_equal(
        "C'est un  test",
        replace_words("C'est un C test", {'c' : ''}, pattern_format = r"\b({})\b(?!')")
    )

def test_replace_patterns(asserts):
    asserts.assert_equal(
        'X-Y-Z', replace_patterns('X Y Z', {r'\s+' : '-'})
    )

def test_remove_tokens(asserts):
    asserts.assert_equal(' World !', remove_tokens('Hello World !', ['hello']))
    asserts.assert_equal('Hello  World !', remove_tokens('Hello the World !', ['this', 'the']))
    # no tokens -> untouched
    asserts.assert_equal('Hello World !', remove_tokens('Hello World !', []))

def test_expand_abreviations(asserts):
    asserts.assert_equal('mister test', expand_abreviations('Mr test', lang = 'en'))
    asserts.assert_equal('mister test', expand_abreviations('Mr. test', lang = 'en'))

def test_expand_special_symbols(asserts):
    asserts.assert_equal('1  equal  1', expand_special_symbols('1 = 1', lang = 'en'))
    asserts.assert_equal('a  percent  b', expand_special_symbols('a % b', lang = 'en'))

def test_remove_markdown(asserts):
    asserts.assert_equal('bold', remove_markdown('**bold**'))


# --- url / file stripping ----------------------------------------------------------

def test_remove_urls(asserts):
    asserts.assert_equal('see  now', remove_urls('see http://example.com now'))

def test_remove_files(asserts):
    asserts.assert_equal('a  b', remove_files('a doc.pdf b'))


# --- ascii / accents ---------------------------------------------------------------

@pytest.mark.parametrize('text,expected', [
    ('café',   'cafe'),
    ('à',      'a'),
    ('Hello',  'Hello'),
])
def test_remove_accents(asserts, text, expected):
    asserts.assert_equal(expected, remove_accents(text))

def test_convert_to_ascii(asserts):
    asserts.assert_equal('cafe', convert_to_ascii('café'))

def test_fr_convert_to_ascii_keeps_french_accents(asserts):
    # 'ç' is in the kept set, 'à' is not -> transliterated
    asserts.assert_equal('ça', fr_convert_to_ascii('çà'))

def test_convert_to_alnum(asserts):
    asserts.assert_equal('a.b c', convert_to_alnum('a.b@c'))


# --- pipelines ---------------------------------------------------------------------

def test_basic_cleaners(asserts):
    asserts.assert_equal(' hello world ', basic_cleaners(' Hello  WORLD '))


# --- `get_cleaners_fn` -------------------------------------------------------------

def test_get_cleaners_fn_by_name(asserts):
    fns = get_cleaners_fn('basic_cleaners')
    asserts.assert_equal(1, len(fns))
    asserts.assert_equal(' hello world ', fns[0](' Hello  WORLD '))

def test_get_cleaners_fn_list(asserts):
    fns = get_cleaners_fn(['lowercase', 'collapse_whitespace'])
    asserts.assert_equal(2, len(fns))

def test_get_cleaners_fn_with_kwargs(asserts):
    # tuple form `(name, kwargs)` -> a `functools.partial` binding the kwargs
    (fn, ) = get_cleaners_fn([('collapse_repetitions', {'max_repetition' : 2})])
    asserts.assert_equal('aabb', fn('aaabbb'))

def test_get_cleaners_fn_unknown_raises():
    with pytest.raises(ValueError):
        get_cleaners_fn('does_not_exist')

def test_get_cleaners_fn_not_callable_raises():
    # `_abreviations` is a module-level dict -> resolvable but not callable
    with pytest.raises(ValueError):
        get_cleaners_fn('_abreviations')
