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

""" Pure-``pytest`` tests for :mod:`utils.text.text_processing`.

This file holds the tests that previously lived (mis-named) in
``test_paragraphs_processing.py`` : they exercise ``split_sentences`` / ``merge_texts``
which are defined in ``text_processing`` (the real ``paragraphs_processing`` module is
now covered by its own file).

Ported from ``absl.parameterized`` to ``@pytest.mark.parametrize``. Notably,
``test_merging_words`` was a **dead test** in the original suite (its body called
``merge_texts`` but asserted nothing) — the missing assertion is restored here.

Everything is pure-python (``re``) so the tests run unmarked. ``test_format_jinja``
needs ``jinja2`` and is skipped if it is missing.
"""

import pytest

from utils.text.text_processing import (
    split_sentences, merge_texts, split_text, split_and_join, get_pairs, bpe, format_text,
)

_words = lambda text: text.split()


# --- `split_sentences` (count-based, faithful port) --------------------------------

@pytest.mark.parametrize('text,n_sentences', [
    ('Hello World !', 1),
    ('Hello World ! This is a test', 2),
    ('Hello World ? This is a test', 2),
    ('Hello World. This is a test', 2),
    ('Hello World... This is a test.', 2),

    ('This is an url : http://example.example.com', 1),
    ('This is an email : example.example@example.com', 1),

    ('1. First item.\n2. Second item.\n3. 3rd item.', 3),
    ('Examples :\n1. First item.\n2. Second item.\n3. 3rd item.', 4),
    ('Examples : \n1. First item.\n2. Second item.\n3. 3rd item.', 4),
    ('Example :\n1. First item\n    1.1 First item A\n    1.2 First item B\n2. Second item', 5),
    ('Items are : 1) First item 2) Second item 3) Third item', 1),
    ('List of items :\n- First item\n- Second item\n- Third item', 4),
    ('Equations :\n- 1 + 1 = 2\n- 1 - 1 = 0\n- -1 * 2 = -2', 4),

    ('Equation : 1.2 + 1.8 = 3.0', 1),
    ('Equation 1 : 1.2 + 1.8 = 3. \nEquation 2 : 1.8 - 1.8 = 0.\nend', 3),
    ('1.2 + 1.3 = 2.5. 1.3 + 1.2 = 2.5. Addition is commutative', 3),

    ('She said "Hello World !"', 1),
    ('E.g., "Hello World !"', 1),
    ('E.g. "Hello World !"', 1),

    ('M.H.C.P. stands for "Mental Health Counsuling Program"', 1),
])
def test_split_sentences(asserts, text, n_sentences):
    sentences = split_sentences(text)
    asserts.assert_equal(n_sentences, len(sentences), msg = 'Result : {}'.format(sentences))


# --- `merge_texts` -----------------------------------------------------------------

@pytest.mark.parametrize('texts,max_length,expected_indices', [
    (['a', 'b', 'c', 'd'],      2, [[0, 1], [2, 3]]),
    (['a', 'b', 'c', 'd'],      3, [[0, 1, 2], [3]]),
    (['ab', 'c', 'def', 'g'],   3, [[0, 1], [2], [3]]),   # char tokenizer (default `list`)
])
def test_merge_texts_char(asserts, texts, max_length, expected_indices):
    """ Default tokenizer is ``list`` -> length is the number of characters. """
    _, _, indices = merge_texts(texts, max_length)
    asserts.assert_equal(expected_indices, indices)

@pytest.mark.parametrize('texts,max_length,expected_indices', [
    (['a', 'b', 'c', 'd'],      2, [[0, 1], [2, 3]]),
    (['a', 'b', 'c', 'd'],      3, [[0, 1, 2], [3]]),
    (['ab', 'c', 'def', 'g'],   3, [[0, 1, 2], [3]]),     # word tokenizer -> each is 1 token
    (['Hello World', '!'],      3, [[0, 1]]),
    (['Hello', 'World', '!', 'This', 'is a test'], 3, [[0, 1, 2], [3], [4]]),
])
def test_merge_texts_words(asserts, texts, max_length, expected_indices):
    """ Restores the assertion the original ``test_merging_words`` was missing. """
    _, _, indices = merge_texts(texts, max_length, tokenizer = _words)
    asserts.assert_equal(expected_indices, indices)


# --- `split_text` ------------------------------------------------------------------

def test_split_text_short_is_untouched(asserts):
    asserts.assert_equal(['Hello world'], split_text('Hello world', max_length = 10, tokenizer = _words))

def test_split_text_chunks_respect_max_length():
    text = 'Hello world. This is a test. Another sentence here.'
    chunks = split_text(text, max_length = 4, tokenizer = _words)

    assert len(chunks) > 1, 'Expected the text to be split, got {}'.format(chunks)
    for chunk in chunks:
        assert len(_words(chunk)) <= 4, 'Chunk too long : {!r}'.format(chunk)


# --- helpers : `split_and_join`, `get_pairs`, `bpe`, `format_text` -----------------

@pytest.mark.parametrize('text,patterns,expected', [
    ('a!b',   ('!', ),      ['a', '!', 'b']),
    ('a.b.c', ('.', ),      ['a', '.', 'b', '.', 'c']),
    ('a!b.c', ('!', '.'),   ['a', '!', 'b', '.', 'c']),
])
def test_split_and_join(asserts, text, patterns, expected):
    asserts.assert_equal(expected, split_and_join(text, * patterns))

@pytest.mark.parametrize('text,n,expected', [
    ('hello', 2, [('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')]),
    ('abcd',  3, [('a', 'b', 'c'), ('b', 'c', 'd')]),
    ('a',     2, []),   # too short for a pair
])
def test_get_pairs(asserts, text, n, expected):
    asserts.assert_equal(expected, get_pairs(text, n))

def test_bpe_no_rank_returns_chars(asserts):
    asserts.assert_equal(('h', 'e', 'l', 'l', 'o'), bpe('hello', {}))

def test_bpe_merges_ranked_pair(asserts):
    asserts.assert_equal(('h', 'e', 'll', 'o'), bpe('hello', {('l', 'l') : 0}))

def test_bpe_single_char(asserts):
    asserts.assert_equal('a', bpe('a', {}))

@pytest.mark.parametrize('fmt,kwargs,expected', [
    ('Hello {name}', {'name' : 'World'}, 'Hello World'),
    ('plain text',   {},                 'plain text'),
    ('no placeholder { here }', {},      'no placeholder { here }'),  # space inside braces -> as-is
])
def test_format_text(asserts, fmt, kwargs, expected):
    asserts.assert_equal(expected, format_text(fmt, ** kwargs))

def test_format_text_jinja(asserts):
    pytest.importorskip('jinja2')
    asserts.assert_equal('World', format_text('{{ name }}', name = 'World'))
