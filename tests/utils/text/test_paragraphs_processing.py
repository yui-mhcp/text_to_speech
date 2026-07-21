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

""" Pure-``pytest`` tests for :mod:`utils.text.paragraphs_processing`.

NEW coverage : the original ``test_paragraphs_processing.py`` actually tested
``text_processing`` helpers (now in ``test_text_processing.py``). This file targets the
real ``paragraphs_processing`` module — the dict-based ``paragraph_to_text`` /
``group_paragraphs`` / ``merge_paragraphs`` / ``chunks_from_paragraphs`` /
``process_paragraphs`` helpers — which had no test at all.

Everything is pure-python (dicts + ``re``) so the tests run unmarked.
"""

import pytest

from utils.text.paragraphs_processing import (
    paragraph_to_text, group_paragraphs, merge_paragraphs,
    chunks_from_paragraphs, process_paragraphs,
)

_words = lambda text: text.split()


# --- `paragraph_to_text` -----------------------------------------------------------

def test_paragraph_to_text_from_str(asserts):
    asserts.assert_equal('hello', paragraph_to_text('hello'))

def test_paragraph_to_text_from_text_entry(asserts):
    asserts.assert_equal('foo', paragraph_to_text({'text' : 'foo'}))

def test_paragraph_to_text_from_content(asserts):
    para = {'content' : [{'text' : 'a'}, {'text' : 'b'}]}
    asserts.assert_equal('a\n\nb', paragraph_to_text(para))

def test_paragraph_to_text_with_format(asserts):
    asserts.assert_equal(
        'Hello World', paragraph_to_text({'text' : 'World'}, format = 'Hello {text}')
    )

def test_paragraph_to_text_list_type(asserts):
    asserts.assert_equal(
        '\n- x\n-y', paragraph_to_text({'type' : 'list', 'items' : ['x', 'y']})
    )

@pytest.mark.parametrize('ptype', ['image', 'audio', 'video'])
def test_paragraph_to_text_multimodal_is_none(ptype):
    assert paragraph_to_text({'type' : ptype}) is None

def test_paragraph_to_text_unknown_type_raises():
    with pytest.raises(ValueError):
        paragraph_to_text({'type' : 'does_not_exist'})


# --- `group_paragraphs` ------------------------------------------------------------

def test_group_paragraphs_single_key(asserts):
    paragraphs = [
        {'section' : 'A', 'text' : '1'},
        {'section' : 'B', 'text' : '2'},
        {'section' : 'A', 'text' : '3'},
    ]
    groups = group_paragraphs(paragraphs, 'section')

    asserts.assert_equal([('A', ), ('B', )], list(groups.keys()))   # insertion-ordered
    asserts.assert_equal(2, len(groups[('A', )]))
    asserts.assert_equal(1, len(groups[('B', )]))

def test_group_paragraphs_multi_key(asserts):
    paragraphs = [
        {'doc' : 'x', 'page' : 1, 'text' : 'a'},
        {'doc' : 'x', 'page' : 1, 'text' : 'b'},
        {'doc' : 'x', 'page' : 2, 'text' : 'c'},
    ]
    groups = group_paragraphs(paragraphs, ['doc', 'page'])

    asserts.assert_equal(2, len(groups))
    asserts.assert_equal(2, len(groups[('x', 1)]))


# --- `merge_paragraphs` ------------------------------------------------------------

def test_merge_paragraphs_hoists_common_keys(asserts):
    paragraphs = [
        {'type' : 'text', 'text' : 'a', 'page' : 1},
        {'type' : 'text', 'text' : 'b', 'page' : 1},
    ]
    merged = merge_paragraphs(paragraphs)

    # the shared `page` is hoisted to the merged paragraph and popped from the children
    asserts.assert_equal(1, merged['page'])
    asserts.assert_equal(2, len(merged['content']))
    assert 'page' not in merged['content'][0]
    # `text` is never hoisted (kept per-child)
    asserts.assert_equal('a', merged['content'][0]['text'])

def test_merge_paragraphs_keeps_divergent_keys_in_children(asserts):
    paragraphs = [
        {'type' : 'text', 'text' : 'a', 'page' : 1},
        {'type' : 'text', 'text' : 'b', 'page' : 2},
    ]
    merged = merge_paragraphs(paragraphs)

    assert 'page' not in merged          # diverging -> not hoisted
    asserts.assert_equal(1, merged['content'][0]['page'])
    asserts.assert_equal(2, merged['content'][1]['page'])


# --- `chunks_from_paragraphs` ------------------------------------------------------

def test_chunks_from_paragraphs_no_max_length_is_identity(asserts):
    paragraphs = [{'text' : 'a'}, {'text' : 'b'}]
    asserts.assert_equal(paragraphs, chunks_from_paragraphs(paragraphs))

def test_chunks_from_paragraphs_short_texts_preserved(asserts):
    paragraphs = [{'text' : 'Hello world.'}, {'text' : 'Foo bar.'}]
    chunks = chunks_from_paragraphs(paragraphs, max_length = 100, tokenizer = _words)
    asserts.assert_equal(['Hello world.', 'Foo bar.'], [c['text'] for c in chunks])


# --- `process_paragraphs` ----------------------------------------------------------

def test_process_paragraphs_skips_header(asserts):
    paragraphs = [{'text' : 'title', 'is_header' : True}, {'text' : 'body'}]
    result = process_paragraphs(paragraphs)
    asserts.assert_equal(['body'], [p['text'] for p in result])

def test_process_paragraphs_skips_page_number(asserts):
    paragraphs = [{'text' : '12', 'is_page_number' : True}, {'text' : 'body'}]
    result = process_paragraphs(paragraphs)
    asserts.assert_equal(['body'], [p['text'] for p in result])

def test_process_paragraphs_strips_urls(asserts):
    paragraphs = [{'text' : 'see http://example.com'}]
    result = process_paragraphs(paragraphs, skip_urls = True)
    asserts.assert_equal('see', result[0]['text'])
