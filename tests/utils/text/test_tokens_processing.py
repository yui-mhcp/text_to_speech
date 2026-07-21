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

""" Pure-``pytest`` tests for :mod:`utils.text.tokens_processing`.

Ported from the original ``unittest`` ``TestTokensProcessing`` :
  - ``test_text_filtering`` -> split into focused ``filter_texts`` cases sharing a
    seeded ``(texts, lengths)`` fixture ;
  - ``test_logits_filtering`` (``self.subTest`` loop) -> ``@pytest.mark.parametrize``
    over the slice index.

These functions go through ``utils.keras.ops``, which dispatches ``np.ndarray`` inputs
to the numpy implementation, so the tests run keras-free and are left unmarked.
"""

import numpy as np
import pytest

from utils.text.tokens_processing import (
    filter_texts, mask_slice_tokens, mask_batch_tokens, process_model_output,
)


# --- `filter_texts` ----------------------------------------------------------------

@pytest.fixture
def filter_data():
    """ 10 texts of 10 tokens each, right-padded with ``-1`` to per-row ``lengths``. """
    texts   = np.tile(np.arange(10)[np.newaxis], [10, 1]).astype(np.int32)
    lengths = np.array([3, 5, 7, 1, 2, 4, 8, 9, 6, 10], dtype = np.int32)
    for i, l in enumerate(lengths):
        texts[i, l:] = -1
    return texts, lengths

def test_filter_texts_noop(asserts, filter_data):
    texts, lengths = filter_data
    asserts.assert_equal((texts, lengths), filter_texts(texts, lengths))

def test_filter_texts_max_length(asserts, filter_data):
    texts, lengths = filter_data
    asserts.assert_equal(
        (texts[lengths <= 5, :5], lengths[lengths <= 5]),
        filter_texts(texts, lengths, max_text_length = 5)
    )

def test_filter_texts_min_and_max_length(asserts, filter_data):
    texts, lengths = filter_data
    mask = np.logical_and(lengths <= 5, lengths >= 2)
    asserts.assert_equal(
        (texts[mask, :5], lengths[mask]),
        filter_texts(texts, lengths, min_text_length = 2, max_text_length = 5)
    )

def test_filter_texts_max_total_length(asserts, filter_data):
    texts, lengths = filter_data
    asserts.assert_equal(
        (texts[[0], :3], lengths[[0]]),
        filter_texts(texts, lengths, min_text_length = 2, max_total_length = 5)
    )
    asserts.assert_equal(
        (texts[[0, 4], :3], lengths[[0, 4]]),
        filter_texts(texts, lengths, min_text_length = 2, max_total_length = 5, sort_by_length = True)
    )
    asserts.assert_equal(
        (texts[[0, 1], :5], lengths[[0, 1]]),
        filter_texts(texts, lengths, max_total_length = 8)
    )
    asserts.assert_equal(
        (texts[[0, 3, 4], :3], lengths[[0, 3, 4]]),
        filter_texts(texts, lengths, max_total_length = 8, sort_by_length = True)
    )

def test_filter_texts_required_idx(asserts, filter_data):
    texts, lengths = filter_data
    asserts.assert_equal(
        (texts[[0, 1], :5], lengths[[0, 1]]),
        filter_texts(texts, lengths, max_total_length = 8, required_idx = 1)
    )
    asserts.assert_equal(
        (texts[[1, 3, 4], :5], lengths[[1, 3, 4]]),
        filter_texts(texts, lengths, max_total_length = 8, sort_by_length = True, required_idx = 1)
    )
    # `required_idx` longer than `max_text_length` -> everything is dropped
    asserts.assert_equal(
        (texts[[]], lengths[[]]),
        filter_texts(texts, lengths, max_text_length = 4, required_idx = 1)
    )


# --- logits masking ----------------------------------------------------------------

@pytest.mark.parametrize('index', [0, 5, 10, 15, 20])
def test_mask_slice_tokens(asserts, index):
    rng    = np.random.default_rng(0)
    logits = rng.normal(size = (5, 25)).astype(np.float32)

    expected = logits.copy()
    expected[:, index :] = - np.inf
    asserts.assert_equal(expected, mask_slice_tokens(logits, index, remove_after = True))

    expected = logits.copy()
    expected[:, : index] = - np.inf
    asserts.assert_equal(expected, mask_slice_tokens(logits, index, remove_after = False))

@pytest.mark.parametrize('index', [0, 5, 10, 15, 20])
def test_mask_batch_tokens(asserts, index):
    rng    = np.random.default_rng(1)
    logits = rng.normal(size = (5, 25)).astype(np.float32)

    indexes  = [index, 24, 1]
    expected = logits.copy()
    expected[:, indexes] = - np.inf
    asserts.assert_equal(expected, mask_batch_tokens(logits, indexes))


# --- `process_model_output` --------------------------------------------------------

def test_process_model_output_slices_scalar_length(asserts):
    """ Scalar ``lengths`` -> ``output[offset:lengths]``. """
    output = np.arange(10, dtype = np.int32)
    asserts.assert_equal(
        np.array([2, 3, 4], dtype = np.int32),
        process_model_output(output, offset = 2, lengths = np.array(5, dtype = np.int32))
    )
