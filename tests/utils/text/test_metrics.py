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

""" Pure-``pytest`` tests for :mod:`utils.text.metrics`.

Ported from the original ``unittest`` ``TestMetrics`` (which only covered ``text_f1``),
and extended to the other distance methods exposed through the ``text_distance``
dispatcher : ``edit_distance`` (weighted Levenstein), ``hamming_distance`` and
``exact_match``.

All metrics here are pure-numpy / pure-python (no keras backend) so every test runs
unmarked. The ``[EM, F1, precision, recall]`` quadruplets returned by ``text_f1`` are
compared through the tolerant ``asserts`` fixture (the recall/precision are floats).
"""

import numpy as np
import pytest

from utils.text.metrics import edit_distance, hamming_distance, exact_match, text_f1


# --- `text_f1` (faithful port of the original test) --------------------------------

@pytest.mark.parametrize('y_true,y_pred,kwargs,expected', [
    ('Hello World !', 'Hello ! World', {},                   [1, 1, 1, 1]),
    ('Hello World !', 'Hello ! World', {'normalize' : False}, [0, 1, 1, 1]),
    ('Hello World !', 'Hello ! world', {'normalize' : False}, [0, 2 / 3, 2 / 3, 2 / 3]),
    ('Hello World !', 'Hello world',   {},                   [1, 1, 1, 1]),
    ([0, 1, 2],       [0, 2, 1],       {},                   [0, 1, 1, 1]),
    ([0, 1, 2],       [0, 2],          {'exclude' : [1]},    [1, 1, 1, 1]),
    ([0, 1, 2],       [0, 2],          {},                   [0, 0.8, 1, 2 / 3]),
])
def test_text_f1(asserts, y_true, y_pred, kwargs, expected):
    asserts.assert_equal(expected, text_f1(y_true, y_pred, ** kwargs))

def test_text_f1_as_matrix(asserts):
    """ Two lists + ``as_matrix`` -> a ``(n_true, n_pred, 4)`` cross-product. """
    result = text_f1(
        ['hello world', 'foo'], ['hello world', 'bar'], as_matrix = True
    )
    expected = np.array([
        [[1, 1, 1, 1], [0, 0, 0, 0]],   # 'hello world' vs each prediction
        [[0, 0, 0, 0], [0, 0, 0, 0]],   # 'foo' vs each prediction
    ], dtype = 'float64')
    asserts.assert_equal((2, 2, 4), tuple(result.shape))
    asserts.assert_equal(expected, result)


# --- `exact_match` -----------------------------------------------------------------

@pytest.mark.parametrize('y_true,y_pred,expected', [
    ('hello world', 'hello world', 1),
    ('hello world', 'hello',       0),
    ('',            '',            1),
])
def test_exact_match(asserts, y_true, y_pred, expected):
    asserts.assert_equal(expected, exact_match(y_true, y_pred))


# --- `edit_distance` (weighted Levenstein) -----------------------------------------

@pytest.mark.parametrize('hyp,truth,expected', [
    ('abc',    'abc',     0),   # identical
    ('abc',    'abd',     1),   # one substitution
    ('abc',    'ab',      1),   # one deletion
    ('ab',     'abc',     1),   # one insertion
    ('kitten', 'sitting', 3),   # textbook Levenstein
    ('',       'abc',     3),   # full insertion
])
def test_edit_distance(asserts, hyp, truth, expected):
    asserts.assert_equal(expected, edit_distance(hyp, truth, normalize = False))

def test_edit_distance_normalized(asserts):
    # normalized on the truth length by default
    asserts.assert_equal(1 / 3, edit_distance('abc', 'abd', normalize = True))

def test_edit_distance_return_matrix(asserts):
    distance, matrix = edit_distance('ab', 'abc', normalize = False, return_matrix = True)
    asserts.assert_equal(1, distance)
    asserts.assert_equal((len('ab') + 1, len('abc') + 1), tuple(matrix.shape))

def test_edit_distance_custom_replacement_cost(asserts):
    # a -> b costs 0.5 instead of the default 1
    asserts.assert_equal(
        0.5,
        edit_distance(
            'a', 'b', normalize = False, replacement_cost = {'a' : {'b' : 0.5}}
        )
    )


# --- `hamming_distance` ------------------------------------------------------------

@pytest.mark.parametrize('hyp,truth,expected', [
    ('abc', 'abc', 0),
    ('abc', 'abd', 1),
    ('abc', 'aXX', 2),
    ('abc', 'ab', -1),     # different lengths -> -1
])
def test_hamming_distance(asserts, hyp, truth, expected):
    asserts.assert_equal(expected, hamming_distance(hyp, truth, normalize = False))

def test_hamming_distance_normalized(asserts):
    asserts.assert_equal(1 / 3, hamming_distance('abc', 'abd', normalize = True))


# --- `mode` forcing (similarity <-> distance negation) -----------------------------

def test_mode_negates_for_distance_metric(asserts):
    """ ``hamming`` is a distance ; forcing ``similarity`` negates it, ``distance`` is a no-op. """
    base = hamming_distance('abc', 'abd', normalize = False)
    asserts.assert_equal(- base, hamming_distance('abc', 'abd', normalize = False, mode = 'similarity'))
    asserts.assert_equal(base,   hamming_distance('abc', 'abd', normalize = False, mode = 'distance'))
