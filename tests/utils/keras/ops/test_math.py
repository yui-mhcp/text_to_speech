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

""" Pure-``pytest`` tests for :mod:`utils.keras.ops.math`.

Focuses on the custom ``segment_*`` reductions (which have a bespoke numpy
implementation, so the reference is built independently in numpy -> the
``numpy`` backend stays keras-free) and the custom ``top_k``.

Graph / XLA compatibility is checked in dedicated ``tensorflow``-marked tests.
"""

import numpy as np
import pytest

from utils.keras import ops
from ._base import check_ops, prepare

_SEGMENT_OPS = ['sum', 'min', 'max', 'mean', 'argsort']


# --- fixtures / references ---------------------------------------------------------

@pytest.fixture(scope = 'module')
def segment_data():
    """ Deterministic ``(64, 64)`` data split into 8 contiguous segments of 8 rows. """
    rng = np.random.default_rng(0)
    data        = rng.uniform(0., 10., size = (64, 64)).astype('float32')
    segment_ids = np.repeat(np.arange(8), 8).astype('int32')
    return data, segment_ids, 8

def _segment_reference(name, data, segment_ids, num_segments):
    """ Per-segment numpy reduction (keras-free reference). """
    np_fn = getattr(np, name)
    parts = [np_fn(data[segment_ids == i], axis = 0) for i in range(num_segments)]
    if name == 'argsort':
        return np.concatenate(parts, axis = 0).astype('int32')
    return np.array(parts)


# --- segment_* ---------------------------------------------------------------------

@pytest.mark.parametrize('name', _SEGMENT_OPS)
@pytest.mark.parametrize('axis', [0, 1])
def test_segment(name, axis, backend, segment_data, asserts):
    data, segment_ids, num_segments = segment_data

    ref = _segment_reference(name, data if axis == 0 else data.T, segment_ids, num_segments)
    if axis == 1: ref = ref.T

    check_ops(
        getattr(ops, 'segment_' + name),
        prepare(backend, data), segment_ids, num_segments, axis = axis,
        ref = ref, backend = backend, asserts = asserts
    )

@pytest.mark.tensorflow
@pytest.mark.parametrize('name', _SEGMENT_OPS)
def test_segment_graph(name, segment_data, asserts):
    data, segment_ids, num_segments = segment_data
    ref = _segment_reference(name, data, segment_ids, num_segments)
    asserts.assert_graph_compatible(
        getattr(ops, 'segment_' + name), data, segment_ids, num_segments, axis = 0, target = ref
    )


# --- top_k -------------------------------------------------------------------------

# hand-written expectations (descending order, like `keras.ops.top_k`)
_TOP_K_DATA = np.array([[5., 1., 4., 2., 3.], [9., 8., 0., 7., 6.]], dtype = 'float32')
_TOP_K_REF  = {
    1 : ([[5.], [9.]],             [[0], [0]]),
    3 : ([[5., 4., 3.], [9., 8., 7.]], [[0, 2, 4], [0, 1, 3]]),
}

@pytest.mark.parametrize('k', [1, 3])
def test_top_k(backend, k, asserts):
    values_ref, indices_ref = _TOP_K_REF[k]
    check_ops(
        ops.top_k, prepare(backend, _TOP_K_DATA), k,
        ref = (np.array(values_ref, dtype = 'float32'), np.array(indices_ref)),
        backend = backend, asserts = asserts
    )

@pytest.mark.tensorflow
@pytest.mark.parametrize('k', [1, 3])
def test_top_k_graph(k, asserts):
    asserts.assert_graph_compatible(
        ops.top_k, _TOP_K_DATA, k, target = ops.top_k(_TOP_K_DATA, k)
    )


# --- segment_weighted_mean ---------------------------------------------------------
# Intended semantics : per-segment mean weighted by `weights`. The generic
# `_segment_op_wrapper` does not thread a `weights` argument through (and the op body
# references a bare `einsum`), so this is expected to surface those issues (per design).

def test_segment_weighted_mean(backend, asserts):
    data        = np.array([1., 2., 3., 4.], dtype = 'float32')
    segment_ids = np.array([0, 0, 1, 1], dtype = 'int32')
    weights     = np.array([1., 3., 1., 1.], dtype = 'float32')

    # seg0 = (1*1 + 2*3) / (1 + 3) = 1.75 ; seg1 = (3*1 + 4*1) / 2 = 3.5
    ref = np.array([1.75, 3.5], dtype = 'float32')

    check_ops(
        ops.segment_weighted_mean,
        prepare(backend, data), segment_ids, prepare(backend, weights), 2,
        ref = ref, backend = backend, asserts = asserts
    )
