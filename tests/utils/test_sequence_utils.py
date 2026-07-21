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

""" Pure-``pytest`` tests for :mod:`utils.sequence_utils`.

Written as plain module-level functions + ``@pytest.mark.parametrize`` (no
``unittest.TestCase`` / ``absl.parameterized``). Tolerant numpy comparisons go
through the ``asserts`` fixture (``tests.asserts``).

Backend handling (``backend`` fixture, mirrors :mod:`tests.utils.test_distances`) :
  - ``numpy``  : pure ``np.ndarray`` inputs. ``utils.sequence_utils`` only uses
    ``utils.keras.ops``, which dispatches ``np.ndarray`` to numpy, so these cases
    run **without keras** and are therefore left unmarked.
  - ``tensor`` : inputs converted to ``keras`` tensors -> marked ``keras`` (lazy
    import so the numpy cases stay keras-free).

Two behaviours of ``pad_batch`` drive the type assertions below :
  - the **padding path** (heterogeneous shapes) always materialises a
    ``np.ndarray`` — even for tensor inputs — because it falls back to
    ``ops.convert_to_numpy`` + ``np.full``. Type is only preserved on the
    *non-padding* paths : single element (``batch[0][None]``), identical shapes,
    and ``pad_value=None`` (both go through ``ops.stack``).
  - ``pad_batch`` is therefore **not** graph-compatible by design and gets no
    ``tensorflow`` test ; ``pad_to_multiple`` (pure ``ops.pad`` / ``ops.shape``)
    is checked behind the ``tensorflow`` marker.
"""

import pytest
import numpy as np

from utils.sequence_utils import pad_batch, pad_to_multiple


# --- backend helpers ---------------------------------------------------------------

def _to_tensor(x, dtype = None):
    """ Lazily import keras so the ``numpy`` cases stay keras-free. """
    import keras.ops as K
    return K.convert_to_tensor(x, dtype = dtype)

_BACKENDS = [
    pytest.param('numpy',  id = 'numpy'),
    pytest.param('tensor', id = 'tensor', marks = pytest.mark.keras),
]

@pytest.fixture(params = _BACKENDS)
def backend(request):
    return request.param

def _convert(backend, x):
    """ Convert a (numpy) reference array to the input for a given backend. """
    return x if backend == 'numpy' else _to_tensor(x)

def _assert_type(backend, x, asserts):
    """ ``utils.keras.ops`` central property : np in -> np out, tensor in -> tensor out.

        Only valid on the *type-preserving* paths (single / stack / no-op) — the
        padding path of ``pad_batch`` always returns a ``np.ndarray`` (see module docstring).
    """
    if backend == 'numpy':  asserts.assert_array(x)
    else:                   asserts.assert_tensor(x)


# --- `pad_batch` : padding path (heterogeneous shapes) -----------------------------

_PAD_SHAPES = [
    pytest.param((4, 5),         [(1, ), (2, ), (5, ), (4, )],                          id = 'vectors'),
    pytest.param((4, 8, 8),      [(2, 3), (3, 4), (5, 8), (8, 3)],                      id = 'matrix'),
    pytest.param((4, 32, 32, 3), [(16, 16, 3), (16, 16, 3), (32, 32, 3), (32, 32, 3)], id = 'images'),
]

@pytest.mark.parametrize('target_shape, shapes', _PAD_SHAPES)
def test_pad_batch_pads_to_max_shape(backend, asserts, target_shape, shapes):
    inputs  = [_convert(backend, np.ones(s)) for s in shapes]
    batch   = pad_batch(inputs)

    # the padding path always materialises a numpy array, whatever the input type
    asserts.assert_array(batch)
    asserts.assert_equal(target_shape, tuple(batch.shape))
    for s, inp, b in zip(shapes, inputs, batch):
        asserts.assert_equal(inp, b[tuple(slice(0, dim) for dim in s)])

def test_pad_batch_simple_values(asserts):
    """ Faithful port of the original 1-D value checks. """
    asserts.assert_equal(np.array([[1, 2, 0], [1, 2, 3]]), pad_batch([[1, 2], [1, 2, 3]]))
    asserts.assert_equal(np.array([[1, 2, 3], [1, 2, 0]]), pad_batch([[1, 2, 3], [1, 2]]))
    asserts.assert_equal(
        np.array([[1, 2, -1], [1, 2, 3]]), pad_batch([[1, 2], [1, 2, 3]], pad_value = -1)
    )
    asserts.assert_equal(
        np.array([[0, 1, 2], [1, 2, 3]]), pad_batch([[1, 2], [1, 2, 3]], pad_mode = 'before')
    )

def test_pad_batch_before_placement(asserts):
    """ In ``before`` mode the real data sits in the bottom-right corner. """
    a = np.arange(1, 7).reshape(2, 3)    # (2, 3)
    b = np.arange(1, 13).reshape(3, 4)   # (3, 4) == max shape
    batch = pad_batch([a, b], pad_mode = 'before')

    asserts.assert_equal((2, 3, 4), tuple(batch.shape))
    # `a` is padded up to (3, 4) -> occupies the last 2 rows / last 3 cols
    asserts.assert_equal(a, batch[0, -2:, -3:])
    asserts.assert_equal(b, batch[1])
    # the leading (padded) row and column of `a` are zero-filled (batch is int64)
    asserts.assert_equal(np.zeros(4, dtype = batch.dtype), batch[0, 0, :])
    asserts.assert_equal(np.zeros(3, dtype = batch.dtype), batch[0, :, 0])


# --- `pad_batch` : non-padding / edge-case paths -----------------------------------

def test_pad_batch_empty():
    assert len(pad_batch([])) == 0

def test_pad_batch_single_adds_batch_dim(backend, asserts):
    inp   = _convert(backend, np.ones((3, 4)))
    batch = pad_batch([inp])

    asserts.assert_equal((1, 3, 4), tuple(batch.shape))
    asserts.assert_equal(inp, batch[0])
    _assert_type(backend, batch, asserts)   # single element -> type preserved

def test_pad_batch_same_shape_stacks(backend, asserts):
    inputs = [_convert(backend, np.ones((2, 3))) for _ in range(4)]
    batch  = pad_batch(inputs)

    asserts.assert_equal((4, 2, 3), tuple(batch.shape))
    _assert_type(backend, batch, asserts)   # identical shapes -> `ops.stack`, type preserved

def test_pad_batch_pad_value_none_stacks(backend, asserts):
    """ ``pad_value=None`` forces a plain ``stack`` (caller guarantees same shapes). """
    inputs = [_convert(backend, np.ones((2, 3))) for _ in range(3)]
    batch  = pad_batch(inputs, pad_value = None)

    asserts.assert_equal((3, 2, 3), tuple(batch.shape))
    _assert_type(backend, batch, asserts)

def test_pad_batch_scalars(asserts):
    result = pad_batch([1, 2, 3])
    asserts.assert_array(result)
    asserts.assert_equal(np.array([1, 2, 3]), result)

@pytest.mark.parametrize('inputs', [
    pytest.param([[1, 2], [1, 2, 3]],            id = 'nested_lists'),
    pytest.param([np.ones((2, )), np.ones((3, ))], id = 'arrays'),
])
def test_pad_batch_dtype(asserts, inputs):
    batch = pad_batch(inputs, dtype = 'float32')
    assert batch.dtype == np.float32, 'Expected float32 output, got {}'.format(batch.dtype)


# --- `pad_batch` : error paths -----------------------------------------------------

def test_pad_batch_invalid_pad_mode():
    with pytest.raises(AssertionError):
        pad_batch([[1, 2], [1, 2, 3]], pad_mode = 'middle')

def test_pad_batch_different_ranks():
    with pytest.raises(AssertionError):
        pad_batch([np.ones((2, 2)), np.ones((3, ))])


# --- `pad_to_multiple` -------------------------------------------------------------

def test_pad_to_multiple_noop(backend, asserts):
    """ An already-multiple axis is returned unchanged (same object). """
    data   = _convert(backend, np.ones((8, ), dtype = 'float32'))
    result = pad_to_multiple(data, 4)

    assert result is data, 'Already-multiple input must be returned unchanged'
    _assert_type(backend, result, asserts)

@pytest.mark.parametrize('pad_mode, lead, trail', [
    pytest.param('after',  0, 3, id = 'after'),
    pytest.param('before', 3, 0, id = 'before'),
    pytest.param('even',   1, 2, id = 'even'),
])
def test_pad_to_multiple_1d(backend, asserts, pad_mode, lead, trail):
    data   = _convert(backend, np.arange(1, 6, dtype = 'float32'))   # (5,) -> next mult. of 4 is 8 (pad 3)
    result = pad_to_multiple(data, 4, pad_mode = pad_mode)

    asserts.assert_equal((8, ), tuple(result.shape))
    _assert_type(backend, result, asserts)

    expected = np.concatenate(
        [np.zeros(lead), np.arange(1, 6), np.zeros(trail)]
    ).astype('float32')
    asserts.assert_equal(expected, result)

@pytest.mark.parametrize('multiple', [
    pytest.param(4,      id = 'int_broadcast'),
    pytest.param([4, 4], id = 'list'),
])
def test_pad_to_multiple_multi_axis(backend, asserts, multiple):
    data   = _convert(backend, np.ones((5, 6), dtype = 'float32'))   # -> (8, 8)
    result = pad_to_multiple(data, multiple, axis = [0, 1])

    asserts.assert_equal((8, 8), tuple(result.shape))
    # original content preserved in the top-left corner (`after` mode)
    asserts.assert_equal(np.ones((5, 6), dtype = 'float32'), result[:5, :6])
    _assert_type(backend, result, asserts)

def test_pad_to_multiple_negative_axis(backend, asserts):
    data   = _convert(backend, np.ones((5, 6), dtype = 'float32'))
    result = pad_to_multiple(data, 4, axis = -2)   # -2 -> axis 0 only

    asserts.assert_equal((8, 6), tuple(result.shape))
    asserts.assert_equal(np.ones((5, 6), dtype = 'float32'), result[:5])
    _assert_type(backend, result, asserts)


# --- graph / XLA compatibility (requires tensorflow) -------------------------------

# NOTE: `pad_to_multiple` needs a statically-known rank (negative-axis normalisation,
# `paddings` sized by rank, `shape[ax]` indexing). All graph cases here therefore share
# the *same* input shape : `get_graph_function` is `@cache`d per `fn`, so every case
# reuses one `tf.function(reduce_retracing=True)` ; mixing different ranks/shapes would
# make TF relax the signature to an unknown rank and raise at trace time. The rank-1
# no-op case is intentionally not graph-tested for this reason (and is trivial anyway).
@pytest.mark.tensorflow
@pytest.mark.parametrize('pad_mode', ['after', 'before', 'even'])
def test_pad_to_multiple_graph_compatible(asserts, pad_mode):
    data = np.ones((5, 6), dtype = 'float32')
    asserts.assert_graph_compatible(pad_to_multiple, data, 4, pad_mode = pad_mode)
