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

""" Pure-``pytest`` tests for :mod:`utils.keras.ops.numpy`.

Covers the most-used unary / binary elementwise ops, the indexing helpers
(``take`` / ``take_along_axis``) and the custom ``unique`` / ``bincount`` /
``isin``.

References are built in numpy, so the ``numpy`` backend runs keras-free ; the
``tensor`` backend (marked ``keras``) checks that the keras path matches the
same reference. Graph / XLA compatibility lives in ``tensorflow``-marked tests.
"""

import numpy as np
import pytest

from utils.keras import ops
from ._base import check_ops, prepare, to_tensor

_UNARY_REF = {
    'abs'   : np.abs,
    'sum'   : np.sum,
    'min'   : np.min,
    'max'   : np.max,
    'mean'  : np.mean,
    'std'   : np.std,
    'norm'  : np.linalg.norm,
    'normalize' : lambda x: x / np.linalg.norm(x, axis = -1, keepdims = True),
}

_BINARY_REF = {
    'add'       : np.add,
    'subtract'  : np.subtract,
    'multiply'  : np.multiply,
    'divide'    : np.divide,
    'divide_no_nan' : np.divide,    # references use non-zero denominators
}


# --- unary / binary elementwise ----------------------------------------------------

@pytest.mark.parametrize('name', list(_UNARY_REF))
def test_unary_operation(name, backend, asserts):
    x   = np.random.default_rng(0).normal(size = (5, 6)).astype('float32')
    ref = _UNARY_REF[name](x)
    check_ops(getattr(ops, name), prepare(backend, x), ref = ref, backend = backend, asserts = asserts)

@pytest.mark.parametrize('name', list(_BINARY_REF))
def test_binary_operation(name, backend, asserts):
    rng = np.random.default_rng(0)
    x   = rng.normal(size = (5, 6)).astype('float32')
    y   = rng.normal(size = (1, 6)).astype('float32')     # broadcasting + never zero
    ref = _BINARY_REF[name](x, y)
    check_ops(
        getattr(ops, name), prepare(backend, x), prepare(backend, y),
        ref = ref, backend = backend, asserts = asserts
    )

@pytest.mark.keras
@pytest.mark.parametrize('name', list(_BINARY_REF))
@pytest.mark.parametrize('tensor_arg', ['x', 'y'])
def test_binary_operation_mixed(name, tensor_arg, asserts):
    """ As soon as one input is a tensor, the keras path is taken (tensor output). """
    rng = np.random.default_rng(0)
    x   = rng.normal(size = (5, 6)).astype('float32')
    y   = rng.normal(size = (1, 6)).astype('float32')
    ref = _BINARY_REF[name](x, y)

    xb = to_tensor(x) if tensor_arg == 'x' else x
    yb = to_tensor(y) if tensor_arg == 'y' else y

    out = getattr(ops, name)(xb, yb)
    asserts.assert_tensor(out)
    asserts.assert_equal(ref, out)

def test_divide_no_nan_zero_denominator(backend, asserts):
    """ Zero denominators must yield ``0`` (not nan / uninitialized) on both backends. """
    x = np.array([1., 2., 3.], dtype = 'float32')
    y = np.array([0., 2., 0.], dtype = 'float32')
    check_ops(
        ops.divide_no_nan, prepare(backend, x), prepare(backend, y),
        ref = np.array([0., 1., 0.], dtype = 'float32'), backend = backend, asserts = asserts
    )


# --- unique ------------------------------------------------------------------------

def test_unique(backend, asserts):
    base = np.array([1, 2, 1, 3, 5, 1, 3])
    check_ops(
        ops.unique, prepare(backend, base),
        ref = np.unique(base), backend = backend, asserts = asserts
    )


# --- take / take_along_axis --------------------------------------------------------

@pytest.mark.parametrize('dtype', ['uint8', 'int32', 'float32'])
@pytest.mark.parametrize('shape', [(16, ), (4, 4), (4, 2, 2), (2, 2, 2, 2)])
def test_take(backend, shape, dtype, asserts):
    x = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
    data = prepare(backend, x)
    for axis in range(-1, len(shape)):
        check_ops(
            ops.take, data, [1, 0], axis = axis,
            ref = np.take(x, [1, 0], axis = axis), backend = backend, asserts = asserts
        )

@pytest.mark.parametrize('shape', [(16, ), (4, 4), (4, 2, 2), (2, 2, 2, 2)])
def test_take_along_axis(backend, shape, asserts):
    rng      = np.random.default_rng(0)
    shuffled = np.arange(np.prod(shape)).astype(np.int32)
    rng.shuffle(shuffled)
    shuffled = shuffled.reshape(shape)

    data = prepare(backend, shuffled)
    for axis in range(-1, len(shape)):
        indices = np.argsort(shuffled, axis = axis).astype(np.int32)
        check_ops(
            ops.take_along_axis, data, indices, axis = axis,
            ref = np.take_along_axis(shuffled, indices, axis = axis),
            backend = backend, asserts = asserts
        )

@pytest.mark.tensorflow
@pytest.mark.parametrize('shape', [(4, 4), (4, 2, 2)])
def test_take_along_axis_graph(shape, asserts):
    rng      = np.random.default_rng(0)
    shuffled = np.arange(np.prod(shape)).astype(np.int32)
    rng.shuffle(shuffled)
    shuffled = shuffled.reshape(shape)

    for axis in range(-1, len(shape)):
        indices = np.argsort(shuffled, axis = axis).astype(np.int32)
        asserts.assert_graph_compatible(
            ops.take_along_axis, shuffled, indices, axis = axis,
            target = np.take_along_axis(shuffled, indices, axis = axis)
        )


# --- bincount (custom, supports 2D) — NEW ------------------------------------------

def test_bincount_1d(backend, asserts):
    x = np.array([1, 1, 2, 3, 3, 3, 0], dtype = 'int32')
    check_ops(
        ops.bincount, prepare(backend, x), minlength = 5,
        ref = np.bincount(x, minlength = 5), backend = backend, asserts = asserts
    )

def test_bincount_2d_numpy(asserts):
    """ keras-free : the custom numpy ``bincount`` handles 2D inputs (per-row counts). """
    x   = np.array([[0, 1, 1], [2, 2, 2]], dtype = 'int32')
    out = ops.bincount(x, minlength = 3)
    asserts.assert_array(out)
    asserts.assert_equal(np.array([[1, 2, 0], [0, 0, 3]], dtype = 'int32'), out)


# --- isin (custom keras impl, numpy falls back to np.isin) — NEW -------------------

def test_isin(backend, asserts):
    elements      = np.array([1, 2, 3, 4, 5], dtype = 'int32')
    test_elements = np.array([2, 4, 6], dtype = 'int32')
    check_ops(
        ops.isin, prepare(backend, elements), prepare(backend, test_elements),
        ref = np.isin(elements, test_elements), backend = backend, asserts = asserts
    )


# --- normalize : zero-norm edge (regression) ---------------------------------------

def test_normalize_zero_norm_row(asserts):
    """ keras-free : a zero-norm row must yield ``0`` (not uninitialized values).

        Regression for `_np_normalize` : `np.divide(..., where=...)` without an explicit
        zero-initialized `out` left the masked entries uninitialized.
    """
    x   = np.array([[3., 4.], [0., 0.]], dtype = 'float32')
    out = ops.normalize(x)

    asserts.assert_array(out)
    asserts.assert_equal(np.array([[0.6, 0.8], [0., 0.]], dtype = 'float32'), out)


# --- concatenate (custom `nested_arg` dispatch) ------------------------------------

@pytest.mark.parametrize('axis', [0, 1])
def test_concatenate(backend, axis, asserts):
    a = np.arange(6).reshape(2, 3).astype('float32')
    b = np.arange(6, 12).reshape(2, 3).astype('float32')
    inputs = [prepare(backend, a), prepare(backend, b)]
    check_ops(
        ops.concatenate, inputs, axis = axis,
        ref = np.concatenate([a, b], axis = axis), backend = backend, asserts = asserts
    )

@pytest.mark.keras
def test_concatenate_mixed(asserts):
    """ A list mixing a numpy array and a tensor takes the keras path (tensor output). """
    a = np.arange(6).reshape(2, 3).astype('float32')
    b = np.arange(6, 12).reshape(2, 3).astype('float32')

    out = ops.concatenate([a, to_tensor(b)], axis = 0)
    asserts.assert_tensor(out)
    asserts.assert_equal(np.concatenate([a, b], axis = 0), out)


# --- expand_dims -------------------------------------------------------------------

@pytest.mark.parametrize('axis', [0, 1, -1])
def test_expand_dims(backend, axis, asserts):
    x = np.arange(6).reshape(2, 3).astype('float32')
    check_ops(
        ops.expand_dims, prepare(backend, x), axis,
        ref = np.expand_dims(x, axis), backend = backend, asserts = asserts
    )


# --- swapaxes (custom tensorflow impl) ---------------------------------------------

@pytest.mark.parametrize('axis1,axis2', [(0, 1), (0, -1), (1, 2)])
def test_swapaxes(backend, axis1, axis2, asserts):
    x = np.arange(24).reshape(2, 3, 4).astype('float32')
    check_ops(
        ops.swapaxes, prepare(backend, x), axis1, axis2,
        ref = np.swapaxes(x, axis1, axis2), backend = backend, asserts = asserts
    )

@pytest.mark.tensorflow
def test_swapaxes_graph(asserts):
    x = np.arange(24).reshape(2, 3, 4).astype('float32')
    asserts.assert_graph_compatible(ops.swapaxes, x, 0, 2, target = np.swapaxes(x, 0, 2))
