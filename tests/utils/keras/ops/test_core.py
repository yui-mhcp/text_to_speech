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

""" Pure-``pytest`` tests for :mod:`utils.keras.ops.core`.

Covers the most-used core helpers : tensor creation, shape / rank, dtype
inspection & conversion, casting, ``numpy`` <-> tensor conversions and the
indexing / structural ops (``slice``, ``stack``, ``unstack``, ``cond``).

Backend handling (see the ``backend`` fixture in ``conftest``) :
  - ops with a numpy path are run on both ``numpy`` (keras-free, unmarked) and
    ``tensor`` (marked ``keras``) inputs through the shared :func:`check_ops` ;
  - creation ops have ``numpy_fn is None`` (they always build a tensor) -> the
    value tests are inherently ``keras``-marked, with ``keras.ops`` as oracle ;
  - graph / XLA compatibility lives in dedicated ``tensorflow``-marked tests.
"""

import pytest
import numpy as np

from utils.keras import ops
from ._base import check_ops, prepare, to_tensor


# ===================================================================================
# Tensor creation (numpy disabled -> always keras)
# ===================================================================================

@pytest.mark.keras
@pytest.mark.parametrize('dtype', ['uint8', 'int32', 'bfloat16', 'float16', 'float32', 'bool'])
@pytest.mark.parametrize('shape', [(), (5, ), (5, 5)])
@pytest.mark.parametrize('operation', ['array', 'empty', 'zeros', 'ones', 'full'])
def test_tensor_creation(operation, shape, dtype, asserts):
    import keras
    import keras.ops as K

    fn = getattr(ops, operation)
    assert fn.numpy_fn is None, 'numpy should be disabled for creation ops'

    args = (shape, )
    if operation == 'full': args += (5, )

    target = getattr(K, operation)(* args, dtype = dtype)
    value  = fn(* args, dtype = dtype)
    asserts.assert_tensor(value)

    if dtype == 'bfloat16':
        # bfloat16 values can't be compared reliably -> only check the shape
        asserts.assert_equal(tuple(target.shape), tuple(value.shape))
    elif operation != 'empty' or keras.backend.backend() == 'tensorflow':
        asserts.assert_equal(target, value)
    else:
        # `empty` is uninitialized on non-tensorflow backends -> shape-only
        asserts.assert_equal(shape, tuple(value.shape))


@pytest.mark.keras
@pytest.mark.parametrize('dtype', ['int32', 'float32'])
@pytest.mark.parametrize('start,end,step', [
    (5, None, None), (5, 10, None), (-5, 10, None), (0, 10, 2), (10, 0, -1)
])
def test_range(start, end, step, dtype, asserts):
    import keras.ops as K

    assert ops.arange.numpy_fn is None, 'numpy should be disabled'
    args = [a for a in (start, end, step) if a is not None]
    check_ops(
        ops.arange, * args, dtype = dtype,
        ref = K.arange(* args, dtype = dtype), backend = 'tensor', asserts = asserts
    )


@pytest.mark.keras
@pytest.mark.parametrize('dtype', ['int32', 'bool', 'float32'])
def test_eye(dtype, asserts):
    import keras.ops as K

    assert ops.eye.numpy_fn is None, 'numpy should be disabled'
    check_ops(
        ops.eye, 5, dtype = dtype,
        ref = K.eye(5, dtype = dtype), backend = 'tensor', asserts = asserts
    )


# ===================================================================================
# Shape / rank (always return python `tuple` / `int`)
# ===================================================================================

@pytest.mark.parametrize('data,target', [
    pytest.param(1,           (),     id = 'int'),
    pytest.param(1.,          (),     id = 'float'),
    pytest.param(True,        (),     id = 'bool'),
    pytest.param([1, 2],      (2, ),  id = 'list'),
    pytest.param([[1], [2]],  (2, 1), id = 'nested_list'),
])
def test_shape_python_objects(data, target):
    """ keras-free : python scalars / (nested) lists go through ``np.shape`` / ``np.ndim``. """
    assert ops.shape(data) == target
    assert isinstance(ops.shape(data), tuple)
    assert ops.rank(data) == len(target)
    assert isinstance(ops.rank(data), int)

@pytest.mark.parametrize('shape', [(3, ), (5, 5)])
def test_shape_array_like(backend, shape):
    data = prepare(backend, np.ones(shape, dtype = 'float32'))

    assert ops.shape(data) == shape
    assert isinstance(ops.shape(data), tuple)
    assert ops.rank(data) == len(shape)
    assert isinstance(ops.rank(data), int)

@pytest.mark.tensorflow
@pytest.mark.parametrize('shape', [(3, ), (5, 5)])
def test_shape_graph(shape, asserts):
    data = np.ones(shape, dtype = 'float32')
    asserts.assert_graph_compatible(ops.shape, data, target = shape, is_tensor_output = True)
    asserts.assert_graph_compatible(ops.rank, data, target = len(shape))


# ===================================================================================
# dtype inspection
# ===================================================================================

@pytest.mark.parametrize('data,target', [
    pytest.param(0,                 'int32',   id = 'int'),
    pytest.param(0.,                'float32', id = 'float'),
    pytest.param(True,              'bool',    id = 'bool'),
    pytest.param('Hello World !',   'string',  id = 'string'),
    pytest.param([0, 1, 2],         'int32',   id = 'list_int'),
    pytest.param([0., 1., 2.],      'float32', id = 'list_float'),
    pytest.param([0, 1., 2.],       'int32',   id = 'list_mixed'),    # takes the 1st item type
    pytest.param([[0], [1]],        'int32',   id = 'nested_list_int'),
    pytest.param([[0.], [1.]],      'float32', id = 'nested_list_float'),
    pytest.param(np.zeros((5, ), dtype = 'bool'),    'bool',    id = 'array_bool'),
    pytest.param(np.zeros((5, ), dtype = 'uint8'),   'uint8',   id = 'array_uint8'),
    pytest.param(np.zeros((5, ), dtype = 'int32'),   'int32',   id = 'array_int32'),
    pytest.param(np.zeros((5, ), dtype = 'int64'),   'int32',   id = 'array_int64'),
    pytest.param(np.zeros((5, ), dtype = 'float16'), 'float32', id = 'array_float16'),
    pytest.param(np.zeros((5, ), dtype = 'float32'), 'float32', id = 'array_float32'),
])
def test_get_convertion_dtype(data, target):
    """ keras-free : the inferred convertion dtype standardizes numpy / python types. """
    assert ops.dtype_to_str(ops.get_convertion_dtype(data)) == target

@pytest.mark.keras
@pytest.mark.parametrize('dtype', ['bool', 'uint8', 'int32', 'float16', 'float32'])
def test_get_convertion_dtype_tensor(dtype):
    """ For a tensor, the convertion dtype is the tensor's own dtype (no standardization). """
    tensor = to_tensor(np.zeros((5, ), dtype = dtype))
    assert ops.dtype_to_str(ops.get_convertion_dtype(tensor)) == dtype


@pytest.mark.parametrize('dtype', ['int32', 'float32', 'uint8', 'bool', 'float16'])
def test_dtype_to_str_passthrough(dtype):
    """ keras-free : a standard dtype string is returned untouched. """
    assert ops.dtype_to_str(dtype) == dtype

def test_dtype_to_str_from_numpy_dtype():
    """ keras-free : numpy ``dtype`` objects resolve to their ``.name``. """
    assert ops.dtype_to_str(np.dtype('float32')) == 'float32'
    assert ops.dtype_to_str(np.zeros(1, dtype = 'int32').dtype) == 'int32'

@pytest.mark.keras
def test_dtype_to_str_float_alias():
    """ The ``'float'`` alias resolves to the keras default float dtype. """
    import keras
    assert ops.dtype_to_str('float') == keras.backend.floatx()


# ===================================================================================
# is_array / is_tensor
# ===================================================================================

@pytest.mark.parametrize('value,is_array,is_tensor', [
    pytest.param(1,                 False, False, id = 'integer'),
    pytest.param(2.5,               False, False, id = 'float'),
    pytest.param(True,              False, False, id = 'bool'),
    pytest.param('Hello World !',   False, False, id = 'string'),
    pytest.param([1, 'hello', True], False, False, id = 'list'),
    pytest.param(np.ones((5, 5)),   True,  False, id = 'array'),
])
def test_is_array_is_tensor_keras_free(value, is_array, is_tensor):
    assert ops.is_array(value) == is_array
    assert ops.is_tensor(value) == is_tensor

@pytest.mark.keras
def test_is_array_is_tensor_with_tensor():
    import keras.ops as K

    tensor = K.ones((5, 5))
    assert ops.is_array(tensor) == True
    assert ops.is_tensor(tensor) == True
    # a python list of tensors is neither an array nor a tensor
    assert ops.is_array([tensor]) == False
    assert ops.is_tensor([tensor]) == False


# ===================================================================================
# cast (+ is_int / is_float / is_numeric / is_bool)
# ===================================================================================

def _assert_dtype_predicates(value, dtype):
    assert ops.is_int(value)     == ('int' in dtype)
    assert ops.is_float(value)   == ('float' in dtype)
    assert ops.is_numeric(value) == (dtype != 'bool')
    assert ops.is_bool(value)    == (dtype == 'bool')

@pytest.mark.parametrize('dtype', ['uint8', 'int32', 'float', 'float16', 'float32', 'bool'])
def test_cast_array(dtype, asserts):
    """ keras-free : numpy inputs are cast with ``ndarray.astype`` (same instance if no-op). """
    array  = np.zeros((5, 5), dtype = 'float32')
    casted = ops.cast(array, dtype)

    asserts.assert_array(casted)
    if dtype != 'float':
        assert casted.dtype.name == dtype
    if dtype in ('float', 'float32'):
        assert array is casted, 'casting to the same dtype must return the same instance'

    _assert_dtype_predicates(casted, dtype)

@pytest.mark.keras
@pytest.mark.parametrize('dtype', ['uint8', 'int32', 'float', 'float16', 'float32', 'bfloat16', 'bool'])
def test_cast_tensor(dtype, asserts):
    import keras.ops as K

    tensor = K.zeros((5, 5), dtype = 'float32')
    casted = ops.cast(tensor, dtype)

    asserts.assert_tensor(casted)
    if dtype != 'float':
        assert ops.dtype_to_str(casted.dtype) == dtype
    if dtype in ('float', 'float32'):
        assert tensor is casted, 'casting to the same dtype must return the same instance'

    _assert_dtype_predicates(casted, dtype)

@pytest.mark.tensorflow
@pytest.mark.parametrize('dtype', ['uint8', 'int32', 'float16', 'float32'])
def test_cast_graph(dtype, asserts):
    array = np.zeros((5, 5), dtype = 'float32')
    asserts.assert_graph_compatible(ops.cast, array, dtype, target = ops.cast(array, dtype))


# ===================================================================================
# convert_to_numpy / convert_to_tensor / convert_to_tf_tensor
# ===================================================================================

@pytest.mark.parametrize('dtype', ['int32', 'float16', 'float32'])
def test_convert_to_numpy_from_array(dtype, asserts):
    """ keras-free : a numpy input is returned as-is (or re-cast) without importing keras. """
    data = np.array([1, 2], dtype = dtype)

    assert ops.convert_to_numpy(data) is data, 'same-dtype numpy input must return the same instance'

    casted = ops.convert_to_numpy(data, 'float32')
    asserts.assert_array(casted)
    assert casted.dtype.name == 'float32'
    if dtype == 'float32':
        assert casted is data

@pytest.mark.keras
@pytest.mark.parametrize('dtype', ['int32', 'float16', 'float32', 'bfloat16'])
def test_convert_to_numpy_from_tensor(dtype, asserts):
    import keras.ops as K

    tensor = K.array([1, 2], dtype = dtype)
    array  = ops.convert_to_numpy(tensor)
    asserts.assert_array(array)
    if dtype != 'bfloat16':
        assert array.dtype.name == dtype

    array = ops.convert_to_numpy(tensor, 'float32')
    asserts.assert_array(array)
    assert array.dtype.name == 'float32'

@pytest.mark.tensorflow
def test_convert_to_numpy_graph(asserts):
    data = np.array([1, 2], dtype = 'int32')
    asserts.assert_graph_compatible(
        ops.convert_to_numpy, data, 'float32', target = ops.convert_to_numpy(data, 'float32')
    )


def _make_convert_data(spec):
    """ ``'array'`` / ``'shape'`` are keras-free ; ``'tensor'`` needs keras. """
    if spec == 'array':    return np.ones((5, ), dtype = 'float16')
    if spec == 'shape':    return (256, 256)
    return to_tensor(np.ones((5, ), dtype = 'float16'))

@pytest.mark.keras
@pytest.mark.parametrize('dtype', [None, 'int32', 'float', 'float16', 'float32'])
@pytest.mark.parametrize('spec', ['array', 'tensor', 'shape'])
def test_convert_to_tensor(spec, dtype, asserts):
    import keras.ops as K

    data   = _make_convert_data(spec)
    tensor = ops.convert_to_tensor(data, dtype)

    asserts.assert_tensor(tensor)
    if dtype not in (None, 'float'):
        assert ops.dtype_to_str(tensor.dtype) == dtype
    if K.is_tensor(data) and dtype in (None, 'float', 'float16'):
        assert tensor is data, 'no-op convertions must not create a new tensor'

@pytest.mark.tensorflow
@pytest.mark.parametrize('dtype', ['int32', 'float16', 'float32'])
def test_convert_to_tensor_graph(dtype, asserts):
    data = np.ones((5, ), dtype = 'float16')
    asserts.assert_graph_compatible(
        ops.convert_to_tensor, data, dtype, target = ops.convert_to_tensor(data, dtype)
    )

@pytest.mark.tensorflow
@pytest.mark.keras
@pytest.mark.parametrize('dtype', [None, 'int32', 'float', 'float16', 'float32'])
@pytest.mark.parametrize('spec', ['array', 'tensor', 'shape'])
def test_convert_to_tf_tensor(spec, dtype, asserts):
    import keras
    import keras.ops as K

    data   = _make_convert_data(spec)
    tensor = ops.convert_to_tf_tensor(data, dtype)

    asserts.assert_tf_tensor(tensor)
    if dtype not in (None, 'float'):
        assert ops.dtype_to_str(tensor.dtype) == dtype
    if K.is_tensor(data) and dtype in (None, 'float', 'float16') and keras.backend.backend() == 'tensorflow':
        assert tensor is data, 'no-op convertions must not create a new tensor'


# ===================================================================================
# convert_data_dtype (range-aware rescaling, pure numpy) — NEW
# ===================================================================================

def test_convert_data_dtype_noop_same_dtype():
    x = np.array([1, 2, 3], dtype = 'int32')
    assert ops.convert_data_dtype(x, 'int32') is x

def test_convert_data_dtype_float_to_float(asserts):
    x = np.array([0., 0.5, 1.], dtype = 'float32')
    out = ops.convert_data_dtype(x, 'float16')
    asserts.assert_equal(x.astype('float16'), out)
    assert out.dtype.name == 'float16'

def test_convert_data_dtype_float_to_uint8(asserts):
    # floats are assumed in [0, 1] and rescaled to [0, 255]
    x   = np.array([0., 0.5, 1.], dtype = 'float32')
    out = ops.convert_data_dtype(x, 'uint8')
    asserts.assert_equal(np.array([0, 127, 255], dtype = 'uint8'), out)
    assert out.dtype.name == 'uint8'

def test_convert_data_dtype_uint8_to_float(asserts):
    x   = np.array([0, 127, 255], dtype = 'uint8')
    out = ops.convert_data_dtype(x, 'float32')
    asserts.assert_equal(x.astype('float32') / 255., out, max_err = 1e-6)
    assert out.dtype.name == 'float32'

def test_convert_data_dtype_uint8_to_uint16(asserts):
    x   = np.array([0, 1, 255], dtype = 'uint8')
    out = ops.convert_data_dtype(x, 'uint16')
    asserts.assert_equal(np.array([0, 257, 65535], dtype = 'uint16'), out)
    assert out.dtype.name == 'uint16'


# ===================================================================================
# slice / slice_update
# ===================================================================================

def _ref_slice(arr, start, lengths):
    idx = tuple(slice(s, s + l) for s, l in zip(start, lengths))
    return arr[idx]

@pytest.mark.parametrize('start,lengths', [
    ([0, 0, 0], None),       # full slice
    ([0, 0, 0], [1, 3, 2]),
    ([1, 2, 0], [1, 1, 2]),
])
def test_slice(backend, start, lengths, asserts):
    base = np.arange(16).reshape(2, 4, 2).astype('int32')
    if lengths is None: lengths = list(base.shape)

    data = prepare(backend, base)
    check_ops(
        ops.slice, data, start, lengths,
        ref = _ref_slice(base, start, lengths), backend = backend, asserts = asserts
    )

def test_slice_update(backend, asserts):
    base   = np.arange(16).reshape(2, 4, 2).astype('int32')
    update = np.arange(4).reshape(2, 2, 1).astype('int32')

    ref = base.copy()
    ref[0:2, 1:3, 0:1] = update     # start = [0, 1, 0], update.shape = (2, 2, 1)

    data = prepare(backend, base)
    value = check_ops(
        ops.slice_update, data, [0, 1, 0], prepare(backend, update),
        ref = ref, backend = backend, asserts = asserts
    )
    if backend == 'numpy':
        assert value is data, 'the numpy slice update must be in-place'


# ===================================================================================
# stack / unstack
# ===================================================================================

def test_stack_scalars(asserts):
    check_ops(
        ops.stack, [0, 1, 2], axis = 0,
        ref = np.array([0, 1, 2]), backend = 'numpy', asserts = asserts
    )

@pytest.mark.parametrize('arrays,axis', [
    ([np.array([1, 2]), np.array([2, 3])],       0),
    ([np.array([1, 2]), np.array([2, 3])],       1),
    ([np.array([[1], [2]]), np.array([[2], [3]])], 0),
])
def test_stack_arrays(backend, arrays, axis, asserts):
    inputs = [prepare(backend, a) for a in arrays]
    check_ops(
        ops.stack, inputs, axis = axis,
        ref = np.stack(arrays, axis = axis), backend = backend, asserts = asserts
    )

@pytest.mark.parametrize('axis', [0, 1])
def test_unstack(backend, axis, asserts):
    base = np.arange(6).reshape((3, 2))
    ref  = list(base) if axis == 0 else list(base.T)

    check_ops(
        ops.unstack, prepare(backend, base), axis = axis,
        ref = ref, backend = backend, asserts = asserts
    )


# ===================================================================================
# cond (lazy branch semantics) — NEW
# ===================================================================================

@pytest.mark.parametrize('pred', [True, False])
def test_cond_numpy(pred, asserts):
    """ keras-free : ``cond`` selects (and only evaluates) the matching branch. """
    out = ops.cond(pred, lambda: np.ones((2, )), lambda: np.zeros((2, )))
    asserts.assert_equal(np.ones((2, )) if pred else np.zeros((2, )), out)

def test_cond_only_evaluates_selected_branch():
    def boom(): raise AssertionError('the unselected branch must not be evaluated')

    assert ops.cond(True, lambda: 1, boom) == 1
    assert ops.cond(False, boom, lambda: 0) == 0

@pytest.mark.keras
def test_cond_tensor(asserts):
    import keras.ops as K

    pred = K.convert_to_tensor(True)
    out  = ops.cond(pred, lambda: K.ones((2, )), lambda: K.zeros((2, )))
    asserts.assert_tensor(out)
    asserts.assert_equal(np.ones((2, ), dtype = 'float32'), out)


# ===================================================================================
# scatter / scatter_update
# ===================================================================================
# Reference follows the `keras.ops.scatter` convention : `indices` has shape
# `(num_updates, rank)`, each row pointing at one cell of the `shape`-d output.
# NB: the current numpy impl (`x[indices] = updates`) does NOT follow this convention,
# so the `numpy` backend is expected to surface the divergence (kept per design).

def test_scatter(backend, asserts):
    indices = np.array([[0, 1], [2, 0]], dtype = 'int32')
    values  = np.array([5., 7.], dtype = 'float32')

    ref = np.zeros((3, 3), dtype = 'float32')
    ref[0, 1], ref[2, 0] = 5., 7.

    check_ops(
        ops.scatter, prepare(backend, indices), prepare(backend, values), (3, 3),
        ref = ref, backend = backend, asserts = asserts
    )

def test_scatter_update(backend, asserts):
    base    = np.zeros((3, 3), dtype = 'float32')
    indices = np.array([[0, 1], [2, 0]], dtype = 'int32')
    updates = np.array([5., 7.], dtype = 'float32')

    ref = base.copy()
    ref[0, 1], ref[2, 0] = 5., 7.

    check_ops(
        ops.scatter_update, prepare(backend, base), prepare(backend, indices), prepare(backend, updates),
        ref = ref, backend = backend, asserts = asserts
    )


# ===================================================================================
# while_loop (keras-free numpy path) — regression
# ===================================================================================

def test_while_loop_numpy(asserts):
    """ keras-free : the numpy ``while_loop`` runs without importing keras.

        Regression for `_np_while`, which used `keras.tree.map_structure(convert_to_tensor)`
        (undefined `tree` + wrong convertion) — now maps `convert_to_numpy` keras-free.
    """
    cond = lambda i, acc: i < 5
    body = lambda i, acc: (i + 1, acc + i)

    i, acc = ops.while_loop(cond, body, (np.array(0, 'int32'), np.array(0, 'int32')))

    asserts.assert_equal(5, i)
    asserts.assert_equal(10, acc)    # 0 + 1 + 2 + 3 + 4
    asserts.assert_array(acc)
