# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import keras
import numpy as np
import keras.ops as K

from keras import tree
from functools import cache

from loggers import timer
from .ops_builder import build_op, build_custom_op, executing_eagerly

class TensorflowNotAvailable(Exception):
    def __init__(self):
        super().__init__('Tensorflow is not available')

"""
Functions to convert `Tensor` to `np.ndarray` and vice-versa, and create `Tensor`
"""

def _tf_fill_with_dtype(shape, value, dtype = None):
    import tensorflow as tf
    res = tf.fill(shape, value)
    return cast(res, dtype) if dtype is not None else res

@timer(debug = True)
def convert_to_numpy(x, dtype = None):
    if not isinstance(x, np.ndarray):
        if 'tensorflow' in sys.modules:
            import tensorflow as tf
            if not tf.executing_eagerly():
                return tf.cast(x, dtype) if dtype is not None else x
            elif tf.is_tensor(x):
                x = x.numpy()
        elif not ops.executing_eagerly():
            return convert_to_tensor(x, dtype)
        
        if not isinstance(x, np.ndarray): # if it was not a `tf.Tensor`
            if not K.is_tensor(x): return np.array(x, dtype = dtype)
            x = K.convert_to_numpy(x)
    elif dtype == 'float' and is_float(x):
        return x
    
    return x if dtype is None else cast(x, dtype)

def get_convertion_dtype(x):
    if is_tensor(x): return dtype_to_str(x.dtype)
    elif isinstance(x, (list, tuple)):
        if len(x) == 0: return 'float32'
        data = tree.flatten(x)[0]
    else:
        data = x
    
    if isinstance(data, str) or (isinstance(data, np.ndarray) and data.dtype == object):
        return 'string'
    elif is_bool(data):     return 'bool'
    elif is_float(data):    return 'float32'
    elif isinstance(data, int): return 'int32'
    elif hasattr(data, 'dtype'):
        _dtype = dtype_to_str(data.dtype)
        if _dtype[0] == 'u':    return _dtype
        elif 'int' in _dtype:   return 'int32'
        else: raise ValueError('Unknown data dtype : {}'.format(data.dtype))
    else: raise ValueError('Unknown data type : {}\n{}'.format(type(data), data))

@timer(debug = True)
def convert_to_tensor(x, dtype = None):
    """
        Converts `data` ((list / dict) of `str` or `np.ndarray`) by inferring the output dtype :
            - bool      --> tf.bool
            - str       --> tf.string
            - uint8     --> tf.uint8    # useful for images
            - int / int32 / int64       --> tf.int32
            - float / float32 / float64 --> tf.float32
        This function uses the `tf.cast` with the inferred output type
        The other benefit is the standardization between types (e.g., floating --> tf.float32)
    """
    if dtype is not None:
        if not is_tensor(x):
            return convert_to_tensor_op(x, dtype_to_str(dtype))
        elif dtype == 'float' and is_float(x):
            return x
        else:
            return cast(x, dtype)
    
    elif x is None or is_tensor(x): return x
    elif isinstance(x, dict): return {k : convert_to_tensor(v) for k, v in x.items()}
    
    return convert_to_tensor_op(x, get_convertion_dtype(x))

def convert_to_tf_tensor(x, dtype = None):
    try:
        import tensorflow as tf
    except ImportError:
        raise TensorflowNotAvailable()
    
    if dtype is not None:
        if not tf.is_tensor(x):
            if is_tensor(x): x = convert_to_numpy(x)
            return tf.convert_to_tensor(x, dtype_to_str(dtype))
        elif dtype == 'float' and is_float(x):
            return x
        else:
            return tf.cast(x, dtype)
    
    elif x is None or tf.is_tensor(x): return x
    elif isinstance(x, dict): return {k : convert_to_tf_tensor(v) for k, v in x.items()}
    
    dtype = get_convertion_dtype(x)
    if is_tensor(x): x = convert_to_numpy(x)
    return tf.convert_to_tensor(x, dtype)

def _tf_array(* args, dtype = None, ** kwargs):
    import tensorflow as tf
    arr = tf.constant(* args, ** kwargs)
    if dtype is not None: arr = tf.cast(arr, dtype)
    return arr

convert_to_tensor_op    = build_op('convert_to_tensor', disable_np = True)
is_array    = build_op('is_tensor', np_op = lambda x: isinstance(x, np.ndarray))
is_tensor   = build_op('is_tensor', disable_np = True)

array   = constant  = build_op('array', tf_op = _tf_array, disable_np = True)
empty   = build_op('empty', 'zeros', disable_np = True)
zeros   = build_op('zeros', disable_np = True)
ones    = build_op('ones', disable_np = True)
full    = fill  = build_op('full', _tf_fill_with_dtype, disable_np = True)
eye     = build_custom_op(tf_fn = 'eye', keras_fn = 'eye', disable_np = True)

zeros_like  = empty_like    = build_op('zeros_like', disable_np = True)
ones_like   = build_op('ones_like', disable_np = True)

range   = arange    = build_op('arange', 'range', disable_np = True)
linspace    = build_op('linspace', disable_np = True)



"""
Functions to convert `Tensor` types
"""

@timer(debug = True)
@cache
def dtype_to_str(dtype):
    if isinstance(dtype, str):
        return dtype if dtype != 'float' else keras.backend.floatx()
    elif hasattr(dtype, 'name'):
        return dtype.name
    try:
        return keras.backend.standardize_dtype(dtype)
    except ValueError:
        return str(dtype)

@timer(debug = True)
def cast(x, dtype):
    dtype = dtype_to_str(dtype)
    if not is_array(x): return convert_to_tensor_op(x, dtype)
    return x if dtype_to_str(x.dtype) == dtype else cast_op(x, dtype)

def _np_cast(x, dtype):
    return x.astype(dtype) if isinstance(x, np.ndarray) else np.array(x, dtype = dtype)

cast_op = build_op('cast', np_op = _np_cast)
real    = build_op('real', 'math.real')

def is_float(x):
    return (hasattr(x, 'dtype') and 'float' in dtype_to_str(x.dtype)) or isinstance(x, float)

def is_int(x):
    return (hasattr(x, 'dtype') and 'int' in dtype_to_str(x.dtype)) or isinstance(x, int)

def is_bool(x):
    return (hasattr(x, 'dtype') and 'bool' in dtype_to_str(x.dtype)) or isinstance(x, bool)

def is_numeric(x):  return is_float(x) or is_int(x)
def is_string(x):   return (hasattr(x, 'dtype') and x.dtype.name == 'string') or isinstance(x, str)

@timer(debug = True)
def convert_data_dtype(x, dtype, source_dtype = None):
    """
        Similar to the `tf.image.convert_image_dtype`, this function casts after re-scaling the data based on its data range / the expected dtype data range
        If `x.dtype` is floating point, the expected range is [0, 1], while integer are expected in [0, dtype.max]
    """
    source_dtype = dtype_to_str(source_dtype if source_dtype else x.dtype)
    dtype = dtype_to_str(dtype)
    if dtype == source_dtype: return x
    
    if 'float' in source_dtype:
        if 'float' in dtype: return cast(x, dtype)
        return cast(x * np.iinfo(dtype).max, dtype)
    if 'float' in dtype: return cast(x / np.iinfo(source_dtype).max, dtype)
    return cast(x * (np.iinfo(dtype).max / np.iinfo(source_dtype).max), dtype)

complex = build_custom_op(name = 'complex', jax_fn = 'lax.complex')

"""
Functions to manipulate `Tensor` shapes
"""

def _np_unstack(x, axis = None, ** _):
    _python_range   = __builtins__['range']
    if axis in (None, -1): return [x[..., i] for i in _python_range(x.shape[-1])]
    if axis < 0:   axis = len(x.shape) + axis
    _python_slice   = __builtins__['slice']
    
    _slices = tuple([_python_slice(0, None) for _ in _python_range(axis)])
    return [x[_slices + (i, )] for i in _python_range(x.shape[axis])]

def _np_stack(x, axis = None):
    return np.array(x) if axis in (0, None) else np.stack(x, axis = axis)

ensure_shape    = build_custom_op(
    tf_fn       = 'ensure_shape',
    keras_fn    = lambda x, shape: x,
    np_fn       = lambda x, shape: x,
    name    = 'ensure_shape'
)

shape   = build_op(lambda x: x.shape, 'shape', np_op = lambda x: x.shape, name = 'shape')
size    = build_op('size')
rank    = ndim  = build_op(lambda x: len(shape(x)), 'rank', np_op = lambda x: x.ndim, name = 'rank')

pad     = build_op('pad')
tile    = build_op('tile')
repeat  = build_op('repeat')
squeeze = build_op('squeeze')
reshape = build_op('reshape')
swapaxes    = K.swapaxes
transpose   = build_op('transpose')
expand_dims = build_op('expand_dims')
broadcast_to    = build_op('broadcast_to')

split   = build_op('split')
stack   = build_op('stack', np_op = _np_stack, nested = True)
unstack = build_op('unstack', np_op = _np_unstack)
concat  = concatenate   = build_op('concatenate', 'concat', nested = True)
flip    = reverse   = build_op('flip', K.flip)

"""
Indexing functions
"""

def _np_scatter_update(x, indices, updates):
    x[indices] = updates
    return x

def _np_slice(x, start_indices, lengths):
    _python_slice = __builtins__['slice']
    slices  = tuple([
        _python_slice(start, start + length) for start, length in zip(start_indices, lengths)
    ])
    return x[slices]

def _tf_slice_update(x, start_indices, update):
    from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice as xla_dynamic_update_slice

    return xla_dynamic_update_slice(x, update, start_indices)

def _np_slice_update(array, start_indices, updates):
    _python_slice   = __builtins__['slice']
    slices  = tuple([
        _python_slice(start, start + length)
        for start, length in zip(start_indices, updates.shape)
    ])
    array[slices] = updates
    return array

def _tf_take_along_axis(arr, indices, axis):
    import tensorflow as tf
    # adapted from https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/numpy_ops/np_array_ops.py#L1602-L1663
    arr     = tf.convert_to_tensor(arr)
    indices = tf.convert_to_tensor(indices)
    
    if axis is None: return take_along_axis(tf.reshape(arr, [-1]), indices, 0)
    
    rank    = len(tf.shape(arr))
    if axis < 0: axis = axis + rank

    # Broadcast shapes to match, ensure that the axis of interest is not broadcast.
    arr_shape_original      = tf.shape(arr, out_type = indices.dtype)
    indices_shape_original  = tf.shape(indices, out_type = indices.dtype)
    
    arr_shape       = tf.tensor_scatter_nd_update(arr_shape_original, [[axis]], [1])
    indices_shape   = tf.tensor_scatter_nd_update(indices_shape_original, [[axis]], [1])
    
    broadcasted_shape = tf.broadcast_dynamic_shape(arr_shape, indices_shape)
    
    arr_shape       = tf.tensor_scatter_nd_update(
        broadcasted_shape, [[axis]], [arr_shape_original[axis]]
    )
    indices_shape   = tf.tensor_scatter_nd_update(
        broadcasted_shape, [[axis]], [indices_shape_original[axis]]
    )
    arr     = tf.broadcast_to(arr, arr_shape)
    indices = tf.broadcast_to(indices, indices_shape)

    # Save indices shape so we can restore it later.
    possible_result_shape = indices.shape

    # Correct indices since gather doesn't correctly handle negative indices.
    indices = tf.where(indices < 0, indices + arr_shape[axis], indices)

    swapaxes_ = lambda t: tf.experimental.numpy.swapaxes(t, axis, -1)

    dont_move_axis_to_end = tf.equal(axis, rank - 1)
    arr, indices    = tf.cond(
        dont_move_axis_to_end,
        lambda: (arr, indices),
        lambda: (swapaxes_(arr), swapaxes_(indices))
    )

    arr_shape   = tf.shape(arr)
    arr         = tf.reshape(arr, [-1, arr_shape[-1]])

    indices_shape   = tf.shape(indices)
    indices         = tf.reshape(indices, [-1, indices_shape[-1]])

    result = tf.gather(arr, indices, batch_dims = 1)
    result = tf.reshape(result, indices_shape)
    result = tf.cond(
        dont_move_axis_to_end, lambda: result, lambda: swapaxes_(result)
    )
    result.set_shape(possible_result_shape)

    return result

slice   = dynamic_slice = build_op('slice', np_op = _np_slice)
slice_update    = update_slice  = dynamic_update_slice  = build_op(
    'slice_update', _tf_slice_update, np_op = _np_slice_update
)

scatter_update  = scatter_nd_update = build_op(
    'scatter_update', 'tensor_scatter_nd_update', np_op = _np_scatter_update
)

take    = gather    = build_custom_op(
    tf_fn = 'gather',
    torch_fn    = lambda input, indices, * args, ** kwargs: K.take(
        K.convert_to_tensor(input), K.convert_to_tensor(indices), * args, ** kwargs
    ),
    name = 'take'
)
take_along_axis = build_custom_op(tf_fn = _tf_take_along_axis, name = 'take_along_axis')


sort    = build_op('sort')
top_k   = build_op('top_k', 'nn.top_k')
argmin  = build_op('argmin', tf_kwargs = {'output_type' : 'int32'})
argmax  = build_op('argmax', tf_kwargs = {'output_type' : 'int32'})
argsort = build_op('argsort')
bincount    = build_op('bincount', 'math.bincount', tf_kwargs = {'dtype' : 'float32'})

