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
from .ops_builder import _is_numpy, _import_functions, build_op, build_custom_op, executing_eagerly, is_tensorflow_backend, is_torch_backend

class TensorflowNotAvailable(Exception):
    def __init__(self):
        super().__init__('Tensorflow is not available')

""" `Tensor` creation functions """

array   = constant  = build_op('array', disable_np = True)
full    = fill  = build_op('full', disable_np = True)
full_like   = fill_like = build_op('full_like', disable_np = True)
range   = arange    = build_op('arange', disable_np = True)
globals().update({
    k : build_op(k, disable_np = True)
    for k in (
        'empty', 'zeros', 'ones', 'zeros_like', 'ones_like', 'linspace',
        'tri', 'tril', 'triu'
    )
})
eye     = build_custom_op(tf_fn = 'eye', keras_fn = 'eye', disable_np = True)

""" Convertion functions (`Tensor` <--> `ndarray`) """

@timer(debug = True)
def is_array(x):
    return isinstance(x, np.ndarray) or is_tensor(x)

@timer(debug = True)
def convert_to_numpy(x, dtype = None, copy = False):
    if not isinstance(x, np.ndarray):
        if 'tensorflow' in sys.modules:
            import tensorflow as tf
            if not tf.executing_eagerly():
                if dtype == 'float' and is_float(x): return tf.convert_to_tensor(x)
                return tf.cast(x, dtype) if dtype is not None else x
            elif tf.is_tensor(x):
                x = x.numpy()
        elif not executing_eagerly():
            return convert_to_tensor(x, dtype)
        
        if not isinstance(x, np.ndarray): # if it was not a `Tensor`
            if not K.is_tensor(x):
                if hasattr(x, 'cpu'):   x = x.cpu().numpy()
                else:   return np.array(x, dtype = dtype)
            x = K.convert_to_numpy(x)
    elif copy:
        x = x.copy()
    
    if (dtype is None) or (dtype == 'float' and is_float(x)) or (dtype == 'int' and is_int(x)):
        return x
    return cast(x, dtype)

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
    if isinstance(x, dict): return {k : convert_to_tensor(v, dtype) for k, v in x.items()}
    
    if dtype is not None:
        if not is_tensor(x):
            return convert_to_tensor_op(x, dtype_to_str(dtype))
        elif (dtype == 'float' and is_float(x)) or (dtype == 'int' and is_int(x)):
            return x
        else:
            return cast(x, dtype)
    
    elif x is None or is_tensor(x): return x
    
    return convert_to_tensor_op(x, get_convertion_dtype(x))

def convert_to_tf_tensor(x, dtype = None):
    if is_tensorflow_backend(): return convert_to_tensor(x, dtype)
    
    try:
        import tensorflow as tf
    except ImportError:
        raise TensorflowNotAvailable()
    
    if isinstance(x, dict): return {k : convert_to_tf_tensor(v, dtype) for k, v in x.items()}
    if dtype is not None:
        if not tf.is_tensor(x):
            if K.is_tensor(x): x = convert_to_numpy(x)
            return tf.convert_to_tensor(x, dtype_to_str(dtype))
        elif (dtype == 'float' and is_float(x)) or (dtype == 'int' and is_int(x)):
            return x
        else:
            return tf.cast(x, dtype)
    
    elif x is None or tf.is_tensor(x): return x
    
    dtype = get_convertion_dtype(x)
    if K.is_tensor(x): x = convert_to_numpy(x)
    return tf.convert_to_tensor(x, dtype)

def convert_to_torch_tensor(x, dtype = None):
    if is_torch_backend(): return convert_to_tensor(x, dtype)
    
    import keras.src.backend.torch as torch_backend
    
    if isinstance(x, dict): return {k : convert_to_torch_tensor(v, dtype) for k, v in x.items()}
    if dtype is not None:
        if not torch_backend.is_tensor(x):
            if K.is_tensor(x): x = convert_to_numpy(x)
            return torch_backend.convert_to_tensor(x, dtype_to_str(dtype))
        elif (dtype == 'float' and is_float(x)) or (dtype == 'int' and is_int(x)):
            return x
        else:
            return torch_backend.cast(x, dtype)
    
    elif x is None or torch_backend.is_tensor(x): return x
    
    dtype = get_convertion_dtype(x)
    if K.is_tensor(x): x = convert_to_numpy(x)
    return torch_backend.convert_to_tensor(x, dtype)

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

def is_torch_tensor(x):
    if 'torch' not in sys.modules: return False
    import torch
    return torch.is_tensor(x)

convert_to_tensor_op    = build_op('convert_to_tensor', disable_np = True)
is_tensor   = build_op('is_tensor', disable_np = True)

def _fast_ndim(x):
    """ Computes `len(x.shape)`, which is equivalent, but faster, than `K.ndim` """
    return len(shape(x))

def _fast_shape(x):
    """ Returns `x.shape`, which is equivalent, but faster than `K.shape` """
    return x.shape if hasattr(x, 'shape') else np.shape(x)

shape   = build_op(_fast_shape, 'shape', np_op = _fast_shape, name = 'shape')
ndim    = rank  = build_op(_fast_ndim, _fast_ndim, np_op = 'ndim', name = 'rank')

ensure_shape    = build_custom_op(
    tf_fn       = 'ensure_shape',
    keras_fn    = lambda x, shape: x,
    np_fn       = lambda x, shape: x,
    name    = 'ensure_shape'
)

""" ``dtype` convertion functions """

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
imag    = build_op('imag')
real    = build_op('real')

""" Indexing functions """

def _np_slice(x, start_indices, lengths):
    _python_slice = __builtins__['slice']
    slices  = tuple([
        _python_slice(start, start + length)
        for start, length in zip(start_indices, lengths)
    ])
    return x[slices]

def _np_scatter(indices, values, shape):
    return _np_scatter_update(np.zeros(shape, dtype = values.dtype), indices, values)

def _np_scatter_update(x, indices, updates):
    x[indices] = updates
    return x

def _np_slice_update(array, start_indices, updates):
    _python_slice   = __builtins__['slice']
    slices  = tuple([
        _python_slice(start, start + length)
        for start, length in zip(start_indices, updates.shape)
    ])
    array[slices] = updates
    return array

slice   = dynamic_slice = build_op('slice', np_op = _np_slice)
slice_update    = update_slice  = dynamic_update_slice  = build_op(
    'slice_update', np_op = _np_slice_update
)

scatter = build_op('scatter', np_op = _np_scatter)
scatter_update  = scatter_nd_update = build_op('scatter_update', np_op = _np_scatter_update)

""" Other `core` functions """

def _check_numpy_while(args, kwargs, _):
    loop_vars = args[2] if len(args) >= 3 else kwargs['loop_vars']
    return _is_numpy([loop_vars], {}, nested = isinstance(loop_vars, (list, tuple)))

def _np_cond(c, t, f):
    return t if c else f

def _np_stack(x, axis = None):
    return np.array(x) if axis in (0, None) else np.stack(x, axis = axis)

def _np_unstack(x, axis = None, ** _):
    if axis == 0: return list(x)
    
    _python_range   = __builtins__['range']
    if axis in (None, -1): return [x[..., i] for i in _python_range(x.shape[-1])]
    
    if axis < 0:   axis = len(x.shape) + axis
    _python_slice   = __builtins__['slice']
    
    _slices = tuple([_python_slice(0, None) for _ in _python_range(axis)])
    return [x[_slices + (i, )] for i in _python_range(x.shape[axis])]

def _np_while(cond, body, loop_vars, maximum_iterations = None):
    """
        This function comes from the official keras repo :
        https://github.com/keras-team/keras/blob/master/keras/src/backend/numpy/core.py
    """
    current_iter = 0
    iteration_check = (
        lambda iter: maximum_iterations is None or iter < maximum_iterations
    )
    is_tuple = isinstance(loop_vars, (tuple, list))
    loop_vars = tuple(loop_vars) if is_tuple else (loop_vars, )
    loop_vars = tree.map_structure(convert_to_tensor, loop_vars)
    while cond(* loop_vars) and iteration_check(current_iter):
        loop_vars = body(*loop_vars)
        if not isinstance(loop_vars, (list, tuple)):
            loop_vars = (loop_vars, )
        loop_vars = tuple(loop_vars)
        current_iter += 1
    
    return loop_vars if is_tuple else loop_vars[0]


cond = build_op('cond', np_op = _np_cond)

stack   = build_op('stack', np_op = _np_stack, nested = True)
hstack  = build_op('hstack', nested = True)
vstack  = build_op('vstack', nested = True)
unstack = build_op('unstack', np_op = _np_unstack)

while_loop  = build_op('while_loop', np_op = _np_while, is_numpy_check = _check_numpy_while)

globals().update({k : build_op(k) for k in _import_functions(keras.src.ops.core, globals())})

