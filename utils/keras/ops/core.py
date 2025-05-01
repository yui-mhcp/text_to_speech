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

import sys
import numpy as np

from functools import cache

from .. import timer
from .builder import Ops, fast_is_not_tensor
from .execution_contexts import is_tensorflow_graph, is_tensorflow_backend, is_torch_backend

@cache
def __getattr__(name):
    if name in _aliases: name = _aliases[name]
    if name not in globals():
        return Ops(name, submodule = 'core', disable_np = name in _creation_functions)
    return globals()[name]

class TensorflowNotAvailable(Exception):
    def __init__(self):
        super().__init__('Tensorflow is not available')

""" `Tensor` creation functions """

_creation_functions = {
    'empty', 'zeros', 'ones', 'full', 'zeros_like', 'ones_like', 'full_like',
    'arange', 'linspace', 'tri', 'trii', 'triu'
}
array   = constant  = Ops('array', disable_np = True)
eye     = Ops('eye', tensorflow_fn = 'eye', disable_np = True)

_aliases    = {
    'constant'  : 'array',
    'fill'      : 'full',
    'fill_like' : 'full_like',
    'range'     : 'arange'
}
""" Shape functions """

@timer(debug = True)
def shape(x):
    """ Returns `x.shape`, which is equivalent, but faster than `K.shape` """
    if not hasattr(x, 'shape'):
        return np.shape(x)
    elif isinstance(x, np.ndarray) or not is_tensorflow_graph():
        return tuple(x.shape)
    else:     
        return sys.modules['tensorflow'].shape(x)

@timer(debug = True)
def rank(x):
    if not hasattr(x, 'shape') or isinstance(x, np.ndarray): return np.ndim(x)
    else: return len(shape(x))

ndim    = rank

ensure_shape    = Ops(
    'ensure_shape',
    keras_fn    = lambda x, shape: x,
    tensorflow_fn   = 'ensure_shape'
)

""" Convertion functions (`Tensor` <--> `ndarray`) """

@timer(debug = True)
def is_array(x):
    return (hasattr(x, 'shape')) and (isinstance(x, np.ndarray) or is_tensor(x))

@timer(debug = True)
def is_tensor(x):
    if fast_is_not_tensor(x):   return False
    elif 'keras' not in sys.modules:    return False
    elif is_tensorflow_backend() or not is_tensorflow_graph():
        return sys.modules['keras'].ops.is_tensor(x)
    else:
        return sys.modules['tensorflow'].is_tensor(x)

@timer(debug = True)
def is_torch_tensor(x):
    if 'torch' not in sys.modules: return False
    import torch
    return torch.is_tensor(x)


@timer(debug = True)
def convert_to_numpy(x, dtype = None, copy = False):
    if dtype == 'float' and hasattr(x, 'dtype') and is_float(x):
        dtype = None
    elif dtype:
        dtype = dtype_to_str(dtype)
    
    if not fast_is_not_tensor(x) and is_tensorflow_graph():
        return convert_to_tf_tensor(x, dtype)
    elif hasattr(x, 'detach'):
        array = np.asarray(x.detach().cpu().numpy(), dtype = dtype)
    elif hasattr(x, 'numpy'):
        array = np.asarray(x.numpy(), dtype = dtype)
    else:
        array = np.asarray(x, dtype = dtype)
    if copy and x is array: array = array.copy()
    return array

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
    if not is_tensor(x):
        return array(x, dtype_to_str(dtype) if dtype else get_convertion_dtype(x))
    elif dtype is None or (dtype == 'float' and is_float(x)) or (dtype == 'int' and is_int(x)):
        return x
    else:
        return cast(x, dtype)

@timer(debug = True)
def convert_to_tf_tensor(x, dtype = None):
    if is_tensorflow_backend(): return convert_to_tensor(x, dtype)
    
    try:
        import tensorflow as tf
    except ImportError:
        raise TensorflowNotAvailable()
    
    if not tf.is_tensor(x):
        return tf.convert_to_tensor(convert_to_numpy(x, dtype))
    elif dtype is None or (dtype == 'float' and is_float(x)) or (dtype == 'int' and is_int(x)):
        return x
    else:
        return tf.cast(x, dtype)

@timer(debug = True)
def convert_to_torch_tensor(x, dtype = None):
    if is_torch_backend(): return convert_to_tensor(x, dtype)
    
    import keras.src.backend.torch as torch
    
    if not torch.is_tensor(x):
        return torch.convert_to_tensor(convert_to_numpy(x, dtype))
    elif dtype is None or (dtype == 'float' and is_float(x)) or (dtype == 'int' and is_int(x)):
        return x
    else:
        return torch.cast(x, dtype)


""" `dtype` functions """

@timer(debug = True)
def get_convertion_dtype(x):
    while isinstance(x, (list, tuple)):
        if len(x) == 0: return 'float32'
        x = x[0]
    
    if is_tensor(x):            return x.dtype
    elif isinstance(x, str):    return 'string'
    elif is_bool(x):            return 'bool'
    elif is_float(x):           return 'float32'
    elif 'uint' in dtype_to_str(getattr(x, 'dtype', '')):   return 'uint8'
    elif is_int(x):             return 'int32'
    else: raise ValueError('Unknown data type : {}\n{}'.format(type(x), x))

@timer(debug = True)
@cache
def dtype_to_str(dtype):
    if isinstance(dtype, str):
        if dtype != 'float':    return dtype
        elif 'keras' not in sys.modules:    return 'float32'
        else:   return sys.modules['keras'].backend.floatx()
    elif hasattr(dtype, 'name'):
        return dtype.name
    elif 'keras' in sys.modules:
        return sys.modules['keras'].backend.standardize_dtype(dtype)
    else:
        raise ValueError('Unknown dtype : {}'.format(dtype))


def is_int(x):  return isinstance(x, int) or 'int' in dtype_to_str(getattr(x, 'dtype', ''))
def is_bool(x): return isinstance(x, bool) or 'bool' == dtype_to_str(getattr(x, 'dtype', ''))
def is_float(x):    return isinstance(x, float) or 'float' in dtype_to_str(getattr(x, 'dtype', ''))
def is_numeric(x):  return is_float(x) or is_int(x)
def is_string(x):   return isinstance(x, str) or 'str' in dtype_to_str(getattr(x, 'dtype', ''))

@timer(debug = True)
def cast(x, dtype):
    dtype = dtype_to_str(dtype)
    if not is_array(x):                     return array(x, dtype)
    elif dtype == dtype_to_str(x.dtype):    return x
    else:                                   return cast_ops(x, dtype)

def _np_cast(x, dtype):
    return x.astype(dtype) if isinstance(x, np.ndarray) else np.array(x, dtype = dtype)

cast_ops = Ops('cast', numpy_fn = _np_cast)

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

complex = Ops('complex', jax_fn = 'lax.complex')

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

slice   = dynamic_slice = Ops('slice', numpy_fn = _np_slice)
slice_update    = update_slice  = dynamic_update_slice  = Ops(
    'slice_update', numpy_fn = _np_slice_update
)

scatter = Ops('scatter', numpy_fn = _np_scatter)
scatter_update  = scatter_nd_update = Ops('scatter_update', numpy_fn = _np_scatter_update)

""" Other `core` functions """

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
    
    _slices = tuple(_python_slice(0, None) for _ in _python_range(axis))
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


cond = Ops('cond', numpy_fn = _np_cond)

stack   = Ops('stack', numpy_fn = _np_stack, nested_arg = 0)
hstack  = Ops('hstack', nested_arg = 0)
vstack  = Ops('vstack', nested_arg = 0)
unstack = Ops('unstack', numpy_fn = _np_unstack)

while_loop  = Ops('while_loop', numpy_fn = _np_while, nested_arg = 2)

__all__ = list(
    k for k, v in globals().items() if (isinstance(v, Ops)) or (callable(v) and not k.startswith('_'))
)
