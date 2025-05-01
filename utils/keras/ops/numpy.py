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

from loggers import timer
from .builder import Ops

@cache
def __getattr__(name):
    if name in _aliases: name = _aliases[name]
    if name not in globals():
        return Ops(name, submodule = 'numpy')
    return globals()[name]

_aliases    = {
    'reverse'   : 'flip',
    
    'absolute'  : 'abs',
    'pow'       : 'power',
    'clip_by_value' : 'clip',
    
    ** {
        'reduce_' + name : name for name in (
            'any', 'all', 'min', 'max', 'mean', 'median', 'std', 'sum', 'prod'
        )
    },
    'reduce_variance'   : 'var',
    
    'gather'    : 'take',
    
    'acos'  : 'arccos',
    'asin'  : 'arcsin',
    'atan'  : 'arctan',
    'acosh' : 'arccosh',
    'asinh' : 'arcsinh',
    'atanh' : 'arctanh',
    'atan2' : 'arctan2',
    
    'diag'  : 'diagonal',
    'diag_part' : 'diagonal'
}

expand_dims = Ops('expand_dims', tensorflow_fn = 'expand_dims')
concat  = concatenate   = Ops('concatenate', nested_arg = 0)

divide_no_nan   = Ops('divide_no_nan', numpy_fn = lambda x, y: np.divide(x, y, where = y != 0))

# required to be defined in `isin`
any = reduce_any    = Ops('any')

def _np_normalize(x, axis = -1, order = 2):
    x_norm = np.linalg.norm(x, ord = order, axis = axis, keepdims = True)
    return np.divide(x, x_norm, where = x_norm != 0.)

def _tf_normalize(x, axis = -1, order = 2):
    import tensorflow as tf
    return tf.math.divide_no_nan(x, norm(x, axis = axis, ord = order, keepdims = True))

norm    = Ops('norm', numpy_fn = 'linalg.norm')
normalize   = l2_normalize  = Ops(
    'normalize', tensorflow_fn = _tf_normalize, numpy_fn = _np_normalize
)

def _np_bincount(inputs, weights = None, minlength = None):
    """
        Count the number of occurrences of each value in `inputs`.
        This is a highly optimized pure `numpy` implementation that supports 2D inputs.
        
        Arguments :
            - x : 1D or 2D `ndarray` of ids
            - weights   : array of weights
                          - 1D : weight for each class (same length as `minlength`)
                          - 2D : weight for each value in `inputs`
            - minlength : minimal length for the output (default to `np.max(inputs)`)
        Return :
            - counts : 1D or 2D array of counts, with last dimension equal to `minlength`
    """
    if minlength is None:   minlength = np.max(inputs)
    
    if weights is None:
        dtype = np.int32
    else:
        if weights.shape != inputs.shape: weights = weights[inputs]
        dtype = weights.dtype
    
    if len(inputs.shape) == 1:
        return np.bincount(inputs, weights = weights, minlength = minlength)
    elif weights is None:
        weights = 1
    
    result = np.zeros((len(inputs), minlength), dtype = dtype)
    np.add.at(result, (np.arange(len(inputs))[:, None], inputs), weights)
    return result

bincount    = Ops('bincount', numpy_fn = _np_bincount)

def _tf_svd(* args, compute_uv = True, ** kwargs):
    import tensorflow as tf
    out = tf.linalg.svd(* args, compute_uv = compute_uv, ** kwargs)
    return (out[1], out[0], out[2]) if compute_uv else out

svd     = Ops('svd', numpy_fn = 'linalg.svd')

def _tf_take_along_axis(arr, indices, axis):
    import tensorflow as tf
    # adapted from https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/numpy_ops/np_array_ops.py#L1602-L1663
    arr     = tf.convert_to_tensor(arr)
    indices = tf.convert_to_tensor(indices)
    
    if axis is None: return tf.gather(tf.reshape(arr, [-1]), indices)
    
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

take_along_axis = Ops('take_along_axis', tensorflow_fn = _tf_take_along_axis)

def _tf_swapaxes(x, axis1, axis2):
    r = len(x.shape)
    if axis1 < 0: axis1 = r + axis1
    if axis2 < 0: axis2 = r + axis2
    
    perm = list(range(0, r))
    perm[axis1], perm[axis2] = axis2, axis1
    return sys.modules['tensorflow'].transpose(x, perm)
    
swapaxes    = Ops('swapaxes', tensorflow_fn = _tf_swapaxes)

def _tf_unique(x, return_inverse = False, return_counts = False):
    import tensorflow as tf
    
    uniques, indexes = tf.unique(x)
    out = (uniques, )
    if return_inverse:  out = out + (indexes, )
    if return_counts:
        counts = tf.ensure_shape(tf.math.bincount(indexes, dtype = tf.int32), uniques.shape)
        out = out + (counts, )
    return out[0] if len(out) == 1 else out

unique  = Ops(
    'unique', tensorflow_fn = _tf_unique, torch_fn = 'unique', jax_fn = 'numpy.unique'
)

def _keras_isin(element, test_elements):
    test_elements = expand_dims(test_elements, list(range(len(element.shape))))
    return any(element[..., None] == test_elements, axis = -1)

isin  = Ops('isin', keras_fn = _keras_isin)

__all__ = list(
    k for k, v in globals().items() if (isinstance(v, Ops)) or (callable(v) and not k.startswith('_'))
)

