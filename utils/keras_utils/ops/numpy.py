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

import numpy as np

from loggers import timer
from .ops_builder import build_op, build_custom_op

""" Arithmetic functions """

globals().update({
    k : build_op(k) for k in ('round', 'ceil', 'floor', 'floor_divide')
})

globals().update({
    k : build_op(k) for k in (
        'add', 'subtract', 'multiply', 'divide', 'true_divide', 'mod',
        'dot', 'matmul', 'tensordot', 'vdot',
        'square', 'sqrt', 'rsqrt'
    )
})
divide_no_nan   = build_op('divide_no_nan', np_op = lambda x, y: np.divide(x, y, where = y != 0))
    

abs = absolute  = build_op('abs')
pow = power     = build_op('power')

def _np_normalize(x, axis = -1, order = 2):
    x_norm = np.linalg.norm(x, ord = order, axis = axis, keepdims = True)
    return np.divide(x, x_norm, where = x_norm != 0.)

norm    = build_op('norm', np_op = 'linalg.norm')
normalize   = l2_normalize  = build_op('normalize', np_op = _np_normalize)

""" Reduction / statistic functions """

min = reduce_min    = build_op('min')
max = reduce_max    = build_op('max')
mean    = reduce_mean   = build_op('mean')
median  = reduce_median = build_op('median')
std = reduce_std    = build_op('std')
var = reduce_variance   = build_op('var')
sum = reduce_sum    = build_op('sum', 'reduce_sum')
prod    = reduce_prod   = build_op('prod', 'reduce_prod')

@timer(debug = True)
def weighted_mean(data, weights, axis = None, keepdims = False):
    if len(shape(weights)) != len(shape(data)):
        weights = expand_dims(
            weights, list(range(len(shape(weights)), len(shape(data))))
        )
    
    return divide_no_nan(
        sum(data * weights, axis = axis, keepdims = keepdims),
        sum(weights, axis = axis, keepdims = keepdims)
    )

globals().update({
    k : build_op(k) for k in (
        'amin', 'amax', 'argmin', 'argmax',
        'average', 'minimum', 'maximum',
        'cumsum', 'cumprod', 'count_nonzero'
    )
})

""" Trigo functions """

globals().update({
    k : build_op(k) for k in (
        'sin', 'sinh', 'cos', 'cosh', 'tan', 'tanh',
        'exp', 'expm1', 'log', 'log1p', 'log2', 'log10', 'logaddexp',
        'cross', 'correlate', 'conj', 'conjugate', 'trace', 'slogdet'
    )
})

acos    = arccos    = build_op('arccos')
asin    = arcsin    = build_op('arcsin')
atan    = arctan    = build_op('arctan')
acosh   = arccosh   = build_op('arccosh')
asinh   = arcsinh   = build_op('arcsinh')
atanh   = arctanh   = build_op('arctanh')
atan2   = arctan2   = build_op('arctan2')

def _tf_svd(* args, compute_uv = True, ** kwargs):
    import tensorflow as tf
    out = tf.linalg.svd(* args, compute_uv = compute_uv, ** kwargs)
    return (out[1], out[0], out[2]) if compute_uv else out

diag    = diagonal    = diag_part = build_op('diagonal')
svd     = build_op('svd', tf_op = _tf_svd, np_op = 'linalg.svd')
globals
""" Logical / boolean operations """

globals().update({
    k : build_op(k) for k in (
        'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal', 'negative', 'nonzero'
    )
})
globals().update({
    k : build_op(k) for k in ('logical_or', 'logical_and', 'logical_not', 'logical_xor')
})
all = reduce_all    = build_op('all')
any = reduce_any    = build_op('any')

isclose     = is_close  = build_op('isclose')
isfinite    = is_finite = build_op('isfinite')
isinf       = is_inf    = build_op('isinf')
isnan       = is_nan    = build_op('isnan')

where   = build_op('where')

""" Shape functions """

globals().update({
    k : build_op(k) for k in (
        'append', 'broadcast_to', 'pad', 'ravel', 'reshape', 'repeat',
        'size', 'squeeze', 'split', 'swapaxes', 'tile', 'moveaxis', 'transpose'
    )
})
expand_dims = build_custom_op(tf_fn = 'expand_dims', name = 'expand_dims')
concat  = concatenate   = build_op('concatenate', nested = True)
flip    = reverse   = build_op('flip')

""" Other functions """

globals().update({
    k : build_op(k) for k in (
        'argpartition', 'argsort', 'bincount', 'copy', 'diff', 'digitize', 'einsum', 'get_item',
        'identity', 'logspace', 'meshgrid', 'quantile', 'sign',
        'nan_to_num', 'outer', 'reciprocal', 'roll', 'select', 'sort', 'correlate', 'vectorize'
    )
})

clip    = clip_by_value = build_op('clip')

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

take    = gather    = build_op('take')
take_along_axis = build_custom_op(tf_fn = _tf_take_along_axis, name = 'take_along_axis')


def _tf_unique(x, return_inverse = False, return_counts = False):
    import tensorflow as tf
    
    uniques, indexes = tf.unique(x)
    out = (uniques, )
    if return_inverse:  out = out + (indexes, )
    if return_counts:
        counts = tf.ensure_shape(tf.math.bincount(indexes, dtype = tf.int32), uniques.shape)
        out = out + (counts, )
    return out[0] if len(out) == 1 else out

unique  = build_custom_op(
    tf_fn = _tf_unique, torch_fn = 'unique', jax_fn = 'numpy.unique', name = 'unique'
)

def _keras_isin(element, test_elements):
    test_elements = expand_dims(test_elements, list(range(len(element.shape))))
    return any(element[..., None] == test_elements, axis = -1)

isin  = build_custom_op(
    keras_fn = _keras_isin, np_fn = 'isin', name = 'isin'
)
