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
import keras.ops as K

from .ops_builder import build_op, build_custom_op

"""
Arithmetic functions
"""

def _tf_svd(* args, compute_uv = True, ** kwargs):
    import tensorflow as tf
    out = tf.linalg.svd(* args, compute_uv = compute_uv, ** kwargs)
    return (out[1], out[0], out[2]) if compute_uv else out

round   = build_op('round')
ceil    = build_op('ceil', 'math.ceil')
floor   = build_op('floor')

add     = build_op('add')
subtract    = build_op('subtract')
multiply    = build_op('multiply')
divide      = build_op('divide')
divide_no_nan   = build_op(
    'divide_no_nan', 'math.divide_no_nan', np_op = lambda x, y: np.divide(x, y, where = y != 0)
)
matmul  = build_op('matmul')
tensordot   = build_op('tensordot')

abs = absolute  = build_op('abs')
pow = build_op(lambda x, y: x ** y, 'pow', np_op = lambda x, y: x ** y, name = 'pow')
square  = build_op('square')

min = reduce_min    = build_op('min', 'reduce_min')
max = reduce_max    = build_op('max', 'reduce_max')
mean    = reduce_mean   = build_op('mean', 'reduce_mean')
std = reduce_std    = build_op('std', 'math.reduce_std')
var = reduce_variance   = build_op('var', 'math.reduce_variance')

cumsum  = build_op('cumsum')
sum = reduce_sum    = build_op('sum', 'reduce_sum')
prod    = reduce_prod   = build_op('prod', 'reduce_prod')
count_nonzero   = build_op('count_nonzero', 'math.count_nonzero')

norm    = build_op('norm', np_op = 'linalg.norm')
sqrt    = build_op('sqrt')
rsqrt   = build_op('rsqrt', 'math.rsqrt')
l2_normalize    = build_op(
    lambda x, axis = -1: divide_no_nan(x, norm(x, axis = axis, keepdims = True)),
    'math.l2_normalize',
    np_op = lambda x, axis = -1: divide_no_nan(x, norm(x, axis = axis, keepdims = True)),
    name = 'l2_normalize'
)

log     = build_op('log', 'math.log')
log10   = build_op('log10')
exp     = build_op('exp')
svd     = build_op('svd', _tf_svd, np_op = 'linalg.svd')
acos    = arccos    = build_op('arccos', 'acos')
asin    = arcsin    = build_op('arcsin', 'asin')
atan    = arctan    = build_op('arctan', 'atan')
acosh   = arccosh   = build_op('arccosh', 'acosh')
asinh   = arcsinh   = build_op('arcsinh', 'asinh')
atanh   = arctanh   = build_op('arctanh', 'atanh')
atan2   = arctan2   = build_op('arctan2', 'atan2')
conj    = build_op('conj', 'math.conj')
diagonal    = diag_part = build_op('diagonal', 'linalg.diag_part')

clip    = clip_by_value = build_op('clip', 'clip_by_value')
minimum = build_op('minimum')
maximum = build_op('maximum')

"""
Logical / boolean operations
"""

all = reduce_all    = build_op('all', 'reduce_all')
any = reduce_any    = build_op('any', 'reduce_any')

equal   = build_op('equal', 'math.equal')
not_equal   = build_op('not_equal', 'math.not_equal')
less    = build_op('less')
greater = build_op('greater')
less_equal  = build_op('less_equal')
greater_equal   = build_op('greater_equal')

logical_or  = build_op('logical_or')
logical_and = build_op('logical_and')
logical_not = build_op('logical_not')

isfinite    = is_finite = build_op('isfinite', 'math.is_finite')
isinf       = is_inf    = build_op('isinf', 'math.is_inf')
isnan       = is_nan    = build_op('isnan', 'math.is_nan')

cond    = build_op('cond', np_op = lambda c, t, f: t if c else f)
where   = build_op('where')


"""
Custom functions that are currently not supported
"""

def _tf_unique(x, return_inverse = False, return_counts = False):
    import tensorflow as tf
    
    uniques, indexes = tf.unique(x)
    out = (uniques, )
    if return_inverse:  out = out + (indexes, )
    if return_counts:
        counts = tf.ensure_shape(tf.reduce_sum(tf.cast(
            tf.range(tf.size(uniques))[:, tf.newaxis] == indexes[tf.newaxis, :], tf.int32
        ), axis = -1), uniques.shape)
        out = out + (counts, )
    return out[0] if len(out) == 1 else out

unique  = build_custom_op(
    tf_fn = _tf_unique, torch_fn = 'unique', jax_fn = 'numpy.unique', name = 'unique'
)
