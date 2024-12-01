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

import keras
import logging
import functools
import numpy as np

from utils import apply_on_batch, dispatch_wrapper, partial
from utils.keras_utils import TensorSpec, ops, graph_compile

logger = logging.getLogger(__name__)

_str_distance_methods   = {}
_similarity_methods     = {}
_distance_methods       = {}

def distance_method_wrapper(fn = None, name = None, is_similarity = False, expand = True):
    def wrapper(fn):
        @functools.wraps(fn)
        def inner(x, y = None, as_matrix = False, ** kwargs):
            if expand and y is not None:
                x, y = _maybe_expand_for_matrix(x, y, as_matrix = as_matrix)
            return fn(x, y, as_matrix = as_matrix, ** kwargs)
        
        key = name if name else fn.__name__.split('_')[0]
        
        distance.dispatch(inner, key)
        
        if is_similarity:   _similarity_methods[key] = inner
        return inner
    
    return wrapper if fn is None else wrapper(fn)

similarity_method_wrapper = partial(distance_method_wrapper, is_similarity = True)

def _propagate_if_matrix(batch_size, kwargs):
    if kwargs.get('as_matrix', False) and batch_size:
        kwargs.update({'batch_size_x' : batch_size, 'batch_size_y' : batch_size})
        batch_size = None
    return batch_size, kwargs

def _raise_if_not_matrix(batch_size, kwargs):
    if batch_size and not kwargs.get('as_matrix', False):
        raise ValueError('When `as_matrix = False`, only the `batch_size` argument can be used')
    return batch_size, kwargs

@apply_on_batch(batched_arg = ('x', 'y'), cond = _propagate_if_matrix)
@apply_on_batch(batched_arg = 'x', cond = _raise_if_not_matrix)
@apply_on_batch(batched_arg = 'y', concat_axis = 1, cond = _raise_if_not_matrix)
@graph_compile
def compiled_distance(x : TensorSpec(), y : TensorSpec(), method, ** kwargs):
    return distance(x, y, method, ** kwargs)

tf_distance = compiled_distance

@apply_on_batch(batched_arg = ('x', 'y'), cond = _propagate_if_matrix)
@apply_on_batch(batched_arg = 'x', cond = _raise_if_not_matrix)
@apply_on_batch(batched_arg = 'y', concat_axis = 1, cond = _raise_if_not_matrix)
@dispatch_wrapper(_distance_methods, 'method')
def distance(x,
             y,
             method,
             *,
             
             mode   = None,
             as_matrix  = False,
             force_distance = None,
             
             ** kwargs
            ):
    """
        Computes the distance between `x` and `y` with `method` function
        
        Arguments : 
            - x : 1D vector or 2D matrix
            - y : 2D matrix of points
            - method : the name of the distance function to use
            - as_matrix : whether to compute matrix or point-wise distance (see notes)
            - max_matrix_size : maximum number of values in the matrix distance computation
            - kwargs : kwargs to pass to the distance function
        Return : 
            - distance  : the result of `method` function applied to `x` and `y`
        
        Note : 
        If `as_matrix is True` : return a matrix such that `matrix[i, j]` is the distance between `x[i]` and `y[j]`
        Else : `matrix[i]` is the distance between `x[i]` and `y[i]`
        
        This distance can be a scalar (euclidian, manhattan, dot-product) or a vector of element-wise distance (l1, l2)
        
        Important note : this function returns a **distance** score : the lower the score, the more similar they are ! If the `method` is a similarity metric (such as dot-product), the function returns the inverse (- distances) to keep this property.
        You can avoid this by setting `force_distance = False`
    """
    if force_distance is True: mode = 'distance'
    
    distance_fn = method if callable(method) else _distance_methods.get(method, None)
    if distance_fn is None:
        raise ValueError("Distance method is not callable or does not exist !\n  Accepted : {}\n  Got : {}".format(
            list(_distance_methods.keys()), method
        ))
    
    if method in _str_distance_methods:
        return _str_distance_methods[method](x, y, as_matrix = as_matrix, ** kwargs)

    if logger.isEnabledFor(logging.DEBUG) and ops.executing_eagerly():
        logger.debug('Calling {} on matrices with shapes {} and {}'.format(
            method, tuple(x.shape), tuple(y.shape)
        ))
    
    if len(ops.shape(x)) == 1: x = ops.expand_dims(x, axis = 0)
    if len(ops.shape(y)) == 1: y = ops.expand_dims(y, axis = 0)

    result = distance_fn(x, y, as_matrix = as_matrix, ** kwargs)

    if logger.isEnabledFor(logging.DEBUG) and ops.executing_eagerly():
        logger.debug('Result shape : {}'.format(tuple(result.shape)))
    
    if mode == 'distance' and method in _similarity_methods:         result = - result
    elif mode == 'similarity' and method not in _similarity_methods: result = - result
    return result

@similarity_method_wrapper(expand = False)
def cosine_similarity(x, y, *, as_matrix = False, ** kwargs):
    return dot_product(
        ops.l2_normalize(x, axis = -1),
        ops.l2_normalize(y, axis = -1),
        as_matrix   = as_matrix,
        ** kwargs
    )

@similarity_method_wrapper(name = 'dp', expand = False)
def dot_product(x, y, *, as_matrix = False, ** kwargs):
    if as_matrix: return ops.einsum('...ik, ...jk -> ...ij', x, y)
    return ops.einsum('...j, ...j -> ...', x, y)

@distance_method_wrapper
def l1_distance(x, y, ** kwargs):
    return ops.abs(ops.subtract(x, y))

@distance_method_wrapper
def l2_distance(x, y, ** kwargs):
    return ops.square(ops.subtract(x, y))

@distance_method_wrapper
def manhattan_distance(x, y, ** kwargs):
    return ops.sum(ops.abs(ops.subtract(x, y)), axis = -1)

@distance_method_wrapper(expand = False)
def euclidian_distance(x, y, *, fast = True, as_matrix = False, ** kwargs):
    if fast:
        xx = ops.einsum('...i, ...i -> ...', x, x)
        yy = ops.einsum('...i, ...i -> ...', y, y)
        xy = dot_product(x, y, as_matrix = as_matrix)
        if as_matrix: xx, yy = ops.expand_dims(xx, -1), ops.expand_dims(yy, -2)
        return ops.sqrt(xx - 2 * xy + yy)
    
    x, y = _maybe_expand_for_matrix(x, y, as_matrix = as_matrix)
    return ops.sqrt(ops.sum(ops.square(ops.subtract(x, y)), axis = -1))

@similarity_method_wrapper
def dice_coeff(x, y, ** kwargs):
    inter = ops.sum(x * y)
    union = ops.sum(x) + ops.sum(y)
    return ops.divide_no_nan(2 * inter, union)

def _maybe_expand_for_matrix(x, y, *, as_matrix = False):
    """
        Expands `x` and `y` such that a mathematical operation between them will compute all pairs, resulting in a matrix-like operation
        
        Example :
            `x.shape == (A, 1, d)` * `b.shape == (1, B, d)` -> `res.shape == (A, B, d)`
    """
    if as_matrix and len(ops.shape(x)) == len(ops.shape(y)):    y = y[..., None, :, :]
    if len(ops.shape(x)) == len(ops.shape(y)) - 1:              x = x[..., None, :]

    return x, y
