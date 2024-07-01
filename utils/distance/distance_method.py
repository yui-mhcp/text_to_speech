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

import logging
import functools
import numpy as np

from utils.keras_utils import TensorSpec, graph_compile, ops
from utils.wrapper_utils import dispatch_wrapper, partial

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

@graph_compile
def tf_distance(x : TensorSpec(),
                y : TensorSpec(),
                method,
                as_matrix   = True,
                max_slice   = None,
                max_slice_x = None,
                max_slice_y = None,
                force_distance  = False,
                ** kwargs
               ):
    return distance(
        x,
        y,
        method,
        as_matrix   = as_matrix,
        max_slice   = max_slice,
        max_slice_x = max_slice_x,
        max_slice_y = max_slice_y,
        force_distance  = force_distance,
        ** kwargs
    )

@dispatch_wrapper(_distance_methods, 'method')
def distance(x,
             y,
             method,
             force_distance = True,

             as_matrix  = False,
             max_slice  = None,
             max_slice_x    = None,
             max_slice_y    = None,
             
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
    
    use_numpy   = isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    
    if not use_numpy:
        if y is not None: ops.convert_to_tensor(y)
        x = ops.convert_to_tensor(x)
    
    if len(ops.shape(x)) == 1: x = ops.expand_dims(x, axis = 0)
    if len(ops.shape(y)) == 1: y = ops.expand_dims(y, axis = 0)

    if max_slice is not None:
        if max_slice_x is None: max_slice_x = max_slice
        if max_slice_y is None: max_slice_y = max_slice
    
    if (max_slice_x is not None and 0 < max_slice_x < len(x)) or \
    (max_slice_y is not None and 0 < max_slice_y < len(y)):
        if max_slice_x is None: max_slice_x = len(x)
        if max_slice_y is None: max_slice_y = len(y)
        
        if not as_matrix:
            if len(x) == 1 and len(y) > 1:
                x, y, max_slice_x = y, x, max_slice_y
            
            if use_numpy:
                result = np.empty((len(x), ), dtype = np.float32)
                for i in range(0, len(x), max_slice):
                    result[i : i + max_slice] = distance_fn(
                        x[i : i + max_slice], y[i : i + max_slice] if len(y) > 1 else y, ** kwargs
                    )
            else:
                def body(i, res):
                    res = ops.slice_update(
                        res,
                        ops.constant([1], dtype = 'int32') * i,
                        distance_fn(
                            x[i : i + max_slice],
                            y[i : i + max_slice] if len(y) > 1 else y,
                            as_matrix   = False,
                            ** kwargs
                        )
                    )
                    return (i + max_slice, res)
                
                result = ops.while_loop(
                    lambda i, state: i < len(x),
                    body,
                    (0, ops.empty((len(x), ), dtype = 'float32'))
                )[1]
        else:
            if use_numpy:
                result = np.empty((len(x), len(y)), dtype = np.float32)
                for i in range(0, len(x), max_slice_x):
                    for j in range(0, len(y), max_slice_y):
                        result[i : i + max_slice_x, j : j + max_slice_y] = distance_fn(
                            x[i : i + max_slice_x], y[j : j + max_slice_y], as_matrix = True, ** kwargs
                        )
            else:
                def update_body(yi, xi, res):
                    res = ops.slice_update(
                        res,
                        ops.constant([1, 0], dtype = 'int32') * xi + ops.constant([0, 1], dtype = 'int32') * yi,
                        distance_fn(
                            x[xi : xi + max_slice_x],
                            y[yi : yi + max_slice_y],
                            as_matrix   = True,
                            ** kwargs
                        )
                    )
                    return (yi + max_slice_y, xi, res)
                
                def slice_y_body(xi, res):
                    return (
                        xi + max_slice_x,
                        ops.while_loop(
                            lambda yi, xi, s: yi < len(y), update_body, (0, xi, res)
                        )[2]
                    )
                
                result = ops.while_loop(
                    lambda i, s: i < len(x),
                    slice_y_body,
                    (0, ops.empty((len(x), len(y)), dtype = 'float32'))
                )[1]
    else:
        result = distance_fn(x, y, as_matrix = as_matrix, ** kwargs)

    if logger.isEnabledFor(logging.DEBUG) and ops.executing_eagerly():
        logger.debug('Result shape : {}'.format(tuple(result.shape)))
    
    return result if not force_distance or method not in _similarity_methods else 1. - result

@similarity_method_wrapper(expand = False)
def cosine_similarity(x, y, *, as_matrix = False, ** kwargs):
    return dot_product(
        ops.l2_normalize(x, axis = -1),
        ops.l2_normalize(x, axis = -1),
        as_matrix   = as_matrix,
        ** kwargs
    )

@similarity_method_wrapper(name = 'dp')
def dot_product(x, y, ** kwargs):
    return ops.sum(x * y, axis = -1)

@distance_method_wrapper
def l1_distance(x, y, ** kwargs):
    return ops.abs(x - y)

@distance_method_wrapper
def l2_distance(x, y, ** kwargs):
    return ops.square(x - y)

@distance_method_wrapper
def manhattan_distance(x, y, ** kwargs):
    return ops.sum(ops.abs(x - y), axis = -1)

@distance_method_wrapper
def euclidian_distance(x, y, ** kwargs):
    return ops.sqrt(ops.sum(ops.square(x - y), axis = -1))

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
