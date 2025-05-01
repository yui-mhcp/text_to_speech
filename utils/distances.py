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

import logging

from functools import partial

from .wrappers import dispatch_wrapper
from .keras import TensorSpec, ops, graph_compile

logger = logging.getLogger(__name__)

_distance_methods   = {}
_similarity_methods     = set()

def distance_method_wrapper(fn = None, name = None, is_similarity = False):
    def wrapper(fn):
        key = name or fn.__name__.replace('_distance', '')
        distance.dispatch(fn, key)
        
        if is_similarity:   _similarity_methods.add(key)
        return fn
    
    if isinstance(fn, (str, tuple)): fn, name = None, fn
    
    return wrapper if fn is None else wrapper(fn)

similarity_method_wrapper = partial(distance_method_wrapper, is_similarity = True)

@dispatch_wrapper(_distance_methods, 'method')
def distance(x, y, method, *, mode = None, as_matrix = False):
    """
        Computes the distance between `x` and `y` with `method` function
        
        Arguments : 
            - x : 1D vector or 2D matrix
            - y : 2D matrix of points
            - method    : the name of the distance function to use
            - mode      : either 'distance' or 'similarity' to force distance (resp. similarity) property
            - as_matrix : whether to compute matrix or point-wise distance (see notes)
        Return : 
            - distance  : the result of `method` function applied to `x` and `y`
        
        Note : 
        If `as_matrix is True` : return a matrix such that `matrix[i, j]` is the distance between `x[i]` and `y[j]`
        Else : `matrix[i]` is the distance between `x[i]` and `y[i]`
        
        This distance can be a scalar (euclidian, manhattan, dot-product) or a vector of element-wise distance (l1, l2)
        You can avoid this by setting `force_distance = False`
    """
    if method not in _distance_methods:
        raise ValueError("Unknown distance method !\n  Accepted : {}\n  Got : {}".format(
            list(_distance_methods.keys()), method
        ))

    if logger.isEnabledFor(logging.DEBUG) and ops.executing_eagerly():
        logger.debug('Calling {} on matrices with shapes {} and {}'.format(
            method, tuple(x.shape), tuple(y.shape)
        ))
    
    if len(ops.shape(x)) == 1: x = ops.expand_dims(x, axis = 0)
    if len(ops.shape(y)) == 1: y = ops.expand_dims(y, axis = 0)

    result = _distance_methods[method](x, y, as_matrix = as_matrix)

    if logger.isEnabledFor(logging.DEBUG) and ops.executing_eagerly():
        logger.debug('Result shape : {}'.format(tuple(result.shape)))
    
    if (
        (mode == 'distance' and method in _similarity_methods)
        or (mode == 'similarity' and method not in _similarity_methods)):
        result = - result
    
    return result

@similarity_method_wrapper('cosine')
def cosine_similarity(x, y, *, as_matrix = False):
    return dot_product(
        ops.l2_normalize(x, axis = -1),
        ops.l2_normalize(y, axis = -1),
        as_matrix   = as_matrix
    )

@similarity_method_wrapper(name = 'dp')
def dot_product(x, y, *, as_matrix = False):
    if as_matrix: return ops.einsum('...ik, ...jk -> ...ij', x, y)
    return ops.einsum('...j, ...j -> ...', x, y)

@distance_method_wrapper
def l1_distance(x, y, as_matrix = False):
    x, y = _maybe_expand_for_matrix(x, y, as_matrix)
    return ops.abs(ops.subtract(x, y))

@distance_method_wrapper
def l2_distance(x, y, as_matrix = False):
    x, y = _maybe_expand_for_matrix(x, y, as_matrix)
    return ops.square(ops.subtract(x, y))

@distance_method_wrapper
def manhattan_distance(x, y, as_matrix = False):
    x, y = _maybe_expand_for_matrix(x, y, as_matrix)
    return ops.sum(ops.abs(ops.subtract(x, y)), axis = -1)

@distance_method_wrapper
def euclidian_distance(x, y, *, fast = True, as_matrix = False):
    if fast:
        xx = ops.einsum('...i, ...i -> ...', x, x)
        yy = ops.einsum('...i, ...i -> ...', y, y)
        xy = dot_product(x, y, as_matrix = as_matrix)
        if as_matrix: xx, yy = ops.expand_dims(xx, -1), ops.expand_dims(yy, -2)
        return ops.sqrt(xx - 2 * xy + yy)
    
    x, y = _maybe_expand_for_matrix(x, y, as_matrix = as_matrix)
    return ops.sqrt(ops.sum(ops.square(ops.subtract(x, y)), axis = -1))

@similarity_method_wrapper
def dice_coeff(x, y, as_matrix = False):
    if as_matrix: raise NotImplementedError()
    
    inter = ops.sum(x * y)
    union = ops.sum(x) + ops.sum(y)
    return ops.divide_no_nan(2 * inter, union)


@graph_compile
def knn(query   : TensorSpec(shape = (None, None), dtype = 'float') = None,
        embeddings  : TensorSpec(shape = (None, None), dtype = 'float') = None,
        distance_matrix : TensorSpec(shape = (None, None), dtype = 'float') = None,
        distance_metric = None,
        
        k   : TensorSpec(shape = (), dtype = 'int32') = 5,
        ids : TensorSpec(shape = (None, )) = None,
        
        weighted    : TensorSpec(shape = (), dtype = 'bool') = False,
        
        return_scores   = False
       ):
    """
        Compute the k-nn decision procedure for a given x based on a list of labelled embeddings
        
        Arguments :
            - query : the query point(s), 1-D or 2-D `Tensor`
            - embeddings    : the points to use
            - distance_matrix   : the already computed distance matrix (2-D) between `query` and `embeddings`
            
            - ids   : the ids for `embeddings` (**must be numeric values**)
            - k     : the `k` hyperparameter in the K-NN
            - distance_metric   : the metric to use to compute distance (irrelevant if passing `distance_matrix`)
            
            - return_index  : whether to return the nearest indexes
            - weighted      : whether to use the weighted KNN algorithm
            
            - kwargs    : passed to `distance`
        Return :
            If `ids` is not None    : 1-D `Tensor` with the nearest ids
            elif `return_index`     : 1-D `Tensor`, the nearest embeddings' indexes
            else                    : 2-D `Tensor`, the k-nearest embeddings
    """
    assert distance_matrix is not None or (
        query is not None and embeddings is not None and distance_metric
    )
    
    if distance_matrix is None:
        distance_matrix = distance(
            query,
            embeddings,
            distance_metric,

            as_matrix   = True,
            mode    = 'distance'
        )
    # the `- distance` is required as `top_k` takes the highest values
    # while we want the nearest points, i.e., those with the lowest distances
    k_nearest_dists, k_nearest_idx = ops.top_k(- distance_matrix, k)
    
    if ids is None:
        if distance_metric not in _similarity_methods: k_nearest_dists = -k_nearest_dists
        return (k_nearest_dists, k_nearest_idx) if return_scores else k_nearest_idx
    
    unique_ids, pos_ids = ops.unique(ids, return_inverse = True)
    
    nearest_ids = ops.take(pos_ids, k_nearest_idx, axis = 0)

    # shape == [len(query), k]
    weights = ops.cond(
        weighted,
        lambda: 1. / ops.maximum(-k_nearest_dists, 1e-9),
        lambda: ops.ones_like(k_nearest_dists)
    )
    # shape == [len(points), len(unique_ids), k]
    mask    = ops.arange(ops.shape(unique_ids)[0])[None, :, None] == nearest_ids[:, None, :]
    # shape = [len(query), len(unique_ids)]
    scores  = ops.sum(weights[:, None, :] * ops.cast(mask, weights.dtype), axis = -1)
    #scores = K.bincount(
    #    nearest_ids, weights = weights, minlength = K.size(unique_ids)
    #)

    if return_scores: return unique_ids, scores
    return ops.take(unique_ids, ops.argmax(scores, axis = -1))


def _maybe_expand_for_matrix(x, y, as_matrix = False):
    """
        Expands `x` and `y` such that a mathematical operation between them will compute all pairs, resulting in a matrix-like operation
        
        Example :
            `x.shape == (A, 1, d)` * `b.shape == (1, B, d)` -> `res.shape == (A, B, d)`
    """
    if as_matrix and len(ops.shape(x)) == len(ops.shape(y)):    y = y[..., None, :, :]
    if len(ops.shape(x)) == len(ops.shape(y)) - 1:              x = x[..., None, :]

    return x, y
