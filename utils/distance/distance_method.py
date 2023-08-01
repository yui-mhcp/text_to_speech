
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
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
import functools
import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

MAX_MATRIX_SIZE = 512 * 1024 * 1024

def _maybe_expand_for_matrix(x, y, as_matrix = False):
    if isinstance(x, tf.Tensor):
        rank_fn, expand_fn = lambda t: len(tf.shape(t)), tf.expand_dims
    else:
        rank_fn, expand_fn = lambda t: t.ndim, np.expand_dims
    
    if as_matrix and rank_fn(x) == rank_fn(y):
        y = expand_fn(y, axis = -3)
    
    if rank_fn(x) == rank_fn(y) - 1:
        x = expand_fn(x, axis = -2)

    return x, y

@tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
def tf_distance(x : tf.Tensor,
                y : tf.Tensor,
                method,
                as_matrix   = True,
                force_distance  = False,
                max_matrix_size = -1,
                ** kwargs
               ):
    return distance(
        x, y, method, force_distance = force_distance, as_matrix = as_matrix,
        max_matrix_size = max_matrix_size, ** kwargs
    )

def distance(x,
             y,
             method,
             force_distance = True,

             as_matrix  = False,
             max_matrix_size    = MAX_MATRIX_SIZE,
             
             ** kwargs
            ):
    """
        Compute distance between `x` and `y` with `method` function
        
        Arguments : 
            - x : 1D vector or 2D matrix
            - y : 2D matrix of points
            - method : string (the name of the method)
            - as_matrix : whether to compute matrix or point-wise distance (see notes)
            - max_matrix_size : maximum number of values in the matrix distance computation
            - kwargs : kwargs to pass to the distance function
        Return : 
            - `method` function applied to `x` and `y`
        
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
    
    if method in _str_distance_method:
        return _str_distance_method[method](x, y, ** kwargs)

    if tf.executing_eagerly() and logger.isEnabledFor(logging.DEBUG):
        logger.debug('Calling {} on matrices with shapes {} and {}'.format(
            method, tuple(x.shape), tuple(y.shape)
        ))
    
    if isinstance(x, tf.Tensor) or isinstance(y, tf.Tensor):
        rank_fn, expand_fn = lambda t: len(tf.shape(t)), tf.expand_dims
        x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    else:
        rank_fn, expand_fn = lambda t: t.ndim, np.expand_dims
        x, y = x.astype(np.float32), y.astype(np.float32)
        
    if rank_fn(x) == 1: x = expand_fn(x, axis = 0)
    if rank_fn(y) == 1: y = expand_fn(y, axis = 0)

    max_x = -1
    if max_matrix_size > 0 and isinstance(x, tf.Tensor):
        if as_matrix:
            max_x = tf.minimum(tf.shape(x)[0], tf.cast(tf.math.ceil(
                max_matrix_size / (tf.shape(x)[-1] * tf.shape(y)[0])
            ), tf.int32))
        else:
            max_x = tf.minimum(max_matrix_size // tf.square(tf.shape(x)[-1]) + 1, tf.shape(x)[0])
        
    
    if max_x != -1 and max_x < tf.shape(x)[0]:
        n_slices    = tf.cast(tf.math.ceil(tf.shape(x)[0] / max_x), tf.int32)
        elem_shape  = (None, ) if not as_matrix else (None, y.shape[0])
        distances   = tf.TensorArray(
            dtype = tf.float32, size = n_slices, element_shape = elem_shape
        )
        if not as_matrix:
            for i in tf.range(n_slices):
                distances = distances.write(
                    i, tf.reshape(distance_fn(
                        x[i * max_x : (i + 1) * max_x], y[i * max_x : (i + 1) * max_x],
                        as_matrix = False, ** kwargs
                    ), [-1])
                )
        else:
            for i in tf.range(n_slices):
                distances = distances.write(
                    i, distance_fn(x[i * max_x : (i + 1) * max_x], y, as_matrix = True, ** kwargs)
                )
        distances = distances.concat()
    else:
        distances = distance_fn(x, y, as_matrix = as_matrix, ** kwargs)
    
    if tf.executing_eagerly() and logger.isEnabledFor(logging.DEBUG):
        logger.debug('Result shape : {}'.format(tuple(distances.shape)))
    
    return distances if not force_distance or method not in _similarity_methods else 1. - distances

def cosine_similarity(x, y, as_matrix = False, ** kwargs):
    return dot_product(
        tf.math.l2_normalize(x, axis = -1),
        tf.math.l2_normalize(y, axis = -1),
        as_matrix = as_matrix, ** kwargs
    )

def dot_product(x, y, as_matrix = False, ** kwargs):
    x, y = _maybe_expand_for_matrix(x, y, as_matrix = as_matrix)
    return tf.reduce_sum(x * y, axis = -1)

def l1_distance(x, y, as_matrix = False, ** kwargs):
    x, y = _maybe_expand_for_matrix(x, y, as_matrix = as_matrix)
    return tf.abs(x - y)

def l2_distance(x, y, as_matrix = False, ** kwargs):
    x, y = _maybe_expand_for_matrix(x, y, as_matrix = as_matrix)
    return tf.square(x, y)

def manhattan_distance(x, y, as_matrix = False, ** kwargs):
    x, y = _maybe_expand_for_matrix(x, y, as_matrix = as_matrix)
    return tf.reduce_sum(tf.abs(x - y), axis = -1)

def euclidian_distance(x, y, as_matrix = False, ** kwargs):
    x, y = _maybe_expand_for_matrix(x, y, as_matrix = as_matrix)
    return tf.math.sqrt(tf.reduce_sum(tf.square(x - y), axis = -1))

def bbox_metric(x, y, box_mode, metric, as_matrix = False, ** kwargs):
    from utils.image.box_utils import BoxFormat, convert_box_format
    
    if isinstance(x, tf.Tensor):
        expand_fn, minimum_fn, maximum_fn = tf.expand_dims, tf.minimum, tf.maximum
        divide_no_nan = tf.math.divide_no_nan
    elif isinstance(x, np.ndarray):
        expand_fn, minimum_fn, maximum_fn = np.expand_dims, np.minimum, np.maximum
        divide_no_nan   = lambda num, den: np.divide(num, den, where = den != 0)
    else:
        minimum_fn, maximum_fn, divide_no_nan = min, max, lambda a, b: 0. if b == 0. else a / b
    
    shape_fn = tf.shape if not tf.executing_eagerly() else lambda t: t.shape
    
    if isinstance(x, (np.ndarray, tf.Tensor)):
        if box_mode == BoxFormat.POLY:
            if len(shape_fn(x)) == 2: x = expand_fn(x, axis = 0)
            if len(shape_fn(y)) == 2: y = expand_fn(x, axis = 0)

        if as_matrix:
            x = expand_fn(x, axis = -3 if box_mode == BoxFormat.POLY else -2)
            y = expand_fn(y, axis = -3 if box_mode == BoxFormat.POLY else -2)

    
    xmin_1, ymin_1, xmax_1, ymax_1 = convert_box_format(
        x, box_mode = box_mode, output_format = BoxFormat.CORNERS, as_list = True
    )
    xmin_2, ymin_2, xmax_2, ymax_2 = convert_box_format(
        y, box_mode = box_mode, output_format = BoxFormat.CORNERS, as_list = True
    )
    
    if as_matrix:
        xmin_1, xmin_2 = _maybe_expand_for_matrix(xmin_1, xmin_2, as_matrix = True)
        ymin_1, ymin_2 = _maybe_expand_for_matrix(ymin_1, ymin_2, as_matrix = True)
        xmax_1, xmax_2 = _maybe_expand_for_matrix(xmax_1, xmax_2, as_matrix = True)
        ymax_1, ymax_2 = _maybe_expand_for_matrix(ymax_1, ymax_2, as_matrix = True)

    areas_1 = (ymax_1 - ymin_1) * (xmax_1 - xmin_1)
    areas_2 = (ymax_2 - ymin_2) * (xmax_2 - xmin_2)
    
    xmin, ymin = maximum_fn(xmin_1, xmin_2), maximum_fn(ymin_1, ymin_2)
    xmax, ymax = minimum_fn(xmax_1, xmax_2), minimum_fn(ymax_1, ymax_2)

    inter_w, inter_h = maximum_fn(0., xmax - xmin), maximum_fn(0., ymax - ymin)
    
    inter = inter_w * inter_h
    
    if metric == 'iou':
        denom = areas_1 + areas_2 - inter
    elif metric == 'intersect':
        if not as_matrix:
            denom = areas_1
        else:
            arange  = np.arange(len(areas_1))
            denom   = areas_1 * (arange[np.newaxis] > arange[:, np.newaxis]) + areas_2 * (arange[np.newaxis] < arange[:, np.newaxis])

    result = divide_no_nan(inter, denom)
    return result if not as_matrix else result[..., 0]

iou = functools.partial(bbox_metric, metric = 'iou')
intersect   = functools.partial(bbox_metric, metric = 'intersect')

def edit_distance(hypothesis,
                  truth,
                  partial   = False,
                  deletion_cost     = {},
                  insertion_cost    = {}, 
                  replacement_cost  = {},
                  
                  default_del_cost  = 1,
                  default_insert_cost   = 1,
                  default_replace_cost  = 1,
                  
                  normalize     = True,
                  return_matrix = False,
                  verbose   = False,
                  ** kwargs
                 ):
    """
        Compute a weighted Levenstein distance
        
        Arguments :
            - hypothesis    : the predicted value   (iterable)
            - truth         : the true value        (iterable)
            - partial       : whether to make partial alignment or not
            - insertion_cost    : weights to insert a new symbol
            - replacement_cost  : weights to replace a symbol (a --> b) but 
            is not in both sens (a --> b != b --> a) so you have to specify weights in both sens
            - normalize     : whether to normalize on truth length or not
            - return_matrix : whether to return the matrix or not
            - verbose       : whether to show costs for path or not
        Return :
            - distance if not return_matrix else (distance, matrix)
                - distance  : scalar, the Levenstein distance between `hypothesis` and truth `truth`
                - matrix    : np.ndarray of shape (N, M) where N is the length of truth and M the length of hypothesis. 
        
        Note : if `partial` is True, the distance is the minimal distance
        Note 2 : `distance` (without normalization) corresponds to the "number of errors" between `hypothesis` and `truth`. It means that the start of the best alignment (if partial) is `np.argmin(matrix[-1, 1:]) - len(truth) - distance`
    """
    matrix = np.zeros((len(hypothesis) + 1, len(truth) + 1))
    # Deletion cost
    deletion_costs = np.array([0] + [deletion_cost.get(h, default_del_cost) for h in hypothesis])
    matrix[:, 0] = np.cumsum(deletion_costs)
    # Insertion cost
    if not partial:
        matrix[0, :] = np.cumsum([0] + [insertion_cost.get(t, default_insert_cost) for t in truth])

    truth_array = truth if not isinstance(truth, str) else np.array(list(truth))
    for i in range(1, len(hypothesis) + 1):
        deletions = matrix[i-1, 1:] + deletion_costs[i]
        
        matches   = np.array([
            replacement_cost.get(hypothesis[i-1], {}).get(t, default_replace_cost) for t in truth
        ])
        matches   = matrix[i-1, :-1] + matches * (truth_array != hypothesis[i-1])
        
        min_costs = np.minimum(deletions, matches)
        for j in range(1, len(truth) + 1):
            insertion   = matrix[i, j-1] + insertion_cost.get(truth[j-1], 1)

            matrix[i, j] = min(min_costs[j-1], insertion)
    
    if verbose:
        columns = [''] + [str(v) for v in truth]
        index = [''] + [str(v) for v in hypothesis]
        logger.info(pd.DataFrame(matrix, columns = columns, index = index))
    
    distance = matrix[-1, -1] if not partial else np.min(matrix[-1, 1:])
    if normalize:
        distance = distance / len(truth) if not partial else distance / len(hypothesis)
    
    return distance if not return_matrix else (distance, matrix)

def hamming_distance(hypothesis, truth, replacement_matrix = {}, normalize = True,
                     ** kwargs):
    """
        Compute a weighted hamming distance
        
        Arguments : 
            - hypothesis    : the predicted value   (iterable)
            - truth         : the true value        (iterable)
            - replacement_matrix    : weights to replace element 1 to 2 (from hypothesis to truth). Note that this is not in 2 sens so a --> b != b --> a
            - normalize     : whether to normalize on truth length or not
        Return : distance between hypothesis and truth (-1 if they have different length)
    """
    if len(hypothesis) != len(truth): return -1
    distance = sum([
        replacement_matrix.get(c1, {}).get(c2, 1)
        for c1, c2 in zip(hypothesis, truth) if c1 != c2
    ])
    if normalize: distance = distance / len(truth)
    return distance


_str_distance_method    = {
    'hamming'   : hamming_distance,
    'edit'      : edit_distance
}

_similarity_methods = {
    'cosine'    : cosine_similarity,
    'dp'    : dot_product,
    'iou'   : iou,
    'intersect'   : intersect
}

_distance_methods = {
    ** _str_distance_method,
    ** _similarity_methods,
    'l1'        : l1_distance,
    'l2'        : l2_distance,
    'manhattan' : manhattan_distance,
    'euclidian' : euclidian_distance
}