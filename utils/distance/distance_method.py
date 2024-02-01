
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

from utils.tensorflow_utils import tf_compile
from utils.wrapper_utils import dispatch_wrapper, partial

logger = logging.getLogger(__name__)

MAX_MATRIX_SIZE = 1024 * 1024 * 512

_str_distance_methods   = {}
_similarity_methods     = {}
_distance_methods       = {}

def distance_method_wrapper(fn = None, name = None, is_similarity = False, expand = True):
    def wrapper(fn):
        @functools.wraps(fn)
        def inner(x, y, as_matrix = False, ** kwargs):
            if expand: x, y = _maybe_expand_for_matrix(x, y, as_matrix = as_matrix)
            return fn(x, y, as_matrix = as_matrix, ** kwargs)
        
        key = name if name else fn.__name__.split('_')[0]
        
        distance.dispatch(inner, key)
        
        if is_similarity:   _similarity_methods[key] = inner
        return inner
    
    return wrapper if fn is None else wrapper(fn)

similarity_method_wrapper = partial(distance_method_wrapper, is_similarity = True)

@tf_compile(reduce_retracing = True, follow_type_hints = True)
def tf_distance(x : tf.Tensor,
                y : tf.Tensor,
                method,
                as_matrix   = True,
                force_distance  = False,
                ** kwargs
               ):
    return distance(
        x, y, method, force_distance = force_distance, as_matrix = as_matrix, ** kwargs
    )

@dispatch_wrapper(_distance_methods, 'method')
def distance(x,
             y,
             method,
             force_distance = True,

             as_matrix  = False,
             max_matrix_size    = MAX_MATRIX_SIZE,
             
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
        return _str_distance_methods[method](x, y, ** kwargs)

    if tf.executing_eagerly() and logger.isEnabledFor(logging.DEBUG):
        logger.debug('Calling {} on matrices with shapes {} and {}'.format(
            method, tuple(x.shape), tuple(y.shape)
        ))
    
    if isinstance(x, tf.Tensor) or isinstance(y, tf.Tensor):
        expand_fn = tf.expand_dims
        shape_fn    = tf.shape if not tf.executing_eagerly() else lambda t: t.shape
        x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    else:
        shape_fn, expand_fn = lambda t: t.shape, np.expand_dims
        x, y = x.astype(np.float32), y.astype(np.float32)
        
    if len(shape_fn(x)) == 1: x = expand_fn(x, axis = 0)
    if len(shape_fn(y)) == 1: y = expand_fn(y, axis = 0)

    max_x, transpose = -1, False
    if max_matrix_size > 0:
        if len(x) < len(y): x, y, transpose = y, x, as_matrix
        if as_matrix:
            max_x = tf.minimum(shape_fn(x)[0], tf.cast(tf.math.ceil(
                max_matrix_size / (shape_fn(x)[-1] * shape_fn(y)[0])
            ), tf.int32))
        else:
            max_x = tf.minimum(
                shape_fn(x)[0], tf.cast(tf.math.ceil(max_matrix_size / shape_fn(x)[-1]), tf.int32)
            )
        
    if max_x != -1 and max_x < shape_fn(x)[0]:
        n_slices    = tf.cast(tf.math.ceil(shape_fn(x)[0] / max_x), tf.int32)
        elem_shape  = (None, ) if not as_matrix else (None, y.shape[0])
        distances   = tf.TensorArray(
            dtype = tf.float32, size = n_slices, element_shape = elem_shape
        )
        if not as_matrix:
            for i in tf.range(n_slices):
                y_i = y if  shape_fn(y)[0] == 1 else y[i * max_x : (i + 1) * max_x]
                distances = distances.write(
                    i, tf.reshape(distance_fn(
                        x[i * max_x : (i + 1) * max_x], y_i, as_matrix = False, ** kwargs
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
    
    if transpose: distances = tf.transpose(distances)

    if tf.executing_eagerly() and logger.isEnabledFor(logging.DEBUG):
        logger.debug('Result shape : {}'.format(tuple(distances.shape)))
    
    return distances if not force_distance or method not in _similarity_methods else 1. - distances

@similarity_method_wrapper(expand = False)
def cosine_similarity(x, y, as_matrix = False, ** kwargs):
    norm_fn = tf.math.l2_normalize if isinstance(x, tf.Tensor) else lambda x: x / np.linalg.norm(x, axis = -1, keepdims = True)
    return dot_product(
        norm_fn(x), norm_fn(y), as_matrix = as_matrix, ** kwargs
    )

@similarity_method_wrapper(name = 'dp')
def dot_product(x, y, ** kwargs):
    sum_fn = tf.reduce_sum if isinstance(x, tf.Tensor) else np.sum
    return sum_fn(x * y, axis = -1)

@distance_method_wrapper
def l1_distance(x, y, ** kwargs):
    abs_fn = tf.abs if isinstance(x, tf.Tensor) else np.abs
    return abs_fn(x - y)

@distance_method_wrapper
def l2_distance(x, y, ** kwargs):
    square_fn = tf.square if isinstance(x, tf.Tensor) else np.square
    return square_fn(x - y)

@distance_method_wrapper
def manhattan_distance(x, y, ** kwargs):
    sum_fn, abs_fn = (tf.reduce_sum, tf.abs) if isinstance(x, tf.Tensor) else (np.sum, np.abs)
    return sum_fn(abs_fn(x - y), axis = -1)

@distance_method_wrapper
def euclidian_distance(x, y, ** kwargs):
    if isinstance(x, tf.Tensor):
        sqrt_fn, sum_fn, square_fn = tf.math.sqrt, tf.reduce_sum, tf.square
    else:
        sqrt_fn, sum_fn, square_fn = np.sqrt, np.sum, np.square
    
    return sqrt_fn(sum_fn(square_fn(x - y), axis = -1))

@similarity_method_wrapper
def dice_coeff(x, y, ** kwargs):
    if isinstance(x, tf.Tensor):
        sum_fn, max_fn = tf.reduce_sum, tf.maximum
    else:
        sum_fn, max_fn = np.sum, np.maximum

    inter = sum_fn(x * y)
    union = sum_fn(x) + sum_fn(y)
    return 2. * inter / max_fn(union, 1e-6)

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

iou = similarity_method_wrapper(
    partial(bbox_metric, metric = 'iou'), expand = False, name = 'iou'
)
intersect   = similarity_method_wrapper(
    partial(bbox_metric, metric = 'intersect'), expand = False, name = 'intersect'
)

def _maybe_expand_for_matrix(x, y, as_matrix = False):
    """
        Expands `x` and `y` such that a mathematical operation between them will compute all pairs, resulting in a matrix-like operation
        
        Example :
            `x.shape == (A, 1, d)` * `b.shape == (1, B, d)` -> `res.shape == (A, B, d)`
    """
    if isinstance(x, tf.Tensor):
        rank_fn, expand_fn = lambda t: len(tf.shape(t)), tf.expand_dims
    else:
        rank_fn, expand_fn = lambda t: t.ndim, np.expand_dims
    
    if as_matrix and rank_fn(x) == rank_fn(y):
        y = expand_fn(y, axis = -3)
    
    if rank_fn(x) == rank_fn(y) - 1:
        x = expand_fn(x, axis = -2)

    return x, y
