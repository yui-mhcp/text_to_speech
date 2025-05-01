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

import numpy as np

from functools import cache

from .. import timer
from . import core, numpy
from .builder import Ops
from .execution_contexts import is_tensorflow_graph

@cache
def __getattr__(name):
    if name not in globals():
        return Ops(name, submodule = 'math')
    return globals()[name]

def _np_top_k(x, k, sorted = True):
    indices = np.argsort(x, axis = -1)[..., ::-1][..., :k]
    return np.take_along_axis(x, indices, axis = -1), indices

top_k = Ops('top_k', numpy_fn = _np_top_k)

def _np_segment_reduction_fn(data, segment_ids, reduction_method, num_segments, sorted):
    if not isinstance(segment_ids, np.ndarray):
        segment_ids = np.array(segment_ids, dtype = 'int32')
    if num_segments is None:
        num_segments = np.max(segment_ids) + 1

    valid_indices   = segment_ids >= 0  # Ignore segment_ids that are -1
    valid_data      = data[valid_indices]
    valid_segment_ids   = segment_ids[valid_indices]

    data_shape      = list(valid_data.shape)
    data_shape[0]   = num_segments  # Replace first dimension (which corresponds to segments)

    if reduction_method == np.minimum:
        result = np.full(data_shape, np.max(data))
    elif reduction_method == np.maximum:
        result = np.ones(data_shape, dtype = valid_data.dtype) * - np.inf
    else:
        result = np.zeros(data_shape, dtype = valid_data.dtype)

    if sorted:
        reduction_method.at(result, valid_segment_ids, valid_data)
    else:
        sort_indices    = np.argsort(valid_segment_ids)

        reduction_method.at(result, valid_segment_ids[sort_indices], valid_data[sort_indices])

    return result


def _np_segment_sum(data, segment_ids, num_segments = None, sorted = False):
    return _np_segment_reduction_fn(
        data, segment_ids, np.add, num_segments, sorted
    )

def _np_segment_min(data, segment_ids, num_segments = None, sorted = False):
    return _np_segment_reduction_fn(
        data, segment_ids, np.minimum, num_segments, sorted
    )

def _np_segment_max(data, segment_ids, num_segments = None, sorted = False):
    return _np_segment_reduction_fn(
        data, segment_ids, np.maximum, num_segments, sorted
    )

def _tf_segment_min(data, segment_ids, num_segments = None, sorted = False):
    import tensorflow as tf
    
    if sorted:
        return tf.math.segment_min(data, segment_ids, num_segments)
    return tf.math.unsorted_segment_min(data, segment_ids, num_segments)

def _tf_segment_mean(data, segment_ids, num_segments = None, sorted = False):
    import tensorflow as tf
    
    if sorted:
        return tf.math.segment_mean(data, segment_ids, num_segments)
    return tf.math.unsorted_segment_mean(data, segment_ids, num_segments)


def _segment_min(data, segment_ids, num_segments = None, sorted = False):
    return - segment_max_op(- data, segment_ids, num_segments, sorted)

def _segment_mean(data, segment_ids, num_segments = None, sorted = False, weights = None):
    num = segment_sum_op(data, segment_ids, num_segments, sorted)
    den = core.cast(
        numpy.bincount(segment_ids, weights = weights, minlength = num_segments), data.dtype
    )
    
    return numpy.einsum('i..., i -> i...', num, numpy.divide_no_nan(1., den))

segment_sum_op  = Ops('segment_sum', numpy_fn = _np_segment_sum)
segment_max_op  = Ops('segment_max', numpy_fn = _np_segment_max)
segment_min_op  = Ops(
    'segment_min',
    keras_fn    = _segment_min,
    numpy_fn    = _np_segment_min,
    tensorflow_fn   = _tf_segment_min
)
segment_mean_op = Ops(
    'segment_mean',
    keras_fn    = _segment_mean,
    numpy_fn    = _segment_mean,
    tensorflow_fn   = _tf_segment_mean
)

def segment_weighted_mean_op(data, segment_ids, weights, num_segments = None, sorted = None):
    data = einsum('i..., i -> i...', data, weights)
    return _segment_mean(
        data, segment_ids, num_segments, sorted, weights = weights
    )

def segment_argsort_op(data, segment_ids, num_segments = None, sorted = None):
    if isinstance(segment_ids, np.ndarray) and not is_tensorflow_graph():
        _zeros = np.zeros((1, ) + tuple(data.shape[1:]), core.dtype_to_str(data.dtype))
        _indices = np.arange(len(segment_ids), dtype = 'int32')
    else:
        _zeros = core.zeros(numpy.concatenate([
            core.ones((1, ), dtype = 'int32'), core.shape(data)[1:]
        ], axis = 0), dtype = data.dtype)
        _indices = numpy.arange(core.shape(segment_ids)[0], dtype = 'int32')

    max_by_segment = segment_max_op(data, segment_ids, num_segments, sorted) + 1
    max_by_segment = numpy.concatenate([_zeros, numpy.cumsum(max_by_segment, axis = 0)[:-1]], axis = 0)
    max_by_segment = segment_repeat(max_by_segment, segment_ids, num_segments)
    
    indices = numpy.argsort(data + max_by_segment, axis = 0)
    indices = core.cast(indices, 'int32')
    
    first_segment_idx = segment_min_op(
        _indices, segment_ids, num_segments, sorted
    )
    first_segment_idx = segment_repeat(first_segment_idx, segment_ids, num_segments)
    first_segment_idx = numpy.expand_dims(first_segment_idx, list(range(1, len(indices.shape))))

    return numpy.subtract(indices, first_segment_idx)


def _segment_op_wrapper(segment_fn, name = None):
    @timer(debug = True, name = name or segment_fn.__name__)
    def inner(data, segment_ids, num_segments = None, sorted = False, axis = 0):
        if sorted: sorted = not is_tensorflow_graph()
        if axis != 0: data = numpy.swapaxes(data, 0, axis)
        
        data = segment_fn(data, segment_ids, num_segments, sorted)
        
        if axis != 0: data = numpy.swapaxes(data, 0, axis)
        
        return data
    
    inner.__name__  = name or (segment_fn.name if hasattr(segment_fn, 'name') else segment_fn.__name__)
    inner.__doc__   = segment_fn.__doc__
    return inner

segment_sum = _segment_op_wrapper(segment_sum_op)
segment_min = _segment_op_wrapper(segment_min_op)
segment_max = _segment_op_wrapper(segment_max_op)
segment_mean    = _segment_op_wrapper(segment_mean_op)
segment_argsort = _segment_op_wrapper(segment_argsort_op)
segment_weighted_mean   = _segment_op_wrapper(segment_weighted_mean_op)

@timer(debug = True)
def segment_repeat(data, segment_ids, num_segments, axis = 0):
    if num_segments == 1:
        return numpy.repeat(data, core.shape(segment_ids)[0], axis = axis)
    return numpy.take(data, segment_ids, axis = axis)


__all__ = list(
    k for k, v in globals().items() if (isinstance(v, Ops)) or (callable(v) and not k.startswith('_'))
)

