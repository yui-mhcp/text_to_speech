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

import sys
import math
import numpy as np

from loggers import timer
from utils import pad_to_multiple
from utils.keras_utils import TensorSpec, ops, graph_compile
from utils.wrapper_utils import dispatch_wrapper
from .metrics import compute_iou
from .converter import BoxFormat, NORMALIZE_RELATIVE, box_converter_wrapper

_nms_methods    = {}

@dispatch_wrapper(_nms_methods, 'method', default = 'nms')
@timer
@box_converter_wrapper(BoxFormat.XYXY, expand_dict = True)
def nms(boxes,
        scores  = None,
        max_output_size = None,
        nms_threshold   = 0.25,
        *,
        
        method  = None,
        return_indices  = False,
        
        ** kwargs
       ):
    """
        Apply the Non-Max Suppression (NMS) algorithm on `boxes`
        This method is an interface function supporting multiple NMS variants (see below)
        
        Arguments :
            - boxes : a 3D `Tensor` with shape `(batch_size, num_boxes, 4)`, the boxes to process
            - scores    : a 2D `Tensor` with shape `(batch_size, num_boxes)`, representing the confidence score for each box
            - max_output_size   : the maximum number of boxes to output
            - nms_threshold : the IoU threshold used to filter boxes
            
            - method    : the NMS method to use (see below)
            
            - kwargs    : forwarded to the effective NMS function
        Return :
            - boxes     : (possibly modified) boxes
                shape = `(batch_size, num_selected_boxes, 4)`
            - scores    : (possibly modified) scores
                shape = `(batch_size, num_selected_boxes)`
            - valid_mask    : boolean mask representing whether a box is selected or not
                shape = `(batch_size, num_selected_boxes)`
        
        Note that `num_selected_boxes <= max_output_size`
        Note that if `scores` is not provided, it will be equals to 1 for each box
    """
    if nms_threshold >= 1. or len(boxes) == 0:
        return boxes, scores, np.ones((len(boxes), ), dtype = bool)
    
    if method is None: method = 'tensorflow' if 'tensorflow' in sys.modules else 'fast'
    
    if not callable(method) and method not in _nms_methods:
        raise ValueError("Unknown NMS method !\n  Accepted : {}\n  Got : {}".format(
            tuple(_nms_methods.keys()), method
        ))
    
    if len(ops.shape(boxes)) == 2:
        if scores is not None: scores = scores[None]
        boxes = boxes[None]
    
    fn = method if callable(method) else _nms_methods[method]
    return fn(boxes, None, max_output_size, nms_threshold, ** kwargs)

@nms.dispatch('tensorflow')
@timer
@graph_compile(support_xla = False, force_tensorflow = True)
def tensorflow_nms(boxes  : TensorSpec(shape = (None, None, 4), dtype = 'float'),
                   scores : TensorSpec(shape = (None, None), dtype = 'float') = None,
                   max_output_size    : TensorSpec(shape = (), dtype = 'int32')    = None,
                   nms_threshold      : TensorSpec(shape = (), dtype = 'float32')  = 0.25
                  ):
    import tensorflow as tf
    
    if scores is None: scores = tf.ones(tf.shape(boxes)[:2], dtype = 'float32')
    if max_output_size is None: max_output_size = tf.shape(boxes)[1]

    valids = tf.zeros(tf.shape(boxes)[:2], dtype = 'bool')
    for i in range(len(boxes)):
        indices = tf.image.non_max_suppression(
            boxes[i], scores[i], max_output_size, nms_threshold
        )
        valids = tf.tensor_scatter_nd_update(
            valids,
            indices = tf.stack([tf.fill((len(indices), ), i), indices], axis = 1),
            updates = tf.ones((len(indices), ), dtype = 'bool')
        )
    
    return boxes, scores, valids

@timer
def _pad_boxes_to_tile_size(*, boxes, scores = None, tile_size = 512, ** kwargs):
    if tile_size: kwargs['tile_size'] = tile_size

    num_boxes = boxes.shape[1]
    if tile_size and num_boxes % tile_size != 0:
        num_padding = (num_boxes // tile_size + 1) * tile_size - num_boxes
        
        boxes   = ops.pad(boxes, [(0, 0), (0, num_padding), (0, 0)])
        if scores is not None: scores = ops.pad(scores, [(0, 0), (0, num_padding)])
    
    kwargs.update({'boxes' : boxes, 'scores' : scores})
    return kwargs

@nms.dispatch(('nms', 'standard', 'fast'))
@timer
@graph_compile(support_xla = False, prefer_xla = False, prepare_for_graph = _pad_boxes_to_tile_size)
def fast_nms(boxes  : TensorSpec(shape = (None, None, 4), dtype = 'float'),
             scores : TensorSpec(shape = (None, None), dtype = 'float') = None,
             max_output_size    : TensorSpec(shape = (), dtype = 'int32')   = None,
             nms_threshold  : TensorSpec(shape = (), dtype = 'float32') = 0.25,
             tile_size      = None
            ):
    if max_output_size is None: max_output_size = ops.shape(boxes)[-2]
    
    boxes, scores, sorted_indices = _prepare_boxes(boxes, scores)

    nms_threshold = ops.cast(nms_threshold, boxes.dtype)
    
    batch_size  = ops.shape(boxes)[0]
    num_boxes   = ops.shape(boxes)[1]
    
    self_mask = ops.arange(num_boxes)
    self_mask = self_mask[None, None, :] > self_mask[None, :, None]

    selected_boxes = self_suppression(boxes, self_mask, nms_threshold)

    valids = _get_valid_mask(selected_boxes, max_output_size)

    #valids = ops.any(selected_boxes > 0, axis = 2)
    #indexes = _get_selected_indexes(
    #    boxes, scores, sorted_indices, valids, num_boxes, max_output_size
    #)
    #num_valids  = ops.minimum(
    #    ops.cast(ops.count_nonzero(valids, axis = 1), 'int32'), max_output_size
    #)
    #return indexes, num_valids

    return selected_boxes, scores, valids
    

@nms.dispatch(('padded_nms', 'padded'))
@timer
@graph_compile(support_xla = True, prefer_xla = True, prepare = _pad_boxes_to_tile_size)
def padded_nms(boxes    : TensorSpec(shape = (None, None, 4), dtype = 'float'),
               scores   : TensorSpec(shape = (None, None), dtype = 'float') = None,
               max_output_size  : TensorSpec(shape = (), dtype = 'int32')   = None,
               nms_threshold    : TensorSpec(shape = (), dtype = 'float32') = 0.25,
               tile_size    = 512
              ):
    if max_output_size is None: max_output_size = ops.shape(boxes)[-2]
    
    boxes, scores, sorted_indices = _prepare_boxes(boxes, scores)

    nms_threshold = ops.cast(nms_threshold, boxes.dtype)
    
    batch_size  = ops.shape(boxes)[0]
    num_boxes   = ops.shape(boxes)[1]
    num_iterations  = num_boxes // tile_size
    
    self_mask = ops.arange(tile_size)
    self_mask = self_mask[None, None, :] > self_mask[None, :, None]

    def _loop_cond(unused_boxes, output_size, idx):
        return ops.logical_and(
            ops.reduce_min(output_size) < max_output_size, idx < num_iterations
        )

    def _loop_body(boxes, output_size, idx):
        return _suppression_loop_body(
            boxes, output_size, self_mask, nms_threshold, tile_size, idx
        )

    selected_boxes, output_size, _ = ops.while_loop(
        _loop_cond,
        _loop_body,
        [boxes, ops.zeros((batch_size, ), 'int32'), ops.constant(0, 'int32')]
    )
    #num_valid = ops.minimum(output_size, max_output_size)
    
    #return _get_selected_indexes(
    #    boxes, scores, sorted_indices, ops.any(selected_boxes > 0, axis = 2), num_boxes, max_output_size
    #), num_valid
    
    valids = _get_valid_mask(selected_boxes, max_output_size)
    
    return selected_boxes, scores, valids

@timer
def _suppression_loop_body(boxes, output_size, self_mask, iou_threshold, tile_size, idx):
    batch_size  = ops.shape(boxes)[0]
    box_slice   = ops.slice(boxes, [0, idx * tile_size, 0], [batch_size, tile_size, 4])
    # iterates over previous tiles to filter out boxes in the new tile that
    # overlaps with already selected boxes
    box_slice = cross_suppression(boxes, box_slice, iou_threshold, idx, tile_size)

    box_slice = self_suppression(box_slice, self_mask, iou_threshold)

    # Updates output_size.
    output_size += ops.cast(ops.count_nonzero(ops.any(box_slice > 0, axis = 2)), output_size.dtype)

    # set `box_slice` in `boxes`
    #mask    = ops.repeat(ops.arange(num_tiles) == idx, tile_size)[None, :, None]
    #boxes   = ops.where(mask, ops.tile(box_slice, [1, num_tiles, 1]), boxes)
    boxes   = ops.update_slice(boxes, [0, tile_size * idx, 0], box_slice)
    return boxes, output_size, idx + 1

@timer
def cross_suppression(boxes, box_slice, iou_threshold, slice_idx, tile_size):
    def loop_body(box_slice, idx):
        return _cross_suppression_body(boxes, box_slice, idx, iou_threshold, tile_size)
    
    box_slice, _= ops.while_loop(
        lambda _box_slice, inner_idx: inner_idx < slice_idx,
        loop_body,
        [box_slice, ops.constant(0)]
    )
    return box_slice

@timer
def self_suppression(box_slice, mask, iou_threshold):
    iou = compute_iou(box_slice, as_matrix = True, source = 'yxyx')
    iou = iou * ops.cast(ops.logical_and(mask, iou >= iou_threshold), iou.dtype)

    iou_sum = ops.sum(ops.sum(iou, 2), 1)
    suppressed_iou, _, _, _ = ops.while_loop(
        lambda _iou, loop_condition, _iou_sum, _: loop_condition,
        _self_suppression_body,
        [iou, ops.constant(True), iou_sum, iou_threshold]
    )
    suppressed_box = ops.any(suppressed_iou > 0, 1)
    return box_slice * ops.cast(ops.logical_not(suppressed_box), box_slice.dtype)[:, :, None]

@timer
def _cross_suppression_body(boxes, box_slice, inner_idx, iou_threshold, tile_size):
    new_slice   = ops.slice(
        boxes,
        [0, inner_idx * tile_size, 0],
        [ops.shape(boxes)[0], tile_size, 4]
    )

    iou = compute_iou(box_slice, new_slice, as_matrix = True, source = 'yxyx')
    
    box_slice_after_suppression =  box_slice * ops.expand_dims(
        ops.cast(ops.all(iou < iou_threshold, axis = 1), box_slice.dtype), axis = 2
    )
    return box_slice_after_suppression, inner_idx + 1

@timer
def _self_suppression_body(iou, _, iou_sum, iou_threshold):
    can_suppress_others = ops.cast(
        ops.reduce_max(iou, axis = 1) < iou_threshold, iou.dtype
    )[:, :, None]
    iou_after_suppression = iou * ops.cast(
        ops.reduce_max(can_suppress_others * iou, 1) < iou_threshold, iou.dtype
    )[:, :, None]

    iou_sum_new = ops.sum(ops.sum(iou_after_suppression, 2), 1)
    return [
        iou_after_suppression,
        ops.any(iou_sum - iou_sum_new > iou_threshold),
        iou_sum_new,
        iou_threshold
    ]

def _prepare_boxes(boxes, scores):
    if scores is None: return boxes, None, None

    sorted_indices  = ops.flip(ops.argsort(scores, axis = 1), axis = 1)
    boxes   = ops.take_along_axis(boxes, sorted_indices[..., None], axis = 1)
    scores  = ops.take_along_axis(scores, sorted_indices, axis = 1)
    return boxes, scores, sorted_indices

def _get_valid_mask(boxes, max_output_size):
    mask = ops.any(boxes > 0, axis = 2)
    mask = ops.logical_and(mask, ops.cumsum(ops.cast(mask, 'int32'), axis = 1) <= max_output_size)
    return mask

def _get_selected_indexes(boxes, scores, sorted_indices, valids, num_boxes, max_output_size):
    indexes = ops.range(num_boxes, 0, -1)[None]
    indexes = indexes * ops.cast(valids, indexes.dtype)
    #indexes = ops.where(valids, indexes, -1)
    #indexes = num_boxes - ops.cast(ops.top_k(indexes, max_output_size)[0], 'int32')
    indexes = num_boxes - ops.flip(ops.sort(indexes, axis = 1)[:, -max_output_size :], axis = 1)

    if sorted_indices is None: return indexes

    return ops.take_along_axis(sorted_indices, indexes, axis = 1) * ops.cast(valids, indexes.dtype)
