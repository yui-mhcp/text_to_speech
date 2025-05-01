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
import math
import numpy as np

from loggers import timer
from ...wrappers import dispatch_wrapper
from ...sequence_utils import pad_to_multiple
from ...keras import TensorSpec, ops, graph_compile
from .metrics import compute_iou
from .converter import box_converter_wrapper

_nms_methods    = {}

@dispatch_wrapper(_nms_methods, 'method', default = 'nms')
@timer
@box_converter_wrapper('xyxy', force_tensor = True)
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
    
    
    if not callable(method):
        if method is None: method = 'tensorflow' if 'tensorflow' in sys.modules else 'fast'
        
        if method not in _nms_methods:
            raise ValueError("Unknown NMS method !\n  Accepted : {}\n  Got : {}".format(
                tuple(_nms_methods.keys()), method
            ))
        
        method = _nms_methods[method]
    
    if isinstance(boxes, dict): boxes, scores = boxes['boxes'], boxes.get('scores', scores)
    return method(boxes, scores, max_output_size, nms_threshold, ** kwargs)

@nms.dispatch('tensorflow')
@timer
@graph_compile(support_xla = False, force_tensorflow = True)
def tensorflow_nms(boxes  : TensorSpec(shape = (None, 4), dtype = 'float'),
                   scores : TensorSpec(shape = (None, ), dtype = 'float') = None,
                   max_output_size    : TensorSpec(shape = (), dtype = 'int32')    = None,
                   nms_threshold      : TensorSpec(shape = (), dtype = 'float32')  = 0.25
                  ):
    import tensorflow as tf
    
    if max_output_size is None: max_output_size = tf.shape(boxes)[0]
    if scores is None: scores = tf.ones((tf.shape(boxes)[0], ), dtype = tf.float32)
    
    indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size, nms_threshold
    )
    valids = tf.tensor_scatter_nd_update(
        tf.zeros((tf.shape(boxes)[0], ), dtype = tf.bool),
        indices = indices[:, None],
        updates = tf.ones((tf.shape(indices)[0], ), dtype = tf.bool)
    )
    
    return boxes, scores, valids

@timer
def _pad_boxes_to_tile_size(*, boxes, scores = None, tile_size = 512, ** kwargs):
    if not tile_size: return {}

    num_boxes = boxes.shape[0]
    if num_boxes % tile_size != 0:
        num_padding = (num_boxes // tile_size + 1) * tile_size - num_boxes
        
        boxes   = ops.pad(boxes, [(0, num_padding), (0, 0)])
        if scores is not None: scores = ops.pad(scores, [(0, num_padding)])
    
    return {'boxes' : boxes, 'scores' : scores, 'tile_size' : tile_size}

@nms.dispatch(('nms', 'standard', 'fast'))
@timer
@graph_compile(support_xla = True, prefer_xla = False, prepare_for_graph = _pad_boxes_to_tile_size)
def fast_nms(boxes  : TensorSpec(shape = (None, 4), dtype = 'float'),
             scores : TensorSpec(shape = (None, ), dtype = 'float') = None,
             max_output_size    : TensorSpec(shape = (), dtype = 'int32')   = None,
             nms_threshold  : TensorSpec(shape = (), dtype = 'float32') = 0.25,
             tile_size  = 512
            ):
    boxes, scores = _sort_boxes(boxes, scores)

    nms_threshold = ops.cast(nms_threshold, boxes.dtype)
    
    num_boxes = ops.shape(boxes)[0]
    self_mask = ops.arange(num_boxes)
    self_mask = self_mask[None, :] > self_mask[:, None]
    
    selected_boxes = self_suppression(boxes, self_mask, nms_threshold)

    valids = _get_valid_mask(selected_boxes, max_output_size)

    return selected_boxes, scores, valids
    

@nms.dispatch(('padded_nms', 'padded'))
@timer
@graph_compile(support_xla = True, prefer_xla = True, prepare = _pad_boxes_to_tile_size)
def padded_nms(boxes    : TensorSpec(shape = (None, None, 4), dtype = 'float'),
               scores   : TensorSpec(shape = (None, None), dtype = 'float') = None,
               max_output_size  : TensorSpec(shape = (), dtype = 'int32')   = None,
               nms_threshold    : TensorSpec(shape = (), dtype = 'float32') = 0.25,
               tile_size    : TensorSpec(shape = (), dtype = 'int32', static = True) = 512
              ):
    boxes, scores = _sort_boxes(boxes, scores)

    nms_threshold = ops.cast(nms_threshold, boxes.dtype)
    
    num_boxes   = ops.shape(boxes)[0]
    num_iterations  = num_boxes // tile_size
    
    self_mask = ops.arange(tile_size)
    self_mask = self_mask[None, :] > self_mask[:, None]

    def _loop_cond(unused_boxes, idx):
        return idx < num_iterations

    def _loop_body(boxes, idx):
        return _suppression_loop_body(
            boxes, self_mask, nms_threshold, tile_size, idx
        )

    selected_boxes, _ = ops.while_loop(
        _loop_cond, _loop_body, [boxes, ops.constant(0, 'int32')]
    )
    
    valids = _get_valid_mask(selected_boxes, max_output_size)
    
    return selected_boxes, scores, valids

@timer
def _suppression_loop_body(boxes, self_mask, iou_threshold, tile_size, idx):
    box_slice   = ops.slice(boxes, [idx * tile_size, 0], [tile_size, 4])
    # iterates over previous tiles to filter out boxes in the new tile that
    # overlaps with already selected boxes
    box_slice = cross_suppression(boxes, box_slice, iou_threshold, idx, tile_size)

    box_slice = self_suppression(box_slice, self_mask, iou_threshold)

    # set `box_slice` in `boxes`
    #mask    = ops.repeat(ops.arange(num_tiles) == idx, tile_size)[None, :, None]
    #boxes   = ops.where(mask, ops.tile(box_slice, [1, num_tiles, 1]), boxes)
    boxes   = ops.update_slice(boxes, [tile_size * idx, 0], box_slice)
    return boxes, idx + 1

@timer
def self_suppression(box_slice, mask, iou_threshold):
    iou = compute_iou(box_slice, as_matrix = True, source = 'xyxy')
    iou = iou * ops.cast(ops.logical_and(mask, iou >= iou_threshold), iou.dtype)

    iou_sum = ops.sum(iou)
    suppressed_iou, _, _, _ = ops.while_loop(
        lambda _iou, loop_condition, _iou_sum, _: loop_condition,
        _self_suppression_body,
        [iou, ops.constant(True), ops.sum(iou), iou_threshold]
    )
    suppressed_box = ops.any(suppressed_iou > 0, 0)[:, None]
    return box_slice * ops.cast(ops.logical_not(suppressed_box), box_slice.dtype)

@timer
def _self_suppression_body(iou, _, iou_sum, iou_threshold):
    can_suppress_others = ops.cast(
        ops.max(iou, axis = 0) < iou_threshold, iou.dtype
    )[:, None]
    iou_after_suppression = iou * ops.cast(
        ops.max(can_suppress_others * iou, 0) < iou_threshold, iou.dtype
    )[:, None]

    iou_sum_new = ops.sum(iou_after_suppression)
    return [
        iou_after_suppression,
        iou_sum - iou_sum_new > iou_threshold,
        iou_sum_new,
        iou_threshold
    ]

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
def _cross_suppression_body(boxes, box_slice, inner_idx, iou_threshold, tile_size):
    prev_slice   = ops.slice(
        boxes, [inner_idx * tile_size, 0], [tile_size, 4]
    )

    iou = compute_iou(box_slice, prev_slice, as_matrix = True, source = 'xyxy')
    
    box_slice_after_suppression =  box_slice * ops.cast(
        ops.all(iou < iou_threshold, axis = 1, keepdims = True), box_slice.dtype
    )
    return box_slice_after_suppression, inner_idx + 1

def _sort_boxes(boxes, scores):
    if scores is None: return boxes, None

    sorted_indices  = ops.flip(ops.argsort(scores, axis = 0), axis = 0)
    boxes   = ops.take(boxes, sorted_indices, axis = 0)
    scores  = ops.take(scores, sorted_indices, axis = 0)
    return boxes, scores

def _get_valid_mask(boxes, max_output_size):
    mask = ops.any(boxes > 0, axis = 1)
    if max_output_size is not None:
        mask = ops.logical_and(mask, ops.cumsum(ops.cast(mask, 'int32'), axis = 0) <= max_output_size)
    return mask
