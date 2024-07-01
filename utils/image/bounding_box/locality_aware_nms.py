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

import math
import logging

from loggers import timer
from .metrics import compute_iou
from utils.keras_utils import TensorSpec, ops, graph_compile
from .non_max_suppression import _pad_boxes_to_tile_size, _prepare_boxes, nms, fast_nms

logger = logging.getLogger(__name__)

@nms.dispatch(('lanms', 'locality_aware_nms'))
@timer
@graph_compile(
    prefer_xla = True, prepare_for_graph = _pad_boxes_to_tile_size, static_argnames = ('merge_method', )
)
def lanms(boxes    : TensorSpec(shape = (None, None, 4), dtype = 'float'),
          scores   : TensorSpec(shape = (None, None), dtype = 'float') = None,
          max_output_size  : TensorSpec(shape = (), dtype = 'int32')   = None,
          nms_threshold    : TensorSpec(shape = (), dtype = 'float32')     = 0.25,
          merge_threshold  : TensorSpec(shape = (), dtype = 'float')   = 0.3,
          max_iter      : TensorSpec(shape = (), dtype = 'int32')   = None,
          merge_method  = 'union',
          tile_size = None
         ):
    if max_iter is None: max_iter = ops.shape(boxes)[-2]
    if max_output_size is None: max_output_size = ops.shape(boxes)[-2]
    
    boxes, scores, sorted_indices = _prepare_boxes(boxes, scores)
    nms_threshold = ops.cast(nms_threshold, boxes.dtype)
    merge_threshold = ops.cast(merge_threshold, boxes.dtype)
    
    batch_size  = ops.shape(boxes)[0]
    num_boxes   = ops.shape(boxes)[1]
    
    self_mask = ops.arange(num_boxes)
    self_mask = self_mask[None, None, :] > self_mask[None, :, None]

    boxes   = self_merging(boxes, self_mask, merge_threshold, merge_method, max_iter)
    mask    = ops.any(boxes > 0, axis = 2)
    
    nms_mask    = ops.cond(
        nms_threshold < merge_threshold,
        lambda: fast_nms(boxes, None, max_output_size, nms_threshold = nms_threshold)[2],
        lambda: mask
    )
    mask = ops.logical_and(mask, nms_mask)

    return boxes, None, mask

@timer
def self_merging(box_slice, mask, merge_threshold, merge_method, max_iter):
    def loop_cond(_boxes, loop_condition, idx):
        return ops.logical_and(loop_condition, idx < max_iter)
    
    def loop_body(boxes, loop_condition, idx):
        return _self_merging_body(
            boxes, loop_condition, mask, merge_threshold, merge_method, idx
        )
    
    return ops.while_loop(
        loop_cond, loop_body, [box_slice, True, 0]
    )[0]

@timer
def _self_merging_body(boxes, _, mask, merge_threshold, merge_method, idx):
    iou = compute_iou(boxes, as_matrix = True, source = 'yxyx')
    iou = iou * ops.cast(ops.logical_and(mask, iou >= merge_threshold), iou.dtype)
    # iou.shape == [batch_size, num_boxes, num_boxes]
    can_suppress_others = ops.cast(
        ops.reduce_max(iou, axis = 1) < merge_threshold, iou.dtype
    )[:, :, None]
    
    merging_mask    = iou * can_suppress_others >= merge_threshold
    suppressed_box  = ops.any(merging_mask, axis = 1)[:, :, None]

    merged  = _merge_boxes(boxes, boxes, merging_mask, merge_method)
    merged  = merged * ops.cast(ops.logical_not(suppressed_box), merged.dtype)
    
    if ops.executing_eagerly() and logger.isEnabledFor(logging.DEBUG):
        from utils.plot_utils import plot_boxes
        print(iou)
        plot_boxes(
            merged, title = 'Iteration #{}'.format(idx), with_legend = False, source = 'yxyx'
        )
    
    return [
        merged, ops.any(iou * ops.cast(suppressed_box, iou.dtype) >= merge_threshold), idx + 1
    ]

def _merge_boxes(boxes, box_slice, mask, merge_method):
    """
        Merges `box_slice` into `boxes` based on `mask`
        
        Arguments :
            - boxes : the original boxes with shape `(batch_size, num_boxes, 4)`
            - box_slice : the boxes to merge with shape `(batch_size, tile_size, 4)`
            - mask      : the merging mask with shape `(batch_size, num_boxes, tile_size)`
                mask[:, i, j] indicates whether to merge `boxes[:, i]` with `box_slice[:, j]`
        Return :
            - merged_boxes  : the updated `boxes` with same shape and dtype
    """
    mask    = ops.expand_dims(mask, axis = -1)
    merged  = ops.expand_dims(box_slice, 1) * ops.cast(mask, box_slice.dtype)
    if merge_method == 'union':
        union_xy_min   = ops.min(
            ops.where(mask, merged[..., :2], float('inf')), axis = 2
        )
        union_xy_max   = ops.max(merged[..., 2:], axis = 2)
        return ops.concatenate([
            ops.minimum(boxes[:, :, :2], union_xy_min),
            ops.maximum(boxes[:, :, 2:], union_xy_max)
        ], axis = 2)
    elif merge_method == 'average':
        sum_coords = boxes + ops.sum(merged, axis = 2)
        return ops.divide_no_nan(
            sum_coords, ops.cast(1 + ops.count_nonzero(mask, axis = 2), sum_coords.dtype)
        )
        

