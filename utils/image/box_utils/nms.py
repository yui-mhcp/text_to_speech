
# Copyright (C) 2023 yui-mhcp project's author. All rights reserved.
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
import tensorflow as tf

from functools import wraps

from utils.image.box_utils import BoxFormat, convert_box_format, bbox_iou

_nms_methods    = {}

def nms_method_wrapper(fn = None, names = None, box_format = BoxFormat.Y0X0Y1X1):
    def wrapper(fn):
        @wraps(fn)
        def inner(boxes, * args, ** kwargs):
            boxes = convert_box_format(
                boxes, box_format, as_list = False, extended = False, ** kwargs
            )
            kwargs.pop('box_mode', None)
            return fn(boxes, * args, ** kwargs)
        
        aliases = names if names else [fn.__name__]
        if not isinstance(aliases, (list, tuple)): aliases = [aliases]
        for n in aliases: _nms_methods[n] = inner

        return inner
    
    return wrapper if fn is None else wrapper(fn)

def nms(boxes, scores = None, nms_threshold = 0.25, method = 'nms', ** kwargs):
    if not callable(method) and method not in _nms_methods:
        raise ValueError("Unknown NMS method !\n  Accepted : {}\n  Got : {}".format(
            tuple(_nms_methods.keys()), method
        ))
    
    if scores is None and isinstance(boxes, dict): scores = boxes.get('score', None)
    
    fn = method if callable(method) else _nms_methods[method]
    return fn(
        boxes, scores = scores, nms_threshold = nms_threshold, ** kwargs
    )

@nms_method_wrapper(names = ('nms', 'standard', 'standard_nms'), box_format = BoxFormat.Y0X0Y1X1)
def standard_nms(boxes, scores, nms_threshold = 0.25, ** kwargs):
    """ Utility function interfacing to `tf.image.non_max_suppression` """
    indices = tf.image.non_max_suppression(
        boxes, scores, len(scores), nms_threshold
    )
    return tf.gather(boxes, indices), tf.gather(scores, indices)

@nms_method_wrapper(names = ('lanms', 'locality_aware_nms'), box_format = BoxFormat.CORNERS2)
def locality_aware_nms(boxes,
                       scores,
                       nms_threshold     = 0.25,
                       merge_threshold   = 0.01,
                       merge_method      = 'weighted',
                       ** kwargs
                      ):
    if merge_threshold == -1.: merge_threshold = nms_threshold
    
    ious = bbox_iou(
        boxes, box_mode = BoxFormat.CORNERS2, as_matrix = True, use_graph = True, ** kwargs
    )

    mask    = tf.cast(ious >= merge_threshold, scores.dtype)
    sum_mask    = tf.reduce_sum(mask, axis = -1)
    
    score_mask  = tf.expand_dims(scores, axis = 0) * mask
    sum_scores  = tf.reduce_sum(score_mask, axis = -1)
    
    scores  = sum_scores / sum_mask

    if merge_method == 'union':
        mask = tf.expand_dims(mask, axis = -1)
        merged = tf.expand_dims(boxes, axis = 1) * mask
        
        merged = tf.concat([
            tf.reduce_min(
                tf.where(mask == 1., merged[:, :, :2], merged.dtype.max), axis = 1
            ),
            tf.reduce_max(merged[:, :, 2:], axis = 1)
        ], axis = -1)
    else:
        if merge_method == 'weighted':
            mask, denom = score_mask, sum_scores
        else:
            denom = sum_mask
        
        merged  = tf.reduce_sum(
            tf.expand_dims(boxes, axis = 1) * tf.expand_dims(mask, axis = -1), axis = 1
        ) / tf.expand_dims(denom, axis = -1)
    
    return standard_nms(
        merged, scores, nms_threshold, box_mode = BoxFormat.CORNERS2
    )


#@nms_method_wrapper(names = ('lanms', 'locality_aware_nms'), box_format = BoxFormat.CORNERS2)
def locality_aware_nms_slow(boxes,
                       scores,
                       nms_threshold    = 0.25,
                       merge_threshold  = -1,
                       merge_method = 'weighted',
                       ** kwargs
                      ):
    def cond(i, selected_boxes, selected_scores, boxes, scores):
        return tf.shape(boxes)[0] > 0

    def body(i, selected_boxes, selected_scores, boxes, scores):
        ious = bbox_iou(
            boxes[:1], boxes, box_mode = BoxFormat.CORNERS2, use_graph = True, ** kwargs
        )
        
        mask    = ious >= merge_threshold
        merge_scores    = tf.boolean_mask(scores, mask)
        
        if merge_method == 'weighted':
            merged = tf.reduce_sum(
                tf.boolean_mask(boxes, mask) * merge_scores, axis = 0
            ) / tf.reduce_sum(merge_scores)
        elif merge_method == 'average':
            merged = tf.reduce_sum(
                tf.boolean_mask(boxes, mask) * merge_scores, axis = 0
            ) / tf.cast(tf.shape(merge_scores)[0], boxes.dtype)
        elif merge_method == 'union':
            merge_boxes = tf.boolean_mask(boxes, mask)
            merged = tf.concat([
                tf.reduce_min(merge_boxes[:, :2], axis = 0),
                tf.reduce_max(merge_boxes[:, 2:], axis = 0)
            ], axis = -1)
        
        return (
            i + 1,
            selected_boxes.write(i, merged),
            selected_scores.write(i, tf.reduce_mean(merge_scores)),
            tf.boolean_mask(boxes, tf.logical_not(mask)),
            tf.boolean_mask(scores, tf.logical_not(mask))
        )
    
    if merge_threshold == -1.: merge_threshold = nms_threshold
    
    n, boxes, scores, last_box, last_score = tf.while_loop(
        cond,
        body,
        loop_vars   = (
            tf.cast(0, tf.int32),
            tf.TensorArray(
                dtype = boxes.dtype, dynamic_size = True, size = len(scores), element_shape = (4, )
            ),
            tf.TensorArray(
                dtype = scores.dtype, dynamic_size = True, size = len(scores), element_shape = ()
            ),
            boxes,
            tf.expand_dims(scores, axis = -1)
        ),
        maximum_iterations  = len(scores)
    )
    
    return standard_nms(
        boxes.stack()[:n], scores.stack()[:n], nms_threshold, box_mode = BoxFormat.CORNERS2
    )

#@nms_method_wrapper(names = ('lanms', 'locality_aware_nms'), box_format = BoxFormat.CORNERS2)
def locality_aware_nms_v3(boxes,
                          scores,
                          nms_threshold     = 0.25,
                          merge_threshold   = -1.,
                          merge_method      = 'weighted',
                          n_iter    = 1,
                          ** kwargs
                         ):
    if merge_threshold == -1.: merge_threshold = nms_threshold
    
    for i in tf.range(n_iter):
        ious = bbox_iou(
            boxes, box_mode = BoxFormat.CORNERS2, as_matrix = True, use_graph = True, ** kwargs
        )

        mask    = ious >= merge_threshold
        triang_mask = tf.range(len(mask))[tf.newaxis] < tf.range(len(mask))[:, tf.newaxis]
        
        keep_mask   = tf.logical_and(mask, triang_mask)
        keep_mask   = tf.logical_not(tf.reduce_any(keep_mask, axis = -1))
        
        if tf.reduce_all(keep_mask): break
        
        boxes   = tf.boolean_mask(boxes, keep_mask)
        mask    = tf.boolean_mask(mask, keep_mask)
        
        mask    = tf.cast(mask, scores.dtype)
        sum_mask    = tf.reduce_sum(mask, axis = -1)

        score_mask  = tf.expand_dims(scores, axis = 0) * mask
        sum_scores  = tf.reduce_sum(score_mask, axis = -1)

        scores  = sum_scores / sum_mask

        if merge_method == 'union':
            boxes = tf.expand_dims(boxes, axis = 1) * tf.expand_dims(mask, axis = -1)

            boxes = tf.concat([
                tf.reduce_min(
                    tf.where(tf.expand_dims(mask, axis = -1) == 1., boxes[:, :, :2], boxes.dtype.max), axis = 1
                ),
                tf.reduce_max(boxes[:, :, 2:], axis = 1)
            ], axis = -1)
        else:
            if merge_method == 'weighted':
                mask, denom = score_mask, sum_scores
            else:
                denom = sum_mask

            boxes  = tf.reduce_sum(
                tf.expand_dims(boxes, axis = 1) * tf.expand_dims(mask, axis = -1), axis = 1
            ) / tf.expand_dims(denom, axis = -1)
    
    return boxes, scores

#@nms_method_wrapper(names = ('numba_nms', ), box_format = BoxFormat.CORNERS)
def numba_nms(boxes, scores, nms_threshold = 0.25, ** kwargs):
    result = np.empty((len(boxes), 4), dtype = boxes.dtype)
    result_scores   = np.empty((len(boxes), ), dtype = scores.dtype)
    
    indices = np.argsort(scores)[::-1]
    boxes, scores = boxes[indices], scores[indices]
    for i in range(len(boxes)):
        result[i], result_scores[i] = boxes[0], scores[0]

        ious = bbox_iou(
            boxes[0], boxes, box_mode = BoxFormat.CORNERS2, as_matrix = False
        )

        mask    = ious < nms_threshold
        boxes   = boxes[mask]
        scores  = scores[mask]
        if boxes.shape[0] == 0: break
    
    return result[: i + 1], result_scores[: i + 1]

def numpy_nms(boxes, scores = None, labels = None, nms_threshold = 0.25):
    if scores is not None:
        boxes = boxes[np.flip(np.argsort(scores))]
    
    mask = np.arange(len(boxes))[:, np.newaxis] > np.arange(len(boxes))
    
    if labels is not None:
        if len(labels.shape) == 2: labels = np.argmax(labels, axis = -1)
        mask = mask & (labels[:, np.newaxis] == labels)
        print(mask)

    ious = boxes_iou(boxes) * mask

    keep = np.ones((len(boxes), ), dtype = bool)
    for i, iou in enumerate(ious):
        if not keep[i] or i == 0: continue
        if np.any(iou[keep] >= threshold):
            keep[i] = False

    return boxes[keep]
