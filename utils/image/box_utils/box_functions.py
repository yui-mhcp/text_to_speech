
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

import os
import cv2
import enum
import logging
import functools
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from functools import wraps

from utils import get_timer
from utils.wrapper_utils import partial, insert_signature
from utils.generic_utils import get_enum_item
from utils.plot_utils import plot, plot_multiple
from utils.image.mask_utils import apply_mask
from utils.image.image_utils import normalize_color
from utils.image.image_io import load_image, save_image, get_image_size
from utils.image.box_utils.bounding_box import BoundingBox

timer, _, _  = get_timer()

logger = logging.getLogger(__name__)

_numeric_types  = (int, float, np.int32, np.float32)

MAX_01_VALUE    = 1.25

NORMALIZE_NONE  = 0
NORMALIZE_01    = 1
NORMALIZE_WH    = 2

class BoxFormat(enum.IntEnum):
    UNCHANGED   = -2
    DEFAULT     = -1
    XYWH        = 0
    X0Y0WH      = 0
    CORNERS     = 1
    X0Y0X1Y1    = 1
    DICT        = 2
    POLY        = 3
    POLY_FLAT   = 4
    CORNERS2    = 5
    Y0X0Y1X1    = 5
    OBJECT      = 6

class Shape(enum.IntEnum):
    CERCLE  = 0
    CIRCLE  = 0
    OVALE   = 1
    ELLIPSE = 1
    RECT    = 2
    RECTANGLE   = 2

FORMAT_WITH_ANGLE   = (BoxFormat.POLY_FLAT, BoxFormat.DICT, BoxFormat.OBJECT)

def box_converter_wrapper(box_format,
                          as_list,
                          force_np  = False,
                          extended  = None,
                          normalize = NORMALIZE_NONE
                         ):
    """
        This wrapper automatically converts the 1st argument (expected to be `boxes`) to the corresponding `box_format`
        It is a convenient wrapper to avoid calling `convert_box_format` as 1st lines of every function ;)
    """
    def wrapper(fn):
        @wraps(fn)
        def inner(boxes, * args, ** kwargs):
            if force_np: boxes = _to_np(boxes)
            
            boxes = convert_box_format(
                boxes,
                box_format,
                as_list = as_list,
                extended = extended,
                normalize_mode = normalize,
                ** kwargs
            )
            kwargs.pop('box_mode', None)
            return fn(boxes, * args, ** kwargs)
        
        return inner
    
    if extended is None: extended = as_list
    return wrapper

def _get_box_pos_length(box_mode):
    if box_mode == BoxFormat.POLY_FLAT: return 8
    elif box_mode in (BoxFormat.DICT, BoxFormat.OBJECT):    return -1
    else:   return 4

def _is_int(x):
    if isinstance(x, tf.Tensor):    return x.dtype == tf.int32
    elif isinstance(x, np.ndarray): return np.issubdtype(x.dtype, np.integer)
    return isinstance(x, int)

def _is_float(x):
    if isinstance(x, tf.Tensor):    return x.dtype == tf.float32
    elif isinstance(x, np.ndarray): return np.issubdtype(x.dtype, np.floating)
    return isinstance(x, float)

def _to_int(x):
    if isinstance(x, tf.Tensor):    return tf.cast(x, tf.int32)
    elif isinstance(x, np.ndarray): return x.astype(np.int32)
    elif isinstance(x, list):       return [_to_int(xi) for xi in x]
    return int(x)

def _to_float(x):
    if isinstance(x, tf.Tensor):    return tf.cast(x, tf.float32)
    elif isinstance(x, np.ndarray): return x.astype(np.float32)
    elif isinstance(x, list):       return [_to_float(xi) for xi in x]
    return float(x)

def _to_np(boxes):
    if hasattr(boxes, 'numpy'):     return boxes.numpy()
    elif isinstance(boxes, dict):   return {k : _to_np(v) for k, v in boxes.items()}
    elif isinstance(boxes, list):   return [_to_np(b) for b in boxes]
    return np.array(boxes)

def convert_box_format(box,
                       output_format    = BoxFormat.DEFAULT,
                       box_mode = BoxFormat.DEFAULT,
                       as_list  = None,
                       
                       image    = None,
                       image_h  = None,
                       image_w  = None,
                       dezoom_factor    = 1.,
                       normalize_mode   = NORMALIZE_NONE,
                       
                       extended = False,
                       score    = None,
                       angle    = None,
                       label    = None,
                       labels   = None,
                       ** kwargs
                      ):
    """
        Convert a bounding box format to another format
        
        Arguments :
            - box   : list / np.ndarray / tf.Tensor / dict / BoundingBox, the original box
            - output_format : a valid `BoxFormat`, the expected output format
            - as_list   : whether to concatenate the result or return a list
            
            - extended  : whether to add extra information (i.e. [angle, score, label])
            - score     : the box score to possibly add
            - angle     : the angle to possibly add
            - label     : the label to possibly add
            - labels    : list of labels to use if `label` is an `int`
        Return :
            - converted_box : the converted bounding box to `output_format` with possibly additional information (if `extended == True`)
        
        List of supported format (`BoxFormat` enum) :
            - UNCHANGED   : returns same format as the input
            - DEFAULT     : default to `xYWH`
            - XYWH / X0Y0WH : `[x, y, w, h]` where `(x, y)` are the coordinates of the top-left corner
            - X0Y0X1Y1 / CORNERS  : `[x_min, y_min, x_max, y_max]` coordinates
            - DICT        : `dict` with keys `{xmin, xmax, ymin, ymax, width, height}`
            - POLY        : 2-D array of shape `(4, 2)`, the `(x, y)` coordinates of the 4 points
            - POLY_FLAT   : `[x0, y0, x1, y1, x2, y2, x3, y3]`, the `(x, y)` coordinates of the 4 points represented as a 1-D array
            - Y0X0Y1X1 / CORNERS2   : `[y_min, x_min, y_max, x_max]` coordinates
            - OBJECT      : `BoundingBox` object

    """
    if as_list is None: as_list = extended
    
    if isinstance(box, (list, tuple)) and isinstance(box[0], (list, np.ndarray)): box = np.array(box)
    if isinstance(box, tf.Tensor):
        stack_fn, unstack_fn, min_fn, max_fn = tf.stack, tf.unstack, tf.reduce_min, tf.reduce_max
        reshape_fn = tf.reshape
    elif isinstance(box, (np.ndarray, dict)):
        unstack_fn = lambda x, ** kwargs: [x[..., i] for i in range(x.shape[-1])]
        stack_fn, min_fn, max_fn, reshape_fn = np.stack, np.min, np.max, np.reshape
    else:
        unstack_fn = lambda x, ** kwargs: x
        stack_fn, min_fn, max_fn, reshape_fn = lambda x, ** kw: np.array(x), min, max, np.reshape

    shape_fn = tf.shape if not tf.executing_eagerly() else lambda t: t.shape
    
    if isinstance(box, (list, tuple)):
        if isinstance(box[0], (dict, BoundingBox)):
            result = [convert_box_format(
                b,
                output_format,
                labels  = labels,
                box_mode    = box_mode,
                as_list     = as_list,
                extended    = extended,
                image       = image,
                image_h     = image_h,
                image_w     = image_w,
                dezoom_factor   = dezoom_factor,
                normalize_mode  = normalize_mode,
            ) for b in box]
            if output_format in (BoxFormat.OBJECT, BoxFormat.DICT): return result
            
            result = list(zip(* result))
            return result if as_list else stack_fn(
                result, axis = -1 if output_format != BoxFormat.POLY else -2
            )
    
    if box_mode == BoxFormat.DEFAULT:
        if not tf.executing_eagerly():      box_mode = BoxFormat.XYWH
        elif isinstance(box, BoundingBox):  box_mode = BoxFormat.OBJECT
        elif isinstance(box, dict):         box_mode = BoxFormat.DICT
        elif isinstance(box, (np.ndarray, tf.Tensor)):
            if box.shape[-2 :] == (4, 2):   box_mode = BoxFormat.POLY
            elif shape_fn(box)[-1] >= 8:    box_mode = BoxFormat.POLY_FLAT
            else:   box_mode = BoxFormat.XYWH
        else:       box_mode = BoxFormat.XYWH
    
    if output_format == BoxFormat.DEFAULT:      output_format = BoxFormat.XYWH
    elif output_format == BoxFormat.UNCHANGED:  output_format = box_mode
    
    if tf.executing_eagerly() and output_format == box_mode and dezoom_factor == 1. and not _should_normalize(box, normalize_mode) and not extended:
        return box if not as_list else unstack_fn(
            box, axis = -1, num = 2 if box_mode == BoxFormat.POLY else 4
        )

    
    if extended:
        if output_format == BoxFormat.POLY:
            raise ValueError(
                'The `POLY` format does not support extended form ! Prefer the `POLY_FLAT` format'
            )
        
        if score is None: score = get_box_score(box, box_mode = box_mode)
        if label is None: label = get_box_label(box, box_mode = box_mode, labels = labels)
        if angle is None: angle = get_box_angle(box, box_mode = box_mode)
    
    if box_mode == BoxFormat.POLY_FLAT:
        box_mode    = BoxFormat.POLY
        box         = reshape_fn(box[..., :8], box.shape[:-1] + [4, 2])
    
    if box_mode == BoxFormat.XYWH:
        x_min, y_min, w, h = unstack_fn(box, axis = -1, num = 4)
        x_max, y_max = x_min + w, y_min + h
    elif box_mode == BoxFormat.CORNERS:
        x_min, y_min, x_max, y_max = unstack_fn(box, axis = -1, num = 4)
        w, h = x_max - x_min, y_max - y_min
    elif box_mode == BoxFormat.DICT:
        x_min, y_min, x_max, y_max = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        w, h = box['width'], box['height']
    elif box_mode == BoxFormat.POLY:
        x_min, y_min = unstack_fn(min_fn(box, axis = -2), axis = -1, num = 2)
        x_max, y_max = unstack_fn(max_fn(box, axis = -2), axis = -1, num = 2)
        w, h = x_max - x_min, y_max - y_min
    elif box_mode == BoxFormat.CORNERS2:
        y_min, x_min, y_max, x_max = unstack_fn(box, axis = -1, num = 4)
        w, h = x_max - x_min, y_max - y_min
    elif box_mode == BoxFormat.OBJECT:
        x_min, y_min, x_max, y_max = box.rectangle
        w, h = x_max - x_min, y_max - y_min
    else:
        raise ValueError('Invalid input box format : {}'.format(box_mode))
    
    if dezoom_factor != 1.:
        x_min, y_min, w, h = dezoom_box(
            x_min, y_min, w, h, dezoom_factor, image_h = image_h, image_w = image_w, image = image
        )
    
    x_min, y_min, w, h = normalize_box(
        x_min, y_min, w, h, normalize_mode, image = image, image_h = image_h, image_w = image_w
    )
    
    x_max, y_max = x_min + w, y_min + h

    
    if output_format == BoxFormat.XYWH:
        result = [x_min, y_min, w, h]
    elif output_format == BoxFormat.CORNERS:
        result = [x_min, y_min, x_max, y_max]
    elif output_format == BoxFormat.DICT:
        result = {
            'xmin' : x_min, 'xmax' : x_max, 'ymin' : y_min, 'ymax' : y_max, 'width' : w, 'height' : h
        }
        if label is not None: result['label'] = label
        if score is not None: result['score'] = score
        if angle is not None: result['angle'] = angle
        return result
    elif output_format == BoxFormat.POLY:
        result = [
            stack_fn([x_min, y_min], axis = -1),
            stack_fn([x_max, y_min], axis = -1),
            stack_fn([x_max, y_max], axis = -1),
            stack_fn([x_min, y_max], axis = -1)
        ]
    elif output_format == BoxFormat.CORNERS2:
        result = [y_min, x_min, y_max, x_max]
    elif output_format == BoxFormat.OBJECT:
        return BoundingBox(
            x1 = x_min, y1 = y_min, x2 = x_max, y2 = y_max, score = score, angle = angle, label = label
        )
    else:
        raise ValueError('Invalid output format : {}'.format(output_format))
    
    if extended:
        if isinstance(result[0], (list, tuple, np.ndarray, tf.Tensor)):
            if not isinstance(angle, (list, np.ndarray, tf.Tensor)): angle = [angle] * len(result[0])
            if not isinstance(label, (list, np.ndarray, tf.Tensor)): label = [label] * len(result[0])
            if not isinstance(score, (list, np.ndarray, tf.Tensor)): score = [score] * len(result[0])

        result.extend(
            [angle, label, score] if output_format in FORMAT_WITH_ANGLE else [label, score]
        )
    
    return result if as_list else stack_fn(
        result, axis = -1 if output_format != BoxFormat.POLY else -2
    )


poly_to_box = partial(
    convert_box_format, output_format = BoxFormat.CORNERS, box_mode = BoxFormat.POLY, as_list = True
)
get_box_pos = partial(
    convert_box_format, output_format = BoxFormat.XYWH, as_list = True, extended = False
)
get_box_infos = partial(
    convert_box_format, output_format = BoxFormat.DICT
)

def get_box_area(box, ** kwargs):
    """ Returns the area of box(es) """
    _, _, w, h = get_box_pos(box, ** kwargs)
    return w * h

def get_box_additional_info(box, box_mode, index, key, default_value):
    """ Utility function that returns the expected `key` info from `box` """
    if isinstance(box, dict):           return box.get(key, default_value)
    elif isinstance(box, BoundingBox):  return getattr(box, key)
    elif box_mode == BoxFormat.POLY:    return default_value
    elif isinstance(box, list):
        return box[- index] if len(box) >= _get_box_pos_length(box_mode) + index else default_value
    else:
        return box[..., - index] if box.shape[-1] >= _get_box_pos_length(box_mode) + index else default_value

get_box_score   = partial(
    get_box_additional_info, box_mode = BoxFormat.XYWH, index = 1, key = 'score', default_value = 1.
)

def get_box_label(box, box_mode = BoxFormat.XYWH, labels = None):
    label = get_box_additional_info(box, box_mode, index = 2, key = 'label', default_value = None)
    return label if not labels or not isinstance(label, (int, np.integer)) else labels[label]

def get_box_angle(box, box_mode = BoxFormat.XYWH):
    if box_mode not in FORMAT_WITH_ANGLE: return 0.
    return get_box_additional_info(box, box_mode, index = 3, key = 'angle', default_value = 0.)


def _is_box_list(box):
    """ Returns whether `box` is a list of boxes or a single box, depending on its format """
    if isinstance(box, BoundingBox): return False
    elif isinstance(box, dict):
        return True if hasattr(box['width'], 'shape') and len(box['width'].shape) == 1 else False
    if isinstance(box, (tuple, list)):
        if len(box) < 4 or len(box) > 11:       return True
        if isinstance(box[0], _numeric_types):  return False
        return True
    elif hasattr(box, 'shape'):
        if len(box.shape) == 1: return False
        elif box.shape == (4, 2): return False
        elif len(box.shape) == 2: return True
        elif len(box.shape) == 3 and box.shape[1:] == (4, 2): return True
        else:
            raise ValueError('Invalid box shape : {}'.format(box.shape))
    else:
        raise ValueError("Unknown box type ({}) : {}".format(type(box), box))

def _is_01_box(* coords):
    """ Returns whether a list of given coordinates can be interpreted as a normalized box or not """
    def _check_01(c):
        if isinstance(c, dict): return all(_check_01(v) for v in c.values())
        elif isinstance(c, list): return all(_check_01(ci) for ci in c)
        elif _is_int(c): return False
        if isinstance(c, np.ndarray):   return np.all(c <= MAX_01_VALUE)
        elif isinstance(c, tf.Tensor):  return tf.reduce_all(c <= MAX_01_VALUE)
        return c <= MAX_01_VALUE
    if not tf.executing_eagerly():
        return _check_01(coords[0])
    return all(_check_01(c) for c in coords)

def _should_normalize(box, normalize_mode):
    """ Returns whether `box` should be normalized to `normalize_mode` or not """
    if normalize_mode == NORMALIZE_NONE: return False
    is_01 = _is_01_box(box)
    return (is_01 and normalize_mode == NORMALIZE_WH) or (not is_01 and normalize_mode == NORMALIZE_01)

def dezoom_box(x, y, w, h, factor, angle = 0., image = None, image_h = None, image_w = None):
    if factor == 1.: return (x, y, w, h)
    if angle != 0.: raise NotImplementedError('Rotated box is not supported yet !')
    
    if isinstance(x, tf.Tensor):
        min_fn, max_fn = tf.minimum, tf.maximum
    elif isinstance(x, np.ndarray):
        min_fn, max_fn = np.minimum, np.maximum
    else:
        min_fn, max_fn = min, max
    
    is_01 = _is_01_box(x, y, w, h)
    if image is not None: image_h, image_w = get_image_size(image)
    if not is_01:
        assert image_h and image_w, "You must provide image dimensions to avoid overruns"
        MAX_Y, MAX_X    = image_h, image_w
    else:
        MAX_Y, MAX_X    = 1., 1.
    
    new_h, new_w = h * factor, w * factor
    
    new_x   = max_fn(0., x - ((new_w - w) / 2.))
    new_y   = max_fn(0., y - ((new_h - h) / 2.))
    
    new_box = [new_x, new_y, min_fn(new_w, MAX_X - new_x), min_fn(new_h, MAX_Y - new_y)]
    return new_box if is_01 else _to_int(new_box)

def normalize_box(x, y, w, h, normalize_mode, image = None, image_h = None, image_w = None):
    """
        Normalizes the `[x, y, w, h]` coordinates to `normalize_mode`
            - NORMALIZE_NONE    : no normalization
            - NORMALIZE_01      : normalizes to the `[0, 1]` range (i.e. returns `float`)
            - NORMALIZE_WH      : normalizes to the dimension of the images (i.e. returns `int`)
    """
    if normalize_mode == NORMALIZE_NONE:
        return [x, y, w, h]

    if isinstance(x, tf.Tensor):
        min_fn, max_fn, cast_int = tf.minimum, tf.maximum, lambda b: tf.cast(b, tf.int32)
    elif isinstance(x, np.ndarray):
        min_fn, max_fn, cast_int = np.minimum, np.maximum, lambda b: b.astype(np.int32)
    else:
        min_fn, max_fn, cast_int = min, max, int

    x, y, w, h = max_fn(0, x), max_fn(0, y), max_fn(0, w), max_fn(0, h)
    
    is_01 = _is_01_box(x, y, w, h)
    if is_01:
        MAX_Y, MAX_X    = 1., 1.
    else:
        if image_h is None or image_w is None:
            assert image is not None
            image_h, image_w = get_image_size(image)
        x, y, w, h      = [_to_int(coord) for coord in (x, y, w, h)]
        MAX_Y, MAX_X    = image_h, image_w
    
    x, y = min_fn(MAX_X, x), min_fn(MAX_Y, y)
    w, h = min_fn(MAX_X - x, w), min_fn(MAX_Y - y, h)
    
    if (is_01 and normalize_mode == NORMALIZE_01) or (not is_01 and normalize_mode == NORMALIZE_WH):
        return [x, y, w, h]

    if image_h is None or image_w is None:
        assert image is not None
        image_h, image_w = get_image_size(image)
    
    assert image_h and image_w, 'You must provide image dimension to normalize the box(es) !'

    if is_01: # normalize_mode == NORMALIZE_WH
        return [_to_int(coord) for coord in (
            x * image_w, y * image_h, w * image_w, h * image_h
        )]
    return [
        x / image_w, y / image_h, w / image_w, h / image_h
    ]

def sort_boxes(boxes,
               scores = None,
               method = 'top',
               
               threshold    = 0.5,
               columns      = 10,
               
               return_indices = False,
               ** kwargs
              ):
    """
        Sorts the `boxes` based on `method` sorting criterion
        
        Arguments :
            - boxes : the boxes to sort (any format supported by `convert_box_format`)
            - scores    : the boxes scores (only used if `method == 'scores'`)
            - method    : the sorting criterion
                - x / y     : sorts based on the `x` or `y` coordinate of the top-left corner
                - corner    : sorts based on `x + y` coordinate
                - top       : sorts from top to bottom by computing "rows" with tolerance (i.e. 2 boxes may be on the same "line" but not exactly on the same pixel line)
                - left      : sorts from left to right by splitting the image in "columns"
                - score     : sorts in decreasing order of scores (`scores` must be provided)
            
            - threshold : the tolerance threshold for the "top" sorting method
            - columns   : the number of columns for the "left" sorting method
            
            - return_indices    : whether to return the sorted boxes or the sorted indices
            - kwargs    : ignored
        Return :
            If `return_indices == True`:
                - sorted_indices    : list of sorted indices
            Else :
                - sorted_boxes  : same box format as `boxes` sorted
    """
    if len(boxes) == 0:
        if return_indices: return []
        return boxes if scores is None else (boxes, scores)
    
    x, y, w, h = convert_box_format(
        _to_np(boxes), BoxFormat.XYWH, as_list = True, extended = False, ** kwargs
    )

    if isinstance(x, tf.Tensor):
        min_fn, max_fn, sort_fn, round_fn = tf.reduce_min, tf.reduce_max, tf.argsort, tf.math.round
        abs_fn, gather_fn, mask_fn = tf.abs, tf.gather, tf.where
    else:
        def mask_fn(mask, orig, new):
            orig[mask] = new
            return orig
        
        min_fn, max_fn, sort_fn, round_fn = np.min, np.max, np.argsort, np.round
        abs_fn = np.abs
        gather_fn = lambda x, idx: x[idx] if not isinstance(x, list) else [x[i] for i in idx]
    
    gather = lambda x, idx: {k : gather_fn(v, idx) for k, v in x.items()} if isinstance(x, dict) else gather_fn(x, idx)
    
    if method == 'corner':  indices = sort_fn(x + y)
    elif method == 'y':     indices = sort_fn(y)
    elif method == 'x':     indices = sort_fn(x)
    elif method == 'top':
        y_center    = y + h / 2.
        same_rows   = abs_fn(y_center[None, :] - y_center[:, None]) <= h[:, None] * threshold / 2.
        same_rows   = np.logical_or(same_rows, same_rows.T)
        
        indices = []
        to_set  = np.full((len(x), ), True)
        for idx in sort_fn(y + h / 2.):
            if not to_set[idx]: continue
            
            row_indices = np.where(
                np.logical_and(same_rows[idx], to_set)
            )[0]
            indices.extend(row_indices[sort_fn(x[row_indices])])
            to_set[row_indices] = False
        
    elif method == 'left':  indices = sort_fn(round_fn(x * columns) + y)
    elif method == 'score': indices = sort_fn(scores)[::-1]
    elif method == 'area':  indices = sort_fn(w * h)[::-1]
    
    if return_indices: return indices
    
    if scores is None: return gather(boxes, indices)
    return gather(boxes, indices), gather_fn(scores, indices)

def rearrange_rows(rows, align_left = True, align_right = True):
    """
        Sorts the `rows` from top to bottom
        
        Arguments :
            - rows  : 2D `np.ndarray` with shape `[n_rows, 4]` in the `X0Y0X1Y1` format
            - align_left    : whether to set the left border equals for all rows
            - align_right   : whether to set the right border equals to all (except last) rows
        Return :
            - sorted_rows   : `np.ndarra` with same shape and type as `rows`
    """
    if len(rows) <= 1: return rows
    rows = sort_boxes(rows, method = 'y', box_mode = BoxFormat.CORNERS)
    if align_left:  rows[:, 0] = np.min(rows[:, 0])
    if align_right: rows[: -1, 2] = np.max(rows[:, 2])
    return rows

def compute_union(boxes):
    """
        Returns a single box (1D `np.ndarray`) corresponding to the union of `boxes` (in `X0Y0X1Y1` format)
    """
    return np.concatenate([
        np.min(boxes[:, :2], axis = 0), np.max(boxes[:, 2:], axis = 0)
    ], axis = -1)

def _group_boxes(boxes, mask, indices = None, as_list = False):
    """
        Combine boxes according to the combination mask
        
        Arguments :
            - boxes : np.ndarray with shape [N, 4] in `BoxFormat.CORNERS` mode
            - mask  : np.ndarray with shape [N - 1], where mask[i] tells whether to combine `boxes[i]` with `boxes[i + 1]`
            - indices   : list of boxes indices
        Return :
            - comb_boxes    : np.ndarray containing the combined boxes
            - comb_indices  : list of the same length as `comb_boxes` containing, at each index, the (possible list) of the boxes combined
            - individuals   : list of individual boxes composing each box in `comb_boxes`
        
        Example : 
        ```
            boxes   = np.array([
                [0, 0, 2, 2],
                [1, 1, 3, 3],
                [3, 3, 4, 4]
            ])
            # it means : combine box 0 with box 1, not box 1 with box 2
            mask    = np.array([True, False])
            comb_boxes, comb_indices, individuals = _group_boxes(boxes, mask)
            print(comb_boxes) # [[0, 0, 3, 3], [3, 3, 4, 4]]
            print(comb_indices) # [[0, 1], 2]
            print(individuals)  # [array([[0, 0, 2, 2], [1, 1, 3, 3]]), array([3, 3, 4, 4])]
        ```
    """
    if indices is None: indices = list(range(len(boxes)))
    n, result, comb_indices, individuals = 0, [], [], []
    for v, group in itertools.groupby(mask):
        length = len(list(group))

        if not v:
            if n > 0: length -= 1
            if n + length == len(mask): length += 2
            result.extend(boxes[n : n + length])
            individuals.extend(boxes[n : n + length, np.newaxis])
            comb_indices.extend([idx] for idx in indices[n : n + length])
        else:
            length += 1
            result.append(compute_union(boxes[n : n + length]))
            individuals.append(boxes[n : n + length])
            comb_indices.append(indices[n : n + length])
        n += length
    
    if not as_list: result = np.array(result)
    return result, comb_indices, individuals

@timer
@box_converter_wrapper(BoxFormat.CORNERS, as_list = False, force_np = True, normalize = NORMALIZE_01)
def combine_boxes_horizontal(boxes,
                             indices    = None,
                             x_threshold    = 0.025,
                             h_threshold    = 0.025,
                             overlap_threshold  = 0.65,
                             ** kwargs
                            ):
    """
        Combines a list of boxes according to the following criteria :
            1.1 The distance between the right-corner and the left-corner of the next one is smaller than `x_threshold`
            1.2 The two boxes overlap in the x-axis
            2.1 The y-overlap is higher than `overlap_threshold`% of the maximal height
            2.2 The maximal height minus the y-overlap is smaller than `h_threshold`
        
        Note : boxes are normalized to the [0, 1] range. The threshold should be a percentage of the image dimension (except `overlap_threshold` which is a percentage of the maximal height)
        
        Note 2 : for efficiency, boxes are sorted according to the 'top' method (see `sort_boxes` for more details), and are only compared with the next one given the sorted indices.
        It is therefore possible that some (expected) fusions are not performed due to a wrong order
    """
    if indices is None: indices = list(range(len(boxes)))
    
    if len(boxes) <= 1: return boxes, indices, [boxes]
    
    sorted_indexes = sort_boxes(
        boxes, method = 'top', box_mode = BoxFormat.CORNERS, return_indices = True
    )
    boxes   = boxes[sorted_indexes]
    indices = [indices[idx] for idx in sorted_indexes]
    
    h = boxes[:, 3] - boxes[:, 1]
    
    diff_border = np.abs(boxes[:-1, 2] - boxes[1:, 0])
    overlap_x   = (
        np.minimum(boxes[:-1, 2], boxes[1:, 2]) -
        np.maximum(boxes[:-1, 0], boxes[1:, 0])
    ) > 0.
    diff_border[overlap_x] = 0.
    overlap_y   = np.maximum(0., (
        np.minimum(boxes[:-1, 3], boxes[1:, 3]) - np.maximum(boxes[:-1, 1], boxes[1:, 1])
    ))
    max_h   = np.maximum(h[:-1], h[1:])

    should_combine_horizontal = np.logical_and(
        np.logical_or(diff_border <= x_threshold, overlap_x),
        np.logical_or(overlap_y / max_h >= overlap_threshold, max_h - overlap_y <= h_threshold)
    )
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Horizontal fusion : {}\n- Border dist : {}\n- Overlap (x) {}\n- overlap_y / max(h) : {}\nmax(h) - overlap (y) {}\nBoxes :\n{}'.format(
            should_combine_horizontal,
            np.around(diff_border, decimals = 3),
            overlap_x,
            np.around(overlap_y / max_h, decimals = 3),
            np.around(max_h - overlap_y, decimals = 3),
            np.around(boxes, decimals = 3)
        ))
    
    return _group_boxes(boxes, should_combine_horizontal, indices = indices)

@timer
@box_converter_wrapper(BoxFormat.CORNERS, as_list = False, force_np = True, normalize = NORMALIZE_01)
def combine_boxes_vertical(boxes,
                           indices  = None,
                           y_threshold  = 0.015,
                           h_threshold  = 0.02,
                           ** kwargs
                          ):
    """
        Combines a list of boxes according to the following criteria :
            1.1 The distance between the right-corner and the left-corner of the next one is smaller than a threshold
            1.2 The two boxes overlap in the x-axis
            2. The difference between the highest height and the overlap (in y-axis) is higher than the threshold
        
        Note that boxes are normalized to [0, 1] range so the threshold should be a percentage of the image dimension
    """
    if indices is None: indices = list(range(len(boxes)))
    
    if len(boxes) <= 1: return boxes, indices, [boxes]

    sorted_indexes = sort_boxes(
        boxes, method = 'left', return_indices = True, box_mode = BoxFormat.CORNERS
    )
    boxes   = boxes[sorted_indexes]
    indices = [indices[idx] for idx in sorted_indexes]
    
    h = boxes[:, 3] - boxes[:, 1]
    
    diff_border = np.abs(boxes[:-1, 3] - boxes[1:, 1])
    h_diff      = np.maximum(h[:-1], h[1:]) - np.minimum(h[:-1], h[1:]) 
    overlap_x   = np.minimum(boxes[:-1, 2], boxes[1:, 2]) - np.maximum(boxes[:-1, 0], boxes[1:, 0])
    overlap_y   = np.logical_and(
        boxes[:-1, 3] < boxes[1:, 3],
        np.minimum(boxes[:-1, 3], boxes[1:, 3]) - np.maximum(boxes[:-1, 1], boxes[1:, 1]) > 0.
    )
    
    should_combine_vertical = np.logical_and(
        np.logical_or(diff_border <= y_threshold, overlap_y),
        np.logical_and(overlap_x > 0., h_diff <= h_threshold)
    )

    if logger.isEnabledFor(logging.DEBUG):
        #if 'image' in kwargs:
        #    show_boxes(kwargs['image'], boxes, box_mode = BoxFormat.CORNERS, ncols = 5)
        logger.debug('Vertical fusion : {}\n- Distance border : {}\n- h diff : {}\n- overlap (x) : {}\n- overlap (y) : {}'.format(
            should_combine_vertical,
            np.around(diff_border, decimals = 3),
            np.around(h_diff, decimals = 3),
            overlap_x > 0., overlap_y > 0.
        ))

    return _group_boxes(boxes, should_combine_vertical, indices)

@timer
def combine_overlapping_boxes(boxes, indices, rows, iou_threshold = 0.333, ** kwargs):
    def _update_box(main_idx, sub_idx, skip_indexes, boxes, indices, rows):
        if main_idx == sub_idx: return boxes, indices, rows
        main_rows, sub_rows = rows[main_idx], rows[sub_idx]
        
        if len(main_rows) < len(sub_rows):
            main_idx, main_rows, sub_idx, sub_rows = sub_idx, sub_rows, main_idx, main_rows
        
        main_rows_center_y  = np.expand_dims((main_rows[:, 3] + main_rows[:, 1]) / 2., axis = 0)
        sub_rows_center_y   = np.expand_dims((sub_rows[:, 3] + sub_rows[:, 1]) / 2., axis = 1)
        
        y_diff      = np.abs(main_rows_center_y - sub_rows_center_y)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Main row (index {}) {} - sub row shape (index {}) {} - diff : {}'.format(
                main_idx, main_rows.shape, sub_idx, sub_rows.shape, np.around(y_diff, decimals = 3)
            ))
        
        if np.any(np.all(y_diff > iou_threshold, axis = -1)):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Skip fusion as distances are too large')
            return boxes, indices, rows
        
        best_row   = np.argmin(y_diff, axis = -1)
        
        assert len(sub_rows) == len(indices[sub_idx]) == len(best_row), '{} - {} - {}'.format(
            sub_rows.shape, indices[sub_idx], best_row.shape
        )
        for sub_row_i, sub_indices_i, best_row_idx in zip(sub_rows, indices[sub_idx], best_row):
            main_rows[best_row_idx] = np.concatenate([
                np.minimum(main_rows[best_row_idx, :2], sub_row_i[:2]),
                np.maximum(main_rows[best_row_idx, 2:], sub_row_i[2:]),
            ], axis = -1)
            indices[main_idx][best_row_idx].extend(sub_indices_i)
        
        boxes[main_idx] = compute_union(main_rows)
        main_rows[:, 0] = boxes[main_idx, 0]
        
        skip_indexes.update({
            k : v if v != sub_idx else main_idx for k, v in skip_indexes.items()
        })
        skip_indexes[sub_idx] = main_idx
        return boxes, indices, rows
    
    if len(boxes) <= 1: return boxes, indices, rows
    
    sorted_indexes = sort_boxes(
        boxes, method = 'corner', return_indices = True, ** kwargs
    )
    boxes   = boxes[sorted_indexes]
    rows    = [rows[idx] for idx in sorted_indexes]
    indices = [indices[idx] for idx in sorted_indexes]

    intersect   = bbox_intersect(boxes, as_matrix = True, use_graph = False, ** kwargs)
    mask    = intersect >= iou_threshold
    mask[np.eye(len(mask), dtype = bool)] = False
    main_indexes, sub_indexes = np.where(mask)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Boxes :\n{}\nIntersect :\n{}\nMerge indexes :\n{}'.format(
            np.around(boxes, decimals = 2),
            np.around(intersect, decimals = 2),
            np.stack([main_indexes, sub_indexes], -1)
        ))


    skip_indexes = {}
    for main_idx, sub_idx in zip(main_indexes, sub_indexes):
        redirections = set()
        while main_idx in skip_indexes:
            if main_idx == sub_idx: break
            if main_idx in redirections:
                logger.warning('Main index {} (row shape {}) was already fusionned with another index : {} (rows shape {})'.format(main_idx, rows[main_idx].shape, skip_indexes[main_idx], rows[skip_indexes[main_idx]].shape))
                break
            
            redirections.add(main_idx)
            main_idx = skip_indexes[main_idx]
        
        if main_idx in skip_indexes:
            continue
        
        boxes, indices, rows = _update_box(main_idx, sub_idx, skip_indexes, boxes, indices, rows)
    
    boxes   = boxes[[idx for idx in range(len(boxes)) if idx not in skip_indexes]]
    indices = [indices[idx] for idx in range(len(indices)) if idx not in skip_indexes]
    rows    = [
        rearrange_rows(rows_i) for i, rows_i in enumerate(rows) if i not in skip_indexes
    ]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Merged boxes :\n{}'.format(np.around(boxes, decimals = 2)))
    
    return boxes, indices, rows

@timer
@box_converter_wrapper(BoxFormat.CORNERS, as_list = False, force_np = True, normalize = NORMALIZE_01)
@insert_signature(combine_boxes_horizontal, combine_boxes_vertical, combine_overlapping_boxes)
def combine_boxes(boxes, indices = None, ** kwargs):
    """
        Combines `boxes` (list of e.g., single-word boxes) by creating horizontal then vertical combinations.
        This enables, as an example, to combine the individual words detected by `EAST` to sentences (horizontal lines), then paragraphs (vertically grouping lines)
        
        This method calls sequentially
            1) {combine_boxes_horizontal}
            2) {combine_boxes_vertical}
            3) {combine_overlapping_boxes}`
    """
    combined_boxes, combined_indices_h, _ = combine_boxes_horizontal(
        boxes, indices = indices, box_mode = BoxFormat.CORNERS, ** kwargs
    )
    combined_boxes, combined_indices_v, rows = combine_boxes_vertical(
        combined_boxes, indices = combined_indices_h, box_mode = BoxFormat.CORNERS, ** kwargs
    )

    combined_boxes, combined_indices, rows = combine_overlapping_boxes(
        combined_boxes, combined_indices_v, rows, box_mode = BoxFormat.CORNERS, ** kwargs
    )
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Combined indices :\n  Horizontal : {}\n  Vertical : {}\n  Final : {}'.format(
            combined_indices_h, combined_indices_v, combined_indices
        ))
    
    return combined_boxes, combined_indices, rows

@timer
def bbox_iou(box1, box2 = None, as_matrix = False, use_graph = None, metric = 'iou', ** kwargs):
    """
        Computes the Intersect Over Union (IOU) between bounding boxes.
        This method is  equivalent to `utils.distance.distance(box1, box2, method = 'iou', ...)`
        
        Arguments :
            - box1 / box2   : the bounding boxes (any format supported by `convert_box_format`)
            - as_matrix     : whether to compute IOU between all pairs of boxes or not
            - use_graph     : whether to use `tf_distance` (compiled with `tf.function`) instead of `distance` (if not provided, `use_graph = as_matrix`).
            - kwargs    : forwarded to `distance / tf_distance`
        Return :
            - ious      : the IoU between each boxes (if `as_matrix = False`) or the IoU between all pairs of boxes
        
        Note : this function has been optimized to support either np.ndarray either tf.Tensor while avoiding unnecessary convertions (i.e. if boxes are `np.ndarray`, all the computation is performed with numpy operations to avoid casting to tensorflow)
    """
    from utils.distance import tf_distance, distance
    
    box1 = convert_box_format(box1, BoxFormat.CORNERS, as_list = False, extended = False, ** kwargs)
    if box2 is None:
        box2, as_matrix = box1, True
    else:
        box2 = convert_box_format(
            box2, BoxFormat.CORNERS, as_list = False, extended = False, ** kwargs
        )
    
    kwargs['box_mode'] = BoxFormat.CORNERS
    
    if use_graph is None: use_graph = as_matrix
    distance_fn = tf_distance if use_graph else distance
        
    return distance_fn(
        box1, box2, method = metric, force_distance = False, as_matrix = as_matrix, ** kwargs
    )

bbox_intersect = partial(bbox_iou, metric = 'intersect')

def crop_box(filename, box, show = False, extended = True, ** kwargs):
    """
        Returns a tuple `(box_image, box_pos)` box `box_pos` is a tuple (x, y, w, h, label, score)
        `box` can be a list of boxes then the result will be a list of the above (single) result
    """
    image = load_image(filename) if isinstance(filename, str) else filename
    image_h, image_w = get_image_size(image)

    positions = convert_box_format(
        box if not tf.executing_eagerly() or not hasattr(box, 'numpy') else box.numpy(),
        BoxFormat.X0Y0X1Y1,
        extended    = extended,
        as_list = tf.executing_eagerly(),
        
        image_h = image_h,
        image_w = image_w,
        normalize_mode = NORMALIZE_WH,
        
        ** kwargs
    )
    
    if tf.executing_eagerly() and _is_box_list(box):
        result = []
        for infos in zip(* positions):
            x1, y1, x2, y2 = infos[:4]
            result.append((infos, image[y1 : y2, x1 : x2]))
        
        if show:
            plot_multiple(** {'box_{}'.format(i) : box_img for i, (_, box_img) in enumerate(result)})
        
        return result

    if tf.executing_eagerly():
        x1, y1, x2, y2 = positions[:4]
    else:
        tf.ensure_shape(positions, [4])
        x1, y1, x2, y2 = positions[0], positions[1], positions[2], positions[3]
    
    if show:
        plot(image[y1 : y2, x1 : x2], plot_type = 'imshow')
    
    return positions, image[y1 : y2, x1 : x2]

def extract_boxes(filename,
                  boxes,
                  image     = None,
                  directory = None,
                  file_format   = '{}_box_{}.jpg',
                  ** kwargs
                 ):
    kwargs['show'] = False
    
    if image is None: image = load_image(filename)
    if hasattr(image, 'numpy'): image = image.numpy()
    
    if directory and isinstance(file_format, str) and not file_format.startswith(directory):
        file_format = os.path.join(directory, file_format)
    
    if isinstance(filename, str):
        basename = os.path.splitext(os.path.basename(filename))[0]
    else:
        if not directory: directory = '.'
        basename = 'image_{}'.format(len(set([
            f.split('_')[1] for f in os.listdir(directory)
            if f.startswith('image_') and '_box_' in f
        ])))
    
    if not isinstance(file_format, (list, tuple)): file_format = [file_format] * len(boxes)
    if len(file_format) != len(boxes):
        raise RuntimeError('{} filenames for {} is incompatible !'.format(
            len(file_format), len(boxes)
        ))
    
    infos = {}
    for i, (file, box) in enumerate(zip(file_format, boxes)):
        box_pos, box_img = crop_box(image, box, ** kwargs)

        if any(s == 0 for s in box_img.shape):
            logger.error('The box has a 0 dimension ({}) : {}'.format(box_img.shape, box_pos))
            continue
        
        if file.count('{}') == 2:
            box_filename    = file.format(basename, i)
        elif '{}' in file:
            box_filename    = file.format(i)
        else:
            box_filename    = file
        save_image(filename = box_filename, image = box_img)
        
        x1, y1, x2, y2 = box_pos[:4]
        infos[box_filename] = {
            'box'   : [x1, y1, x2 - x1, y2 - y1],
            'label' : box_pos[4],
            'height'    : y2 - y1,
            'width'     : x2 - x1
        }
    
    return infos

def draw_boxes(filename,
               boxes,
               
               show_text    = True,
               
               shape    = Shape.RECTANGLE,
               color    = 'r',
               thickness    = 3,
               with_label   = True,
               
               vertical = True,
               ** kwargs
              ):
    shape = get_enum_item(shape, Shape)
    
    image = load_image(filename) if isinstance(filename, str) else filename
    if hasattr(image, 'numpy'): image = image.numpy()
    image_h, image_w, _ = image.shape
    
    if not isinstance(color, list): color = [color]
    color = [normalize_color(c, image = image) for c in color]
    label_color = {}
    
    boxes     = _to_np(boxes)
    positions = get_box_pos(
        boxes, image_h = image_h, image_w = image_w, normalize_mode = NORMALIZE_WH,
        extended = with_label, ** kwargs
    )
    if not _is_box_list(boxes): positions = [[p] for p in positions]
    
    for i, box in enumerate(zip(* positions)):
        if box is None: print(i, box, boxes[i])
        x, y, w, h = box[:4]
        center_x, center_y = int(x + w / 2), int(y + h / 2)
        c = color[i % len(color)]
        
        if with_label:
            label, conf = box[4:6]
            if label not in label_color: 
                label_color[label] = color[len(label_color) % len(color)]
            c = label_color[label]
            
            if show_text:
                prct    = int(conf * 100)
                text    = '{} ({}%)'.format(label, int(prct)) if label else '{}%'.format(prct)
                image   = cv2.putText(
                    image, text, (x, y - 13), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image_h, c, 3
                )
        
        if shape == Shape.RECTANGLE:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), c, thickness)
        elif shape == Shape.CIRCLE:
            image = cv2.circle(image, (center_x, center_y), min(w, h) // 2, c, thickness)
        elif shape == Shape.ELLIPSE:
            axes = (w // 2, int(h / 1.5)) if vertical else (int(w / 1.5), h // 2)
            image = cv2.ellipse(
                image, angle = 0, startAngle = 0, endAngle = 360, 
                center = (center_x, center_y), axes = axes,
                color = c, thickness = thickness
            )
    
    return image

def box_as_mask(filename, boxes, mask_background = False, ** kwargs):
    image = load_image(filename) if isinstance(filename, str) else filename
    if hasattr(image, 'numpy'): image = image.numpy()
    image_h, image_w, _ = image.shape
    
    mask = np.zeros((image_h, image_w, 3), dtype = np.float32)
    
    mask = draw_boxes(mask, boxes, color = 255, thickness = -1, ** kwargs)
    
    mask = mask[...,:1] > 0.
    
    if mask_background: mask = ~mask
    
    return mask

def mask_boxes(filename, boxes, shape = Shape.RECTANGLE, dezoom_factor = 1., ** kwargs):
    image = load_image(filename) if isinstance(filename, str) else filename
    if hasattr(image, 'numpy'): image = image.numpy()
    
    mask    = box_as_mask(image, boxes, shape = shape, dezoom_factor = dezoom_factor)
    
    return apply_mask(image, mask, ** kwargs)

def show_boxes(filename, boxes, labels = None, dezoom_factor = 1., box_mode = BoxFormat.DEFAULT, ** kwargs):
    """
        Displays a (list of) `boxes` with `utils.plot_multiple`
        
        Arguments :
            - filename  : the image (raw or filename)
            - boxes     : the boxes coordinates
            - labels    : the labels for each box
            - dezoom_factor / box_mode  : forwarded to `convert_box_format`
            - kwargs    : forwarded to `plot_multiple`
    """
    image = load_image(filename) if isinstance(filename, str) else filename
    if hasattr(image, 'numpy'): image = image.numpy()
    image_h, image_w = get_image_size(image)
    
    pairs = []
    labels_nb = {}
    
    cropped = crop_box(
        image, boxes, labels = labels, dezoom_factor = dezoom_factor, box_mode = box_mode
    )
    if not isinstance(cropped, list): cropped = [cropped]
    
    for i, ((x1, y1, x2, y2, label, score), box_img) in enumerate(cropped):
        if label not in labels_nb: labels_nb[label] = 0
        labels_nb[label] += 1
        
        if label:
            box_name = "{} #{} ({:.2f} %)".format(label, labels_nb[label], score * 100)
        else:
            box_name = 'Box #{} ({:.2f} %)'.format(i + 1, score * 100)
        
        pairs.append((box_name, box_img))
    
    plot_multiple(* pairs, plot_type = 'imshow', ** kwargs)

def save_boxes(filename, boxes, labels, append = True, ** kwargs):
    """
        Save boxes to a .txt file with format `x y w h label`
        
        Arguments :
            - filename  : the image filename
            - boxes     : list of boxes
            - labels    : labels for boxes
            - append    : whether to overwrite or append at the end of the file
    """
    open_mode = 'a' if append else 'w'
    
    image_w, image_h = get_image_size(filename)

    text = '{}\n{}\n'.format(filename, len(boxes))
    for box in boxes:
        x, y, w, h, label, score = get_box_pos(
            box, image_h = image_h, image_w = image_w, extended = True, labels = labels,
            normalize_mode = NORMALIZE_WH, ** kwargs
        )
        
        text += "{} {} {} {} {}\n".format(x, y, w, h, label)

    with open(filename, open_mode) as file:
        file.write(text)

