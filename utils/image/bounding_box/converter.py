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

import enum
import numpy as np

from functools import wraps

from loggers import timer
from utils.keras_utils import ops
from utils.wrapper_utils import args_to_kwargs
from utils.generic_utils import get_enum_item
from ..image_io import get_image_size

_keys_to_convert    = ('boxes', 'scores', 'classes', 'angles')

_required_image_error_message = 'You must either provide `image` or (`image_h` and `image_w`)'

class RequiresImagesException(Exception):
    def __init__(self, message = _required_image_error_message, ** kwargs):
        super().__init__(message, ** kwargs)

class BoxFormat(enum.IntEnum):
    UNCHANGED   = -2
    DEFAULT     = 0
    
    XYWH        = 0
    X0Y0WH      = 0
    
    XYXY        = 1
    CORNERS     = 1
    X0Y0X1Y1    = 1
    
    YXYX        = 2
    CORNERS2    = 2
    Y0X0Y1X1    = 2
    
    POLY        = 3
    POLYGON     = 3

class BoxNormalization(enum.IntEnum):
    NONE        = -1
    STANDARDIZE = 0
    RELATIVE    = 1
    ABSOLUTE    = 2
    IMAGE       = 2
    WH          = 2

NORMALIZE_NONE  = BoxNormalization.NONE
NORMALIZE_STANDARDIZE   = BoxNormalization.STANDARDIZE
NORMALIZE_01    = NORMALIZE_RELATIVE    = BoxNormalization.RELATIVE
NORMALIZE_WH    = NORMALIZE_IMAGE       = BoxNormalization.IMAGE

def box_converter_wrapper(target,
                          as_dict   = None,
                          as_list   = False,
                          normalize = NORMALIZE_NONE,
                          dtype     = None,
                          
                          force_np  = False,
                          force_tensor  = False,
                          force_dict    = False,
                          expand_dict   = False
                         ):
    """
        Automatically converts all arguments of the decorated function that starts with `boxes` to the expected format / normalization
        
        Arguments :
            - target    : the target `BoxFormat` expected by the function
            - as_dict   : if `False`, will only pass the `boxes` key if a `dict` was provided
            - normalize : the `BoxNormalization` to apply
            - force_np / force_tensor   : whether to force `numpy / Tensor` convertion
            - force_dict    : whether to create a `dict` if raw boxes were provided
            - expand_dict   : whether to provide each key as separate kwarg if `dict` was provided
    """
    def wrapper(fn):
        @wraps(fn)
        @args_to_kwargs(fn)
        def inner(** kwargs):
            for k in list(kwargs.keys()):
                if k.startswith('boxes') and (
                    isinstance(kwargs[k], (list, dict, np.ndarray)) or ops.is_tensor(kwargs[k])):
                    boxes = kwargs.pop(k)
                    if force_np:        boxes = box_to_numpy(boxes)
                    if force_tensor:    boxes = box_to_tensor(boxes)

                    boxes = convert_box_format(
                        boxes,
                        target  = target,
                        as_list = as_list,
                        dtype   = dtype,
                        normalize_mode = normalize,
                        ** kwargs
                    )
                    if isinstance(boxes, dict):
                        if as_dict is False: boxes = boxes['boxes']
                        elif expand_dict:
                            if k != 'boxes':
                                raise ValueError(
                                    'The `expand_dict` is only supported for the `boxes` argument'
                                )
                            
                            for key in _keys_to_convert:
                                if key in boxes: kwargs[key] = boxes[key]
                            continue
                    elif force_dict:
                        boxes = {'boxes' : boxes}
                    kwargs[k] = boxes
            
            kwargs.pop('source', None)
            return fn(** kwargs)
        return inner
    
    if force_np and force_tensor:
        raise ValueError('`force_np` and `force_tensor` cannot be both `True`')
    if as_dict is False and (force_dict or expand_dict):
        raise ValueError('`as_dict = False` and `force_dict = True` are incompatible')
    return wrapper

@timer
def convert_box_format(boxes,
                       target   = BoxFormat.UNCHANGED,
                       source   = BoxFormat.DEFAULT,
                       as_list  = False,
                       
                       dezoom_factor    = 1.,
                       normalize_mode   = NORMALIZE_NONE,
                       
                       dtype    = None,
                       
                       ** kwargs
                      ):
    """
        Convert a bounding box format to another format
        
        Arguments :
            - boxes     : a valid box format (`ndarray` or `Tensor` with `boxes.shape[-1] == 4`)
            - source    : a valid `BoxFormat`, the input `boxes` format
            - target    : a valid `BoxFormat`, the expected output format
            
            - dezoom_factor : the factor to multiply width / height of the box
            - normalize_mode    : one of `BoxNormalization`
                - NONE  : does not normalize the boxes, no assumption on values can be done
                - STANDARIZE    : normalizes the boxes in a valid range based on its `dtype`
                    - float will clip in range [0, 1], and integer in the range [0, MAX_{W/H}]
                - RELATIVE      : converts the image to float values in the range [0, 1]
                - IMAGE         : converts the image to integer in the range [0, MAX_{W/H}]
        Return :
            - converted_box : the same as `boxes` with the new format
        
        Note : when normalizing the image, it may be required to provide image size information
               you can either provide `image`, `image_shape` (w, h) or `image_h + image_w`

    """
    if isinstance(boxes, dict):
        boxes = boxes.copy()
        boxes['boxes'] = convert_box_format(
            boxes['boxes'],
            target  = target,
            source  = boxes.get('format', source),
            as_list = as_list,
            dtype   = dtype,
            
            dezoom_factor   = dezoom_factor,
            normalize_mode  = normalize_mode,
            ** kwargs
        )
        boxes['format']   = target
        return boxes
    
    if isinstance(source, str): source = get_enum_item(source, BoxFormat)
    if isinstance(target, str): target = get_enum_item(target, BoxFormat)
    if target == BoxFormat.UNCHANGED: target = source

    if as_list and target == BoxFormat.POLY:
        raise ValueError('The `as_list = True` is not supported for `POLYGON` format')
    
    if ops.executing_eagerly() and len(boxes) == 0: return boxes
    if isinstance(boxes, list):     boxes = np.array(boxes)
    if len(boxes.shape) == 1 or (source == BoxFormat.POLY and len(boxes.shape) == 2):
        boxes = boxes[None]
    
    if _is_valid_format(source, target) and dezoom_factor == 1. and not _should_scale(boxes, normalize_mode):
        if dtype is not None: boxes = ops.cast(boxes, dtype)
        return boxes if not as_list else ops.unstack(boxes, axis = -1, num = 4)
    
    if source == BoxFormat.XYWH:
        x_min, y_min, w, h = ops.unstack(boxes, axis = -1, num = 4)
    elif source == BoxFormat.XYXY:
        x_min, y_min, x_max, y_max = ops.unstack(boxes, axis = -1, num = 4)
        w, h = x_max - x_min, y_max - y_min
    elif source == BoxFormat.YXYX:
        y_min, x_min, y_max, x_max = ops.unstack(boxes, axis = -1, num = 4)
        w, h = x_max - x_min, y_max - y_min
    elif source == BoxFormat.POLY:
        xy_min, xy_max = ops.min(boxes, axis = -2), ops.max(boxes, axis = -2)
        x_min, y_min = ops.unstack(xy_min, axis = -1, num = 2)
        w, h = ops.unstack(xy_max - xy_min, axis = -1, num = 2)
    else:
        raise ValueError('Invalid `source` format : {}'.format(source))
    
    if dezoom_factor != 1.:
        x_min, y_min, w, h = dezoom_box(
            x_min, y_min, w, h, dezoom_factor, dtype = dtype, ** kwargs
        )
    
    x_min, y_min, w, h, dtype = normalize_box(x_min, y_min, w, h, normalize_mode, ** kwargs)

    if isinstance(target, (list, tuple)): target = target[0]
    if target == BoxFormat.XYWH:
        result = [x_min, y_min, w, h]
    elif target == BoxFormat.XYXY:
        result = [x_min, y_min, x_min + w, y_min + h]
    elif target == BoxFormat.YXYX:
        result = [y_min, x_min, y_min + h, x_min + w]
    elif target == BoxFormat.POLY:
        x_max, y_max = x_min + w, y_min + h
        result = [
            ops.stack([x_min, y_min], axis = -1),
            ops.stack([x_max, y_min], axis = -1),
            ops.stack([x_min, y_max], axis = -1),
            ops.stack([x_max, y_max], axis = -1)
        ]
    else:
        raise ValueError('Invalid `target` format : {}'.format(target))

    if as_list: return [ops.cast(coord, dtype) for coord in result]
    axis = -1 if target != BoxFormat.POLY else -2
    return ops.cast(ops.stack(result, axis = axis), dtype)

def standardize_boxes(boxes):
    if not isinstance(boxes, dict): boxes = {'boxes' : boxes}
    boxes['boxes'] = ops.convert_to_numpy(boxes['boxes'])
    if len(boxes['boxes'].shape) == 1: boxes['boxes'] = boxes['boxes'][None]
    return boxes

def box_to_numpy(boxes):
    if isinstance(boxes, dict):
        boxes = boxes.copy()
        for k in _keys_to_convert:
            if k in boxes: boxes[k] = ops.convert_to_numpy(boxes[k])
        return boxes
    return ops.convert_to_numpy(boxes)

def box_to_tensor(boxes):
    if isinstance(boxes, dict):
        boxes = boxes.copy()
        for k in _keys_to_convert:
            if k in boxes: boxes[k] = ops.convert_to_tensor(boxes[k])
        return boxes
    return ops.convert_to_tensor(boxes)

def is_relative(boxes):
    return ops.is_float(boxes['boxes'] if isinstance(boxes, dict) else boxes)

def is_batched(boxes):
    return ops.shape(boxes['boxes'] if isinstance(boxes, dict) else boxes) >= 3

def dezoom_box(x, y, w, h, factor, angle = 0., ** _):
    if factor == 1.:    return (x, y, w, h)
    if angle != 0.:     raise NotImplementedError('Rotated box is not supported yet !')
    
    dtype   = x.dtype
    rel     = is_relative(x)
    if not rel: x, y, w, h = [ops.cast(coord, 'float32') for coord in (x, y, w, h)]
    
    if ops.is_tensor(x): factor = ops.convert_to_tensor(factor, x.dtype)
    
    new_h, new_w = h * factor, w * factor
    new_x = (x + w / 2.) - new_w / 2.
    new_y = (y + h / 2.) - new_h / 2.
    
    new_box = [new_x, new_y, new_w, new_h]
    return new_box if rel else [ops.cast(coord, dtype) for coord in new_box]

def normalize_box(x, y, w, h, normalize_mode, dtype = None, ** kwargs):
    """
        Normalizes the `[x, y, w, h]` coordinates to `normalize_mode`
            - NORMALIZE_NONE    : no normalization
            - NORMALIZE_01      : normalizes to the `[0, 1]` range (i.e. returns `float`)
            - NORMALIZE_WH      : normalizes to the dimension of the images (i.e. returns `int`)
    """
    if normalize_mode == NORMALIZE_NONE: return (x, y, w, h, dtype or x.dtype)
    
    rel = is_relative(x)

    image_h, image_w = None, None
    if rel:
        max_x, max_y = 1., 1.
    else:
        max_y, max_x = _get_image_size_from_kwargs(kwargs, raise_exception = False)
    
    x   = ops.clip(x, 0, max_x)
    y   = ops.clip(y, 0, max_y)
    if max_x is not None: w   = ops.clip(max_x - x, 0, w)
    if max_y is not None: h   = ops.clip(max_y - y, 0, h)
    
    if not _should_scale(x, normalize_mode):
        return (x, y, w, h, dtype or x.dtype)

    if image_h is None:
        image_h, image_w = _get_image_size_from_kwargs(kwargs)

    if ops.is_tensor(x):
        image_h, image_w = ops.cast(image_h, 'float32'), ops.cast(image_w, 'float32')
    if rel: # normalize_mode == NORMALIZE_WH
        return (x * image_w, y * image_h, w * image_w, h * image_h, dtype or 'int32')
    x, y, w, h = [ops.cast(coord, 'float32') for coord in (x, y, w, h)]
    return (x / image_w, y / image_h, w / image_w, h / image_h, dtype or 'float32')

def _is_valid_format(source, target):
    if isinstance(target, (list, tuple)): return source in target
    return source == target

def _should_scale(box, normalize_mode):
    """ Returns whether `box` should be normalized to `normalize_mode` or not """
    if normalize_mode in (NORMALIZE_NONE, BoxNormalization.STANDARDIZE): return False
    if is_relative(box):
        return normalize_mode == NORMALIZE_WH
    return normalize_mode == NORMALIZE_01

def _get_image_size_from_kwargs(kwargs, raise_exception = True):
    if 'image_h' in kwargs and 'image_w' in kwargs: return (kwargs['image_h'], kwargs['image_w'])
    elif 'image_shape' in kwargs:   return kwargs['image_shape']
    elif 'image' in kwargs:         return get_image_size(kwargs['image'])
    elif 'filename' in kwargs:      return get_image_size(kwargs['filename'])
    elif raise_exception:           raise RequiresImagesException()
    return (None, None)
