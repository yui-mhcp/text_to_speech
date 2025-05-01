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

import inspect
import numpy as np

from functools import wraps

from loggers import timer
from ...keras import ops
from ..image_io import get_image_size

_keys_to_convert    = ('boxes', 'scores', 'classes', 'angles')

_required_image_error_message = 'You must either provide `image` or (`image_h` and `image_w`)'

class RequiresImagesException(Exception):
    def __init__(self, message = _required_image_error_message, ** kwargs):
        super().__init__(message, ** kwargs)

def box_converter_wrapper(target, as_dict = None, force_np = False, force_tensor = False, ** kw):
    """
        Automatically converts all arguments of the decorated function that starts with `boxes` to the expected format / normalization
        
        Arguments :
            - target    : the target format expected by the function
            - as_dict   : if `False`, will only pass the `boxes` key if a `dict` was provided
            - normalize : the `BoxNormalization` to apply
            - force_np / force_tensor   : whether to force `numpy / Tensor` convertion
            - force_dict    : whether to create a `dict` if raw boxes were provided
    """
    def wrapper(fn):
        @wraps(fn)
        def inner(* args, ** kwargs):
            kwargs.update(kw)
            kwargs.update({name : arg for name, arg in zip(_argnames, args)})
            
            source = kwargs.pop('source', None)
            for k in list(kwargs.keys()):
                if (k.startswith('boxes')) and kwargs[k] is not None:
                    boxes = kwargs.pop(k)
                    if force_np:        boxes = box_to_numpy(boxes)

                    boxes = convert_box_format(
                        boxes, target = target, source = source, ** {** kwargs, ** kw}
                    )
                    if as_dict is False and isinstance(boxes, dict):
                        boxes = boxes['boxes']
                    elif as_dict is True and not isinstance(boxes, dict):
                        boxes = {'boxes' : boxes, 'format' : target}
                    
                    if force_tensor:    boxes = box_to_tensor(boxes)

                    kwargs[k] = boxes
            
            return fn(** kwargs)
        
        _argnames = list(inspect.signature(fn).parameters.keys())
        
        return inner
    
    if force_np and force_tensor:
        raise ValueError('`force_np` and `force_tensor` cannot be both `True`')
    return wrapper

@timer
def convert_box_format(boxes,
                       target   = None,
                       source   = None,
                       as_list  = False,
                       
                       dezoom_factor    = 1.,
                       normalize_mode   = None,
                       
                       dtype    = None,
                       
                       ** kwargs
                      ):
    """
        Convert a bounding box format to another format
        
        Arguments :
            - boxes     : a valid box format (`ndarray` or `Tensor` with `boxes.shape[-1] == 4`)
            - source    : a valid box format, the input `boxes` format
            - target    : a valid box format, the expected output format
            
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
    elif source is None:
        raise ValueError('You must provide the `source` format')
    
    if target is None: target = source

    if as_list and target == 'poly':
        raise ValueError('The `as_list = True` is not supported for `POLYGON` format')
    
    if isinstance(boxes, list):     boxes = np.array(boxes)
    if boxes.shape[0] == 0:         return boxes if not as_list else []
    
    if (len(boxes.shape) == 1) or (source == 'poly' and len(boxes.shape) == 2):
        boxes = boxes[None]
    
    if normalize_mode is not None:
        boxes = normalize_boxes(boxes, source, normalize_mode, dtype = dtype, ** kwargs)
    elif dtype is not None:
        boxes = ops.cast(boxes, dtype)
    
    if source == target and dezoom_factor == 1.:
        return boxes if not as_list else ops.unstack(boxes, axis = -1, num = 4)
    
    if source == 'xywh':
        x_min, y_min, w, h = ops.unstack(boxes, axis = -1, num = 4)
    elif source == 'xyxy':
        x_min, y_min, x_max, y_max = ops.unstack(boxes, axis = -1, num = 4)
        w, h = x_max - x_min, y_max - y_min
    elif source == 'poly':
        xy_min, xy_max = ops.min(boxes, axis = -2), ops.max(boxes, axis = -2)
        x_min, y_min = ops.unstack(xy_min, axis = -1, num = 2)
        w, h = ops.unstack(xy_max - xy_min, axis = -1, num = 2)
    else:
        raise ValueError('Invalid `source` format : {}'.format(source))
    
    if dezoom_factor != 1.:
        x_min, y_min, w, h = dezoom_box(
            x_min, y_min, w, h, dezoom_factor, ** kwargs
        )

    if target == 'xywh':
        result = [x_min, y_min, w, h]
    elif target == 'xyxy':
        result = [x_min, y_min, x_min + w, y_min + h]
    elif target == 'poly':
        x_max, y_max = x_min + w, y_min + h
        result = [
            ops.stack([x_min, y_min], axis = -1),
            ops.stack([x_max, y_min], axis = -1),
            ops.stack([x_min, y_max], axis = -1),
            ops.stack([x_max, y_max], axis = -1)
        ]
    else:
        raise ValueError('Invalid `target` format : {}'.format(target))

    return result if as_list else ops.stack(result, axis = -1 if target != 'poly' else -2)

def normalize_boxes(boxes, source, normalize_mode, dtype = None, ** kwargs):
    assert normalize_mode in ('relative', 'absolute'), 'Got {}'.format(normalize_mode)
    
    rel = _is_relative(boxes)
    if (rel and normalize_mode == 'absolute') or (not rel and normalize_mode == 'relative'):
        if dtype is None: dtype = 'int32' if normalize_mode == 'absolute' else 'float32'
        image_h, image_w = _get_image_size_from_kwargs(kwargs)
        
        if source == 'xyxy' or source == 'xywh':
            factors = ops.convert_to_numpy([image_w, image_h, image_w, image_h], boxes.dtype)
        elif source == 'poly':
            factors = ops.convert_to_numpy([image_w, image_h], boxes.dtype)
        
        if normalize_mode == 'relative':
            boxes = ops.divide(ops.cast(boxes, dtype), ops.cast(factors, dtype))
        else:
            boxes = ops.cast(ops.multiply(boxes, factors), dtype)
    elif dtype is not None:
        boxes = ops.cast(boxes, dtype)
    
    return boxes

def dezoom_box(x, y, w, h, factor, angle = 0., ** kwargs):
    if factor == 1.:    return (x, y, w, h)
    if angle != 0.:     raise NotImplementedError('Rotated box is not supported yet !')
    
    dtype   = x.dtype
    rel     = _is_relative(x)
    if not rel: x, y, w, h = [ops.cast(coord, 'float32') for coord in (x, y, w, h)]
    
    if ops.is_tensor(x): factor = ops.convert_to_tensor(factor, x.dtype)
    
    new_h, new_w = h * factor, w * factor
    
    new_x = ops.maximum((x + w / 2.) - new_w / 2., 0)
    new_y = ops.maximum((y + h / 2.) - new_h / 2., 0)
    
    if rel:
        new_h = ops.minimum(new_h, 1. - new_y)
        new_w = ops.minimum(new_w, 1. - new_x)
    else:
        image_h, image_w = _get_image_size_from_kwargs(kwargs)
        
        new_x, new_y, new_w, new_h = [ops.cast(coord, dtype) for coord in (new_x, new_y, new_w, new_h)]
        
        new_h = ops.minimum(new_h, image_h - new_y)
        new_w = ops.minimum(new_w, image_w - new_x)
    
    return new_x, new_y, new_w, new_h

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

def _is_relative(boxes):
    return ops.is_float(boxes['boxes'] if isinstance(boxes, dict) else boxes)

def _is_absolute(boxes):
    return ops.is_int(boxes['boxes'] if isinstance(boxes, dict) else boxes)

def _get_image_size_from_kwargs(kwargs, raise_exception = True):
    if 'image_h' in kwargs and 'image_w' in kwargs: return (kwargs['image_h'], kwargs['image_w'])
    elif 'image_shape' in kwargs:   return kwargs['image_shape'][:2]
    elif 'image' in kwargs:         return get_image_size(kwargs['image'])
    elif 'filename' in kwargs:      return get_image_size(kwargs['filename'])
    elif raise_exception:           raise RequiresImagesException()
    return (None, None)
