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

import numpy as np

from matplotlib import colors
from keras.layers import RandomRotation

from loggers import timer
from utils.keras_utils import TensorSpec, ops, graph_compile, execute_eagerly

BASE_COLORS = list(colors.BASE_COLORS.keys())

@execute_eagerly(signature = TensorSpec(shape = (3, ), dtype = 'uint8'), numpy = True)
def color_to_rgb(color):
    """
        Returns a RGB np.ndarray color as uint8 values. `color` can be of different types :
            - str (or bytes)  : the color's name (as supported by `matplotlib.colors.to_rgb`)
            - int / float     : the color's value (used as Red, Green and Blue value)
            - 3-tuple / array : the RGB values (either float or int)
    """
    if isinstance(color, bytes): color = color.decode()
    if colors.is_color_like(color):
        color = colors.to_rgb(color)

    if not isinstance(color, (list, tuple, np.ndarray)): color = (color, color, color)
    if isinstance(color[0], (float, np.floating)): color = [c * 255 for c in color]
    return np.array(color, dtype = np.uint8)

def normalize_color(color, image = None, dtype = None):
    color = color_to_rgb(color)
    if image is not None: dtype = image.dtype.name
    if dtype is not None: color = ops.convert_data_dtype(color, dtype)
    return color

@timer
@graph_compile(support_xla = False)
def resize_image(image,
                 
                 target_shape   = None,
                 target_min_shape   = None,
                 target_max_shape   = None,
                 target_multiple_shape  = None,
                 
                 method = 'bilinear',
                 antialias  = False,
                 preserve_aspect_ratio  = False,
                 
                 ** kwargs
                ):
    """
        Resizes `image` to the given shape while possibly preserving aspect ratio + padding
        
        Arguments :
            - image : 3-D or 4-D Tensor, the image(s) to resize
            - target_shape  : tuple (h, w), the fixed expected output shape
            - target_min_shape  : tuple (h, w), the minimum dimension for the output shape
            - target_max_shape  : tuple (h, w), the maximal dimension for the output shape
            - target_multiple_shape : the output shape should be a multiple of this argument
            
            - method / antialias / preserve_aspect_ratio : kwargs for `K.image.resize`
            - kwargs    : propagated to `get_resized_shape` and to `pad_image` (if `preserve_aspect_ratio == True`)
        Return :
            - resized   : the resized image(s) with same rank / dtype as `image`
                if `target_shape` is provided:
                    `shape(resized)[-3 : -1] == target_shape`
                if `target_max_shape` is provided:
                    `reduce_all(shape(resized)[-3 : -1])) <= target_max_shape`
                if `target_multiple_shape` is provided:
                    `shape(resized)[-3 : -1] % target_multiple_shape == [0, 0]`
        
        /!\ WARNING /!\ if multiple `target_{}_shape` are provided, make sure that they are consistant with each other. Otherwise, some assertions may be False !
        /!\ WARNING /!\ If any of `target_shape` or `shape(image)` is 0, the function directly returns the image without resizing !
    """
    img_shape = ops.convert_to_numpy(ops.shape(image))[-3 : -1]
    if ops.any(img_shape == 0): return image
    
    target_shape = get_resized_shape(
        image,
        shape   = target_shape,
        min_shape   = target_min_shape,
        max_shape   = target_max_shape,
        multiples   = target_multiple_shape,
        preserve_aspect_ratio   = preserve_aspect_ratio,
        ** kwargs
    )

    if isinstance(target_shape, tuple) or ops.any(img_shape != target_shape):
        image   = ops.convert_to_tensor(image)

        resized_shape = target_shape
        if preserve_aspect_ratio:
            ratio       = ops.reduce_min(ops.divide(target_shape, img_shape))
            resized_shape = ops.cast(ops.cast(img_shape, ratio.dtype) * ratio, 'int32')

        image = ops.image_resize(
            image,
            resized_shape,
            antialias   = antialias,
            interpolation   = method
        )

        if preserve_aspect_ratio:
            image = pad_image(image, target_shape, ** kwargs)

    return image

def pad_image(image,
              target_shape  = None,
              target_min_shape  = None,
              target_multiple_shape = None,
              
              pad_mode  = 'after',
              pad_value = 0.,
              
              ** kwargs
             ):
    """
        Pads `image` to the expected shape
        
        Arguments :
            - image : 3D or 4D `Tensor`, the image(s) to pad
            - target_shape  : fixed expected output shape
            - target_min_shape  : tuple (h, w), the minimum dimension for the output shape
            - target_multiple_shape : the output shape should be a multiple of this argument
            
            - pad_mode  : where to add padding (one of `after`, `before`, `even`)
            - pad_value : the value to add
            
            - kwargs    : propagated to `get_resized_shape`
        Return :
            - resized   : the resized image(s) with same rank / dtype as `image`
                if `target_shape` is provided:
                    `shape(resized)[-3 : -1] == target_shape`
                if `target_multiple_shape` is provided:
                    `shape(resized)[-3 : -1] % target_multiple_shape == [0, 0]`
        
        /!\ WARNING /!\ if both are provided, it is possible that the 1st assertion will be False
        /!\ WARNING /!\ If any of `target_shape` or `shape(image)` is 0, the function directly returns the image without resizing !
    """
    img_shape    = ops.convert_to_numpy(ops.shape(image))
    if ops.any(img_shape == 0): return image
    
    target_shape = get_resized_shape(
        image, 
        target_shape,
        min_shape   = target_min_shape,
        multiples   = target_multiple_shape,
        prefer_crop = False,
        ** kwargs
    )
    
    pad_h = ops.maximum(0, target_shape[0] - img_shape[-3])
    pad_w = ops.maximum(0, target_shape[1] - img_shape[-2])
    if pad_h > 0 or pad_w > 0:
        # torch backend does not support np.int padding
        if ops.executing_eagerly(): pad_h, pad_w = int(pad_h), int(pad_w)
        padding = None
        if pad_mode == 'before':
            padding = [(pad_h, 0), (pad_w, 0), (0, 0)]
        elif pad_mode == 'after':
            padding = [(0, pad_h), (0, pad_w), (0, 0)]
        elif pad_mode == 'even':
            half_h, half_w  = pad_h // 2, pad_w // 2
            padding = [(half_h, pad_h - half_h), (half_w, pad_w - half_w), (0, 0)]
        elif pad_mode == 'repeat_last':
            batch_axis = [1] if len(ops.shape(image)) == 4 else []
            if pad_w > 0:
                image = ops.concat([
                    image, ops.tile(image[..., -1:, :], batch_axis + [1, pad_w, 1])
                ], axis = -2)
            if pad_h > 0:
                image = ops.concat([
                    image, ops.tile(image[..., -1:, :, :], batch_axis + [pad_h, 1, 1])
                ], axis = -3)
        else:
            raise ValueError('Unknown padding mode : {}'.format(pad_mode))
        
        if padding is not None:
            if len(ops.shape(image)) == 4: padding = [(0, 0)] + padding
            image   = ops.pad(image, padding, constant_values = pad_value)

    return image

def get_resized_shape(image,
                      
                      shape = None,
                      min_shape = None,
                      max_shape = None,
                      multiples = None,
                      
                      prefer_crop   = True,
                      preserve_aspect_ratio = False,
                      
                      ** kwargs
                     ):
    """
        Computes the expected output shape after possible transformation
        
        Arguments :
            - image : 3-D or 4-D `Tensor`, the image to resize
            - shape : tuple (h, w), the expected output shape (if `None`, set to `shape(image)`)
            - min_shape : tuple (h, w), the minimal dimension for the output shape
            - max_shape : tuple (h, w), the maximal dimension for the outputshape
            - multiples : tuple (h, w), the expected multiple for the output shape
                i.e. `output_shape % multiples == [0, 0]`
            - prefer_crop   : whether to take the lower / upper multiple (ignored if `multiples` is not provided)
            - kwargs    : /
        Return :
            - output_shape  : the expected new shape for the image
    """
    if isinstance(shape, tuple) and all(s != -1 for s in shape):
        if any(arg is not None for arg in (min_shape, max_shape, multiples)):
            raise ValueError('When providing a fixed target shape, other arguments are not allowed')
        
        return shape[:2]
    
    img_shape   = ops.convert_to_numpy(ops.shape(image), 'int32')[-3 : -1]
    shape   = ops.convert_to_numpy(shape, 'int32')[:2] if shape is not None else img_shape

    if ops.any(shape == -1):
        if not preserve_aspect_ratio or ops.all(shape == -1):
            shape   = ops.where(shape != -1, shape, img_shape)
        else:
            ratio   = ops.max(shape / img_shape)
            shape   = ops.cast(ops.cast(img_shape, ratio.dtype) * ratio, shape.dtype)

    if max_shape is not None:
        max_shape   = ops.convert_to_numpy(max_shape, dtype = 'int32')
        if not preserve_aspect_ratio:
            shape   = ops.where(max_shape != -1, ops.minimum(shape, max_shape), shape)
        elif ops.any(ops.logical_and(max_shape != -1, img_shape > max_shape)):
            ratio   = ops.reduce_min(ops.where(max_shape == -1, img_shape, max_shape) / img_shape)
            shape   = ops.cast(ops.cast(img_shape, ratio.dtype) * ratio, shape.dtype)
    
    if min_shape is not None:
        min_shape   = ops.convert_to_numpy(min_shape, dtype = 'int32')
        if not preserve_aspect_ratio:
            shape   = ops.where(min_shape != -1, ops.maximum(shape, min_shape), shape)
        elif ops.reduce_any(ops.logical_and(min_shape != -1, img_shape < min_shape)):
            ratio   = ops.reduce_max(min_shape / img_shape)
            shape   = ops.cast(ops.cast(img_shape, ratio.dtype) * ratio, shape.dtype)
    
    if multiples is not None:
        multiples   = ops.convert_to_numpy(multiples, dtype = 'int32')
        if prefer_crop and not preserve_aspect_ratio:
            shape   = shape // multiples * multiples
        else:
            shape   = ops.where(
                shape % multiples != 0, (shape // multiples + 1) * multiples, shape
            )
    
    return shape

def rotate_image(image,
                 angle,
                 fill_mode  = 'constant',
                 fill_value = 0.,
                 interpolation  = 'bilinear',
                 ** kwargs
                ):
    """
        Rotates an image of `angle` degrees clock-wise (i.e. positive value rotates clock-wise)
        
        Arguments :
            - image : 3D or 4D `Tensor`, the image(s) to rotate
            - angle : scalar or 1D `Tensor`, the angle(s) (in degree) to rotate the image(s)
            - fill_mode : the mode of filling values outside of the boundaries
            - fill_value    : filling value (only if `fill_mode = 'constant'`)
            - interpolation : the interpolation method
    """
    if not isinstance(angle, tuple): angle = (- angle / 360., - angle / 360.)
    
    rotation = RandomRotation(
        factor  = angle,
        fill_mode   = fill_mode,
        fill_value  = fill_value,
        interpolation   = interpolation
    )
    return rotation(image)
