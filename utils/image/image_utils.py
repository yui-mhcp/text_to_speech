
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

import cv2
import numpy as np
import tensorflow as tf

from matplotlib import colors
try:
    from keras.layers.preprocessing.image_preprocessing import transform, get_rotation_matrix
except:
    from keras.src.layers.preprocessing.image_preprocessing import transform, get_rotation_matrix

from utils.tensorflow_utils import tf_compile, execute_eagerly

BASE_COLORS = list(colors.BASE_COLORS.keys())

@execute_eagerly(signature = tf.TensorSpec(shape = (3, ), dtype = tf.uint8), numpy = True)
def color_to_rgb(color):
    """
        Returns a RGB np.ndarray color as uint8 values. `color` can be of different types :
            - str (or bytes)  : the color's name (as supported by `matplotlib.colors.to_rgb`)
            - int / float     : the color's value (used as Red, Green and Blue value)
            - 3-tuple / array : the RGB values (either float or int)
    """
    if isinstance(color, tf.Tensor): color = color.numpy()
    if isinstance(color, bytes): color = color.decode()
    if colors.is_color_like(color):
        color = colors.to_rgb(color)

    if not isinstance(color, (list, tuple, np.ndarray)): color = (color, color, color)
    if isinstance(color[0], (float, np.floating)): color = [c * 255 for c in color]
    return np.array(color, dtype = np.uint8)

def rgb2gray(rgb):
    return np.dot(rgb[...:3], [0.2989, 0.5870, 0.1140])

def normalize_color(color, image = None):
    color = color_to_rgb(color)
    if image is not None and image.dtype in (np.float32, tf.float32):
        return tuple(float(ci) / 255. for ci in color)
    return tuple(int(ci) for ci in color)

@tf_compile(reduce_retracing = True, experimental_follow_type_hints = True)
def tf_normalize_color(color : tf.Tensor, image : tf.Tensor = None):
    """ Converts `color` to a 3-items `tf.Tensor` with the same dtype as `image` (if provided, default to `uint8`) """
    color = color_to_rgb(color)
    
    if image is not None: color = tf.image.convert_image_dtype(color, image.dtype)
    
    return color

def resize_image(image,
                 target_shape   = None,
                 target_min_shape   = None,
                 target_max_shape   = None,
                 target_multiple_shape  = None,
                 
                 method = 'bilinear',
                 antialias  = False,
                 preserve_aspect_ratio  = False,
                 manually_compute_ratio = False,
                 
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
            
            - method / antialias / preserve_aspect_ratio : kwargs for `tf.image.resize`
            - kwargs    : propagated to `get_resized_shape` and to `pad_image` (if `preserve_aspect_ratio == True`)
        Return :
            - resized   : the resized image(s) with same rank / dtype as `image`
                if `target_shape` is provided:
                    `tf.shape(resized)[-3 : -1] == target_shape`
                if `target_max_shape` is provided:
                    `tf.reduce_all(tf.shape(resized)[-3 : -1])) <= target_max_shape`
                if `target_multiple_shape` is provided:
                    `tf.shape(resized)[-3 : -1] % target_multiple_shape == [0, 0]`
        
        /!\ WARNING /!\ if multiple `target_{}_shape` are provided, make sure that they are consistant with each other. Otherwise, some assertions may be False !
        /!\ WARNING /!\ If any of `target_shape` or `tf.shape(image)` is 0, the function directly returns the image without resizing !
    """
    if not tf.executing_eagerly() and tf.reduce_any(tf.shape(image) <= 0):  return image
    elif tf.executing_eagerly() and any(s == 0 for s in image.shape):       return image
    
    target_shape = get_resized_shape(
        image,
        shape   = target_shape,
        min_shape   = target_min_shape,
        max_shape   = target_max_shape,
        multiples   = target_multiple_shape,
        preserve_aspect_ratio   = preserve_aspect_ratio,
        ** kwargs
    )

    if (
        (image.shape[0] != target_shape[0] or image.shape[1] != target_shape[1])
        and tf.reduce_all(target_shape > 0)
        and (tf.executing_eagerly() or tf.reduce_all(tf.shape(image) > 0))
        and (not tf.executing_eagerly() or all(s > 0 for s in image.shape))):
        should_pad    = preserve_aspect_ratio
        resized_shape = target_shape
        if preserve_aspect_ratio and manually_compute_ratio:
            img_shape   = tf.shape(image)[-3 : -1]
            ratio       = tf.reduce_min(target_shape / img_shape)
            resized_shape = tf.cast(tf.cast(img_shape, ratio.dtype) * ratio, target_shape.dtype)
            preserve_aspect_ratio   = False

        image = tf.image.resize(
            image,
            resized_shape,
            method  = method,
            antialias   = antialias,
            preserve_aspect_ratio   = preserve_aspect_ratio
        )

        if should_pad:
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
            - image : 3D or 4D `tf.Tensor`, the image(s) to pad
            - target_shape  : fixed expected output shape
            - target_min_shape  : tuple (h, w), the minimum dimension for the output shape
            - target_multiple_shape : the output shape should be a multiple of this argument
            
            - pad_mode  : where to add padding (one of `after`, `before`, `even`)
            - pad_value : the value to add
            
            - kwargs    : propagated to `get_resized_shape`
        Return :
            - resized   : the resized image(s) with same rank / dtype as `image`
                if `target_shape` is provided:
                    `tf.shape(resized)[-3 : -1] == target_shape`
                if `target_multiple_shape` is provided:
                    `tf.shape(resized)[-3 : -1] % target_multiple_shape == [0, 0]`
        
        /!\ WARNING /!\ if both are provided, it is possible that the 1st assertion will be False
        /!\ WARNING /!\ If any of `target_shape` or `tf.shape(image)` is 0, the function directly returns the image without resizing !
    """
    target_shape = get_resized_shape(
        image, 
        target_shape,
        min_shape   = target_min_shape,
        multiples   = target_multiple_shape,
        prefer_crop = False,
        ** kwargs
    )
    if tf.reduce_any(target_shape <= 0) or tf.reduce_any(tf.shape(image) <= 0): return image
    
    pad_h = tf.maximum(0, target_shape[0] - tf.shape(image)[-3])
    pad_w = tf.maximum(0, target_shape[1] - tf.shape(image)[-2])
    if pad_h > 0 or pad_w > 0:
        padding = None
        if pad_mode == 'before':
            padding = [(pad_h, 0), (pad_w, 0), (0, 0)]
        elif pad_mode == 'after':
            padding = [(0, pad_h), (0, pad_w), (0, 0)]
        elif pad_mode == 'even':
            half_h, half_w  = pad_h // 2, pad_w // 2
            padding = [(half_h, pad_h - half_h), (half_w, pad_w - half_w), (0, 0)]
        elif pad_mode == 'repeat_last':
            batch_axis = [1] if len(tf.shape(image)) == 4 else []
            if pad_w > 0:
                image = tf.concat([
                    image, tf.tile(image[..., -1:, :], batch_axis + [1, pad_w, 1])
                ], axis = -2)
            if pad_h > 0:
                image = tf.concat([
                    image, tf.tile(image[..., -1:, :, :], batch_axis + [pad_h, 1, 1])
                ], axis = -3)
        else:
            raise ValueError('Unknown padding mode : {}'.format(pad_mode))
        
        if padding is not None:
            if len(tf.shape(image)) == 4:
                padding = [(0, 0)] + padding

            image   = tf.pad(image, padding, constant_values = pad_value)

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
            - image : 3-D or 4-D `tf.Tensor`, the image to resize
            - shape : tuple (h, w), the expected output shape (if `None`, set to `tf.shape(image)`)
            - min_shape : tuple (h, w), the minimal dimension for the output shape
            - max_shape : tuple (h, w), the maximal dimension for the outputshape
            - multiples : tuple (h, w), the expected multiple for the output shape
                i.e. `output_shape % multiples == [0, 0]`
            - prefer_crop   : whether to take the lower / upper multiple (ignored if `multiples` is not provided)
            - kwargs    : /
        Return :
            - output_shape  : the expected new shape for the image
    """
    if shape is None:
        shape = tf.shape(image)[-3 : -1]

    shape = tf.cast(shape, tf.int32)[:2]

    if tf.reduce_any(shape == -1):
        img_shape   = tf.shape(image)[-3 : -1]
        if not preserve_aspect_ratio or tf.reduce_all(shape == -1):
            shape   = tf.where(shape != -1, shape, img_shape)
        else:
            ratio   = tf.reduce_max(shape / img_shape)
            shape   = tf.cast(tf.cast(img_shape, ratio.dtype) * ratio, shape.dtype)

    if max_shape is not None:
        img_shape   = tf.shape(image)[-3 : -1]
        max_shape   = tf.cast(max_shape, shape.dtype)[:2]
        if not preserve_aspect_ratio:
            shape   = tf.where(max_shape != -1, tf.minimum(shape, max_shape), shape)
        elif tf.reduce_any(tf.math.logical_and(max_shape != -1, img_shape > max_shape)):
            ratio   = tf.reduce_min(tf.where(max_shape == -1, img_shape, max_shape) / img_shape)
            shape   = tf.cast(tf.cast(img_shape, ratio.dtype) * ratio, shape.dtype)
    
    if min_shape is not None:
        img_shape   = tf.shape(image)[-3 : -1]
        min_shape   = tf.cast(min_shape, shape.dtype)[:2]
        if not preserve_aspect_ratio:
            shape   = tf.where(min_shape != -1, tf.maximum(shape, min_shape), shape)
        elif tf.reduce_any(tf.math.logical_and(min_shape != -1, img_shape < min_shape)):
            ratio   = tf.reduce_max(min_shape / img_shape)
            shape   = tf.cast(tf.cast(img_shape, ratio.dtype) * ratio, shape.dtype)
    
    if multiples is not None:
        multiples   = tf.cast(multiples, shape.dtype)
        if prefer_crop and not preserve_aspect_ratio:
            shape   = shape // multiples * multiples
        else:
            shape   = tf.where(shape % multiples != 0, (shape // multiples + 1) * multiples, shape)
    
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
            - image : 3D or 4D `tf.Tensor`, the image(s) to rotate
            - angle : scalar or 1D `tf.Tensor`, the angle(s) (in degree) to rotate the image(s)
            - fill_mode : the mode of filling values outside of the boundaries
            - fill_value    : filling value (only if `fill_mode = 'constant'`)
            - interpolation : the interpolation method
    """
    angle = tf.cast(- angle / 360. * 2. * np.pi, tf.float32)
    
    dim = len(tf.shape(image))
    if dim == 3:
        image = tf.expand_dims(image, axis = 0)
    if len(tf.shape(angle)) == 0:
        angle = tf.expand_dims(angle, axis = 0)
    
    image = transform(
        image,
        get_rotation_matrix(
            angle, tf.cast(tf.shape(image)[-3], tf.float32), tf.cast(tf.shape(image)[-2], tf.float32)
        ),
        fill_mode   = fill_mode,
        fill_value  = fill_value,
        interpolation   = interpolation
    )
    return image if dim == 4 else image[0]
