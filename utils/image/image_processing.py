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

from loggers import timer
from ..keras import TensorSpec, ops, graph_compile, execute_eagerly

@timer
@graph_compile(support_xla = False)
def resize_image(image  : TensorSpec(),
                 size   = None,
                 *,
                 
                 multiples  : TensorSpec(dtype = 'int32')   = None,
                 
                 antialias  = False,
                 interpolation  = 'bilinear',
                 preserve_aspect_ratio  = False,
                 
                 pad_value  = 0.,
                 pad_mode   = 'after',
                 
                 method = None
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
            - resized   : the resized image(s) with same rank as `image` and `float` dtype
    """
    if size is None and multiples is None: return image
    if method is not None: interpolation = method
    
    img_size    = ops.shape(image)[-3 : -1]
    output_size = get_output_size(
        image,
        size    = size,
        multiples   = multiples,
        preserve_aspect_ratio   = preserve_aspect_ratio
    )
    
    _is_graph = ops.is_tensorflow_graph()
    if _is_graph or img_size[0] != output_size[0] or img_size[1] != output_size[1]:
        image   = ops.convert_to_tensor(image)

        intermediate_size = output_size
        if preserve_aspect_ratio:
            img_size    = ops.convert_to_numpy(img_size, 'float32')
            ratio       = ops.min(ops.divide(ops.cast(output_size, 'float32'), img_size))
            intermediate_size = ops.cast(img_size * ratio, 'int32')

        image = ops.image_resize(
            image, intermediate_size, antialias = antialias, interpolation = interpolation
        )

        if preserve_aspect_ratio:
            image = pad_image(image, output_size, pad_value = pad_value, pad_mode = pad_mode)
            if isinstance(size, tuple) and _is_graph:
                shape   = size + (image.shape[-1], )
                if len(image.shape) == 4: shape = (image.shape[0], ) + shape
                image = ops.ensure_shape(image, shape)

    return image

def pad_image(image, size, pad_mode = 'after', pad_value = 0):
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
        
        **WARNING** if both are provided, it is possible that the 1st assertion will be False
        **WARNING** If any of `target_shape` or `shape(image)` is 0, the function directly returns the image without resizing !
    """
    
    pad_h = ops.maximum(0, size[0] - ops.shape(image)[-3])
    pad_w = ops.maximum(0, size[1] - ops.shape(image)[-2])
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
            batch_axis = [1] if len(image.shape) == 4 else []
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
            if len(image.shape) == 4: padding = [(0, 0)] + padding
            image   = ops.pad(image, padding, constant_values = pad_value)

    return image

def get_output_size(image,
                    size    = None,
                    *,
                    
                    multiples   = None,
                    preserve_aspect_ratio = False
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
    if isinstance(size, tuple):
        assert len(size) == 2, 'Invalid size : {}'.format(size)
        if size[0] and size[1]:     return size
        elif size[0] or size[1]:    size = (size[0] or -1, size[1] or -1)
        else:   size = None
    
    img_size    = ops.convert_to_numpy(ops.shape(image), 'int32')[-3 : -1]
    if size is None:
        out_size = img_size
    else:
        out_size = size = ops.convert_to_numpy(size, 'int32')

    if ops.any(out_size == -1):
        if not preserve_aspect_ratio:
            out_size    = ops.where(out_size != -1, out_size, img_size)
        else:
            ratio   = ops.max(out_size / img_size)
            out_size    = ops.cast(ops.cast(img_size, ratio.dtype) * ratio, 'int32')
    
    if multiples is not None:
        multiples   = ops.convert_to_numpy(multiples, dtype = 'int32')
        out_size    = ops.where(
            out_size % multiples != 0, (out_size // multiples + 1) * multiples, out_size
        )
    
    if size is not None:
        out_size = ops.where(size != -1, size, out_size)
    
    return out_size

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
    from keras.layers import RandomRotation

    if not isinstance(angle, tuple): angle = (- angle / 360., - angle / 360.)
    
    rotation = RandomRotation(
        factor  = angle,
        fill_mode   = fill_mode,
        fill_value  = fill_value,
        interpolation   = interpolation
    )
    return rotation(image)
