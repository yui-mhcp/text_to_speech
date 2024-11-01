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

from utils.keras_utils import ops, tensorflow_only_function
from utils.wrapper_utils import dispatch_wrapper
from .image_utils import resize_image, rotate_image

_image_augmentations_fn = {}

@dispatch_wrapper(_image_augmentations_fn, 'method')
def augment_image(image, method, prct = 0.25, ** kwargs):
    """
        Augments `image` by applying sequentially each `method`, each with `prct` probability
        
        Arguments :
            - image     : `Tensor`, the image to augment
            - transforms    : (list of) str / callable, augmentation method(s) to apply
            - prct      : the probability to apply each transformation (between [0., 1.])
            - kwargs    : kwargs passed to each transformation function
        Return :
            - transformed : (maybe) transformed image
        
        All functions have an unused `kwargs` argument, which allows to pass kwargs for each transformation function without disturbing other. 
        
        Note that the majority of the available functions are simply the application of 1 or multiple `tf.image.random_*` function
    """
    if not isinstance(method, (list, tuple)): method = [method]
    
    for transform in method:
        if not callable(transform) and transform not in _image_augmentations_fn:
            raise ValueError("Unknown transformation !\n  Accepted : {}\n  Got : {}".format(
                tuple(_image_augmentations_fn.keys()), transform
            ))
        
        fn = transform if callable(transform) else _image_augmentations_fn[transform]
        
        image = ops.cond(
            ops.random.uniform((), seed = kwargs.get('seed', None)) <= prct,
            lambda: fn(image, ** kwargs),
            lambda: image
        )
    return image


@augment_image.dispatch('flip_vertical')
@tensorflow_only_function
def flip_vertical(image, ** kwargs):
    import tensorflow as tf
    return tf.image.flip_up_down(image)

@augment_image.dispatch('flip_horizontal')
@tensorflow_only_function
def flip_horizontal(image, ** kwargs):
    import tensorflow as tf
    return tf.image.flip_left_right(image)

@augment_image.dispatch
@tensorflow_only_function
def rot90(image, n = None, seed = None, ** kwargs):
    import tensorflow as tf
    if n is None: n = tf.random.uniform((), minval = 0, maxval = 4, dtype = tf.int32, seed = seed)
    return tf.image.rot90(image, n)

@augment_image.dispatch
def noise(image, noise_factor = 1. / 25., clip = True, seed = None, ** kwargs):
    result = image + ops.random.normal(ops.shape(image), seed = seed) * noise_factor
    return ops.clip(result, ops.min(image), ops.max(image)) if clip else result

@augment_image.dispatch
def quality(image, min_jpeg_quality = 25, max_jpeg_quality = 75, seed = None, ** kwargs):
    import tensorflow as tf
    return tf.image.random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed = seed)

@augment_image.dispatch
def hue(image, max_hue_delta = 0.15, seed = None, ** kwargs):
    import tensorflow as tf
    return tf.image.random_hue(image, max_hue_delta, seed = seed)

@augment_image.dispatch
def saturation(image, lower_saturation = 0.5, upper_saturation = 2., seed = None, ** kwargs):
    import tensorflow as tf
    return tf.image.random_saturation(image, lower_saturation, upper_saturation, seed = seed)

@augment_image.dispatch
def brightness(image, max_brightness_delta = 0.15, clip = True, seed = None, ** kwargs):
    import tensorflow as tf
    image = tf.image.random_brightness(image, max_brightness_delta, seed = seed)
    return tf.clip_by_value(image, 0., 1.) if clip else image

@augment_image.dispatch
def contrast(image, lower_contrast = 0.5, upper_contrast = 1.5, clip = True, seed = None, ** kwargs):
    import tensorflow as tf
    image = tf.image.random_contrast(image, lower_contrast, upper_contrast, seed = seed)
    return tf.clip_by_value(image, 0., 1.) if clip else image

@augment_image.dispatch
@tensorflow_only_function
def color(image, ** kwargs):
    image = hue(image, ** kwargs)
    image = saturation(image, ** kwargs)
    image = brightness(image, ** kwargs)
    image = contrast(image, ** kwargs)
    return image

@augment_image.dispatch
def resize(image, min_scale = 0.5, max_scale = 2., seed = None, ** kwargs):
    factors = ops.random.randint((2, ), min_scale, max_scale, seed = seed)
    factors = ops.convert_to_numpy(factors, 'float32')
    
    target_shape = ops.convert_to_numpy(ops.shape(image)[-3 : -1], 'float32')
    target_shape = ops.cast(target_shape * factors, 'int32')
    
    return resize_image(image, target_shape = target_shape, ** kwargs)

@augment_image.dispatch
def rotate(image, min_angle = -45., max_angle = 45., seed = None, ** kwargs):
    angle   = ops.cast(ops.random.randint((), min_angle, max_angle, seed = seed), 'float32')
    
    return rotate_image(image, angle, ** kwargs)

def augment_box(box, min_factor = 0.95, max_factor = 1.2, ** kwargs):
    dtype = box.dtype
    
    x, y, w, h = ops.unstack(ops.cast(box, 'float32'), axis = -1, num = 4)
    
    factor_x = ops.random.uniform((), min_factor, max_factor)
    factor_y = ops.random.uniform((), min_factor, max_factor)
    
    center_x = x + w / 2.
    center_y = y + h / 2.
    
    new_h, new_w = h * factor_y, w * factor_x
    return ops.cast(ops.maximum(0., ops.stack([
        center_x - new_w / 2., center_y - new_h / 2., new_w, new_h
    ], axis = -1)), dtype)

    