
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

import tensorflow as tf

from utils.generic_utils import get_kwargs
from utils.wrapper_utils import dispatch_wrapper
from utils.image.image_utils import resize_image, rotate_image

_image_augmentations_fn = {}
_image_augmentation_kwargs  = {}

def image_augmentation_wrapper(fn = None, nested = None):
    def wrapper(fn, kwargs):
        augment_image.dispatch(fn, fn.__name__)
        

        _image_augmentation_kwargs[fn.__name__] = kwargs
        return fn
    
    if fn is None: return lambda fn: wrapper(fn, get_image_augmentation_config(nested))
    return wrapper(fn, get_kwargs(fn))

def get_image_augmentation_config(methods):
    if not isinstance(methods, (list, tuple)): methods = [methods]
    
    config = {}
    for fn in methods:
        config.update(_image_augmentation_kwargs.get(fn, {}))
    return config

@dispatch_wrapper(_image_augmentations_fn, 'method')
def augment_image(image, method, prct = 0.25, ** kwargs):
    """
        Augments `image` by applying sequentially each `method`, each with `prct` probability
        
        Arguments :
            - image     : `tf.Tensor`, the image to augment
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
        
        image = tf.cond(
            tf.random.uniform((), seed = kwargs.get('seed', None)) <= prct,
            lambda: fn(image, ** kwargs),
            lambda: image
        )
    return image

@image_augmentation_wrapper
def flip_vertical(img, ** kwargs):
    return tf.image.flip_up_down(img)

@image_augmentation_wrapper
def flip_horizontal(img, ** kwargs):
    return tf.image.flip_left_right(img)

@image_augmentation_wrapper
def rot90(img, n = None, seed = None, ** kwargs):
    if n is None: n = tf.random.uniform((), minval = 0, maxval = 4, dtype = tf.int32, seed = seed)
    return tf.image.rot90(img, n)

def zoom(img, min_factor = 0.8, ** kwargs):
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))
    
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        box[i] = [x1, y1, x2, y2]
        
    def random_crop(img):
        crops = tf.image.crop_and_resize([img], boxes = boxes, box_ind = np.zeros(len(scales)), crop_size = (32, 32))
        return crops[tf.random_uniform(shape = [], minval=0, maxval=len(scales), dtype = tf.int32)]
    
    return random_crop(img)

@image_augmentation_wrapper
def noise(img, noise_factor = 1. / 25., clip = True, seed = None, ** kwargs):
    noise = tf.random.normal(tf.shape(img), seed = seed) * noise_factor
    return tf.clip_by_value(img + noise, 0., 1.) if clip else img + noise

@image_augmentation_wrapper
def quality(img, min_jpeg_quality = 25, max_jpeg_quality = 75, seed = None, ** kwargs):
    return tf.image.random_jpeg_quality(img, min_jpeg_quality, max_jpeg_quality, seed = seed)

@image_augmentation_wrapper
def hue(img, max_hue_delta = 0.15, seed = None, ** kwargs):
    return tf.image.random_hue(img, max_hue_delta, seed = seed)

@image_augmentation_wrapper
def saturation(img, lower_saturation = 0.5, upper_saturation = 2., seed = None, ** kwargs):
    return tf.image.random_saturation(img, lower_saturation, upper_saturation, seed = seed)

@image_augmentation_wrapper
def brightness(img, max_brightness_delta = 0.15, clip = True, seed = None, ** kwargs):
    img = tf.image.random_brightness(img, max_brightness_delta, seed = seed)
    return tf.clip_by_value(img, 0., 1.) if clip else img

@image_augmentation_wrapper
def contrast(img, lower_contrast = 0.5, upper_contrast = 1.5, clip = True, seed = None, ** kwargs):
    img = tf.image.random_contrast(img, lower_contrast, upper_contrast, seed = seed)
    return tf.clip_by_value(img, 0., 1.) if clip else img

@image_augmentation_wrapper(nested = ['hue', 'saturation', 'brightness', 'contrast'])
def color(img, ** kwargs):
    img = hue(img, ** kwargs)
    img = saturation(img, ** kwargs)
    img = brightness(img, ** kwargs)
    img = contrast(img, ** kwargs)
    return img

@image_augmentation_wrapper
def random_resize(img, min_scale = 0.5, max_scale = 2., seed = None, ** kwargs):
    factors = tf.random.uniform((2, ), min_scale, max_scale, dtype = tf.float32, seed = seed)
    
    target_shape = tf.cast(tf.shape(img)[-3 : -1], tf.float32)
    target_shape = tf.cast(target_shape * factors, tf.int32)
    
    return resize_image(img, target_shape = target_shape, ** kwargs)

@image_augmentation_wrapper
def random_rotate(img, min_angle = -45., max_angle = 45., seed = None, ** kwargs):
    angle   = tf.random.uniform((), min_angle, max_angle, dtype = tf.float32, seed = seed)
    
    return rotate_image(img, angle, ** kwargs)

def augment_box(box, min_factor = 0.95, max_factor = 1.2, ** kwargs):
    x, y, w, h = tf.unstack(tf.cast(box, tf.float32), axis = -1, num = 4)
    
    factor_x = tf.random.uniform((), min_factor, max_factor, dtype = tf.float32)
    factor_y = tf.random.uniform((), min_factor, max_factor, dtype = tf.float32)
    
    center_x = x + w / 2.
    center_y = y + h / 2.
    
    new_h, new_w = h * factor_y, w * factor_x
    return tf.cast(tf.stack([
        center_x - new_w / 2., center_y - new_h / 2., new_w, new_h
    ], axis = -1), tf.int32)

    