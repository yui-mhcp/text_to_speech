
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

def augment_image(img, transforms, prct = 0.25, ** kwargs):
    """
        Augment `img` by applying sequentially each `transforms`, each with `prct` probability
        
        Arguments :
            - img       : the image to augment
            - transforms    : (list of) str / callable, transformations to apply
            - prct      : the probability to apply each transformation (between [0., 1.])
            - kwargs    : kwargs passed to each transformation function
        Return :
            - transformed : (maybe) transformed image
        
        Supported transformations' names are in the `_image_augmentations_fn` variable which associate name with function.
        All functions have an unused `kwargs` argument which allows to pass kwargs for each transformation function without disturbing other. 
        
        Note that the majority of these available functions are simply the application of 1 or multiple `tf.image.random_*` function
    """
    if not isinstance(transforms, (list, tuple)): transforms = [transforms]
    for transfo in transforms:
        if not callable(transfo) and transfo not in _image_augmentations_fn:
            raise ValueError("Unknown transformation !\n  Accepted : {}\n  Got : {}".format(
                tuple(_image_augmentations_fn.keys()), transfo
            ))
        
        fn = transfo if callable(transfo) else _image_augmentations_fn[transfo]
        
        img = tf.cond(
            tf.random.uniform((), seed = kwargs.get('seed', None)) < prct,
            lambda: fn(img, ** kwargs),
            lambda: img
        )
    return img

def flip_vertical(img, ** kwargs):
    return tf.image.flip_up_down(img)

def flip_horizontal(img, ** kwargs):
    return tf.image.flip_left_right(img)

def rotate(img, n = None, seed = None, ** kwargs):
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

def noise(img, factor = 25., seed = None, ** kwargs):
    noise = tf.random.normal(tf.shape(img), seed = seed) / factor
    return tf.clip_by_value(img + noise, 0., 1.)

def quality(img, min_jpeg_quality = 25, max_jpeg_quality = 75, seed = None, ** kwargs):
    return tf.image.random_jpeg_quality(img, min_jpeg_quality, max_jpeg_quality, seed = seed)

def color(img, ** kwargs):
    img = hue(img, ** kwargs)
    img = saturation(img, ** kwargs)
    img = brightness(img, ** kwargs)
    img = contrast(img, ** kwargs)
    return img

def hue(img, max_delta = 0.15, seed = None, ** kwargs):
    return tf.image.random_hue(img, max_delta, seed = seed)

def saturation(img, lower = 0.5, upper = 2., seed = None, ** kwargs):
    return tf.image.random_saturation(img, lower, upper, seed = seed)

def brightness(img, max_delta = 0.15, seed = None, ** kwargs):
    return tf.clip_by_value(tf.image.random_brightness(img, max_delta, seed = seed), 0., 1.)

def contrast(img, lower = 0.5, upper = 1.5, seed = None, ** kwargs):
    return tf.clip_by_value(tf.image.random_contrast(img, lower, upper, seed = seed), 0., 1.)


_image_augmentations_fn = {
    'flip_vertical'     : flip_vertical,
    'flip_horizontal'   : flip_horizontal,
    'rotate'            : rotate,
    
    'noise'     : noise,
    'quality'   : quality,
    
    'color'     : color,
    'hue'       : hue,
    'saturation'    : saturation,
    'brightness'    : brightness,
    'contrast'      : contrast
}
