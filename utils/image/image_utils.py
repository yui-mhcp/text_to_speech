
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

BASE_COLORS = list(colors.BASE_COLORS.keys())

def _color_to_rgb(color):
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

def rgb2gray(rgb):
    return np.dot(rgb[...:3], [0.2989, 0.5870, 0.1140])

def normalize_color(color, image = None):
    color = _color_to_rgb(color)
    if image is not None and image.dtype in (np.float32, tf.float32):
        color = color / 255.
    return tuple(color)

@tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
def tf_normalize_color(color : tf.Tensor, image : tf.Tensor = None):
    """ Converts `color` to a 3-items `tf.Tensor` with the same dtype as `image` (if provided, default to `uint8`) """
    color = tf.numpy_function(
        _color_to_rgb, [color], Tout = tf.uint8
    )
    color.set_shape([3])
    
    if image is not None: color = tf.image.convert_image_dtype(color, image.dtype)
    
    return color

def resize_image(image, target_shape, preserve_aspect_ratio = False, ** kwargs):
    """ Reshapes `image` to `target_shape` while possibly preserving aspect ratio """
    image = tf.image.resize(
        image, target_shape[:2], preserve_aspect_ratio = preserve_aspect_ratio, ** kwargs
    )
    if preserve_aspect_ratio:
        pad_h = tf.cast(target_shape[0] - tf.shape(image)[0], tf.int32)
        pad_w = tf.cast(target_shape[1] - tf.shape(image)[1], tf.int32)
        
        half_h, half_w  = pad_h // 2, pad_w // 2
        
        padding = [(half_h, pad_h - half_h), (half_w, pad_w - half_w), (0, 0)]
        
        image   = tf.pad(image, padding)

    return image
