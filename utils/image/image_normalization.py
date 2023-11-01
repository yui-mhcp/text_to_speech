
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

import numpy as np
import tensorflow as tf

from utils.wrapper_utils import dispatch_wrapper

_clip_means = [0.48145466, 0.4578275, 0.40821073]
_clip_std   = [0.26862954, 0.26130258, 0.27577711]
_east_means = [0.5, 0.5, 0.5]
_east_std   = [0.5, 0.5, 0.5]
_vggface_vals   = [91.4953, 103.8827, 131.0912]

def normalize_01(image):
    image = image - tf.reduce_min(image)
    image = image / tf.maximum(1e-3, tf.reduce_max(image))
    return image

def build_mean_normalize(means, std):
    def normalize(image):
        expanded_means  = tf.reshape(means, [1] * (len(image.shape) - 1)  + [-1])
        expanded_std    = tf.reshape(std, [1] * (len(image.shape) - 1)  + [-1])
        return (image - expanded_means) / expanded_std
    
    means   = np.reshape(means, [-1]).astype(np.float32)
    std     = np.reshape(std,   [-1]).astype(np.float32)
    return normalize

_image_normalization_styles = {
    '01'    : normalize_01,
    'tanh'  : lambda image: image * 2. - 1.,
    'vgg'   : tf.keras.applications.vgg16.preprocess_input,
    'vgg16' : tf.keras.applications.vgg16.preprocess_input,
    'vgg19' : tf.keras.applications.vgg19.preprocess_input,
    'mobilenet' : tf.keras.applications.mobilenet.preprocess_input,
    'vggface'   : lambda image: image[...,::-1] - tf.reshape(_vggface_vals, [1, 1, 1, 3]) / 255.,
    'clip'      : build_mean_normalize(_clip_means, _clip_std),
    'east'      : build_mean_normalize(_east_means, _east_std),
    'easyocr'   : build_mean_normalize(0.5, 0.5)
}


@dispatch_wrapper(_image_normalization_styles, 'method')
def get_image_normalization_fn(method):
    """ Returns the normalization function associated to `method` """
    if callable(method):        return method
    if isinstance(method, dict): return build_mean_normalize(** method)
    if isinstance(method, (list, tuple)): return build_mean_normalize(* method)
    
    if method not in _image_normalization_styles:
        raise ValueError('Unknown normalization method ! Accepted : {}\n  Got : {}'.format(
            tuple(_image_normalization_styles.keys()), method
        ))
    
    return _image_normalization_styles[method]

get_image_normalization_fn.dispatch(lambda image: image, (None, 'null', 'identity'))
get_image_normalization_fn.dispatch(
    lambda image: (image - tf.reduce_mean(image)) / tf.maximum(1e-6, tf.math.reduce_std(image)),
    ('mean', 'normal')
)

