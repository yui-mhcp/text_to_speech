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

import keras
import numpy as np

from utils.keras_utils import ops
from utils.wrapper_utils import dispatch_wrapper

_clip_means = [0.48145466, 0.4578275, 0.40821073]
_clip_std   = [0.26862954, 0.26130258, 0.27577711]
_east_means = [0.5, 0.5, 0.5]
_east_std   = [0.5, 0.5, 0.5]
_vggface_vals   = [91.4953, 103.8827, 131.0912]
_vgg_means  = np.array([103.939, 116.779, 123.68])[None, None]

def normalize_01(image):
    image = image - ops.reduce_min(image)
    image = image / ops.maximum(1e-3, ops.reduce_max(image))
    return image

def build_mean_normalize(means, std):
    def normalize(image):
        means, std = _means, _std
        if not isinstance(image, np.ndarray):
            means  = ops.convert_to_tensor(means, image.dtype)
            std    = ops.convert_to_tensor(std, image.dtype)

        return (image - means) / std
    
    _means  = np.reshape(means, [-1]).astype(np.float32)[None, None]
    _std    = np.reshape(std,   [-1]).astype(np.float32)[None, None]
    return normalize

def vgg_normalization(image, ** _):
    means = _vgg_means
    if not isinstance(image, np.ndarray):
        means = ops.convert_to_tensor(means, image.dtype)
    return image[..., ::-1] - means

_image_normalization_styles = {
    '01'    : normalize_01,
    'tanh'  : lambda image: image * 2. - 1.,
    'vgg'   : vgg_normalization,
    'vgg16' : vgg_normalization,
    'vgg19' : vgg_normalization,
    'mobilenet' : lambda image: image / 127.5 - 1.,
    'vggface'   : lambda image: image[...,::-1] - ops.reshape(_vggface_vals, [1, 1, 1, 3]) / 255.,
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
    lambda image: (image - ops.mean(image)) / ops.maximum(1e-6, ops.std(image)), ('mean', 'normal')
)

