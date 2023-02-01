
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

import pandas as pd
import tensorflow as tf

from hparams import HParams
from utils.image import load_image, augment_image
from models.interfaces.base_model import BaseModel


_default_augmentations = ['hue', 'brightness', 'saturation', 'contrast', 'noise']

_clip_means = [0.48145466, 0.4578275, 0.40821073]
_clip_std   = [0.26862954, 0.26130258, 0.27577711]
_vggface_vals   = [91.4953, 103.8827, 131.0912]

_image_normalization_styles = {
    None    : lambda image: image,
    "null"  : lambda image: image,
    "identity"  : lambda image: image,
    'tanh'  : lambda image: image * 2. - 1.,
    'vgg'   : lambda image: tf.keras.applications.vgg.preprocess_input(image),
    'vggface'   : lambda image: image[...,::-1] - tf.reshape(_vggface_vals, [1, 1, 1, 3]) / 255.,
    'mobilenet' : lambda image: tf.keras.applications.mobilenet.preprocess_input(image),
    'clip'  : lambda image: (image - tf.reshape(_clip_means, [1, 1, 1, 3])) / tf.reshape(_clip_std, [1, 1, 1, 3])
}

ImageTrainingHParams    = HParams(
    augment_methods = _default_augmentations
)

class BaseImageModel(BaseModel):
    def _init_image(self, input_size, * args, image_normalization = None,
                    resize_kwargs = {}, ** kwargs):
        if image_normalization not in _image_normalization_styles:
            raise ValueError('Unknown normalization style !\n  Accepted : {}\n  Got : {}'.format(
                tuple(_image_normalization_styles.keys()), image_normalization
            ))

        if not isinstance(input_size, (list, tuple)): input_size = (input_size, input_size, 3)
        
        self.input_size = tuple(input_size)
        self.resize_kwargs  = resize_kwargs
        self.image_normalization    = image_normalization
        
        self.image_normalization_fn = _image_normalization_styles[image_normalization]
    
    @property
    def image_signature(self):
        return tf.TensorSpec(shape = (None, ) + self.input_size, dtype = tf.float32)
    
    @property
    def training_hparams_image(self):
        return ImageTrainingHParams(
            augment_methods = _default_augmentations if self.input_size[-1] == 3 else ['noise']
        )

    def _str_image(self):
        des = "- Image size : {}\n".format(self.input_size)
        if self.resize_kwargs: des += "- Resize config : {}\n".format(self.resize_kwargs)
        des += '- Normalization style : {}\n'.format(self.image_normalization)
        return des
    
    def get_image(self, filename):
        if isinstance(filename, (list, tuple)):
            return tf.stack([self.get_image(f) for f in filename])
        elif isinstance(filename, pd.DataFrame):
            return tf.stack([self.get_image(row) for idx, row in filename.iterrows()])
        
        if isinstance(filename, (dict, pd.Series)):
            filename = filename['image'] if 'image' in filename else filename['filename']
        
        return load_image(
            filename, target_shape = self.input_size, dtype = tf.float32, resize_kwargs = self.resize_kwargs
        )
    
    def augment_image(self, image):
        return augment_image(
            image, self.augment_methods, self.augment_prct / len(self.augment_methods)
        )
    
    def preprocess_image(self, image):
        return self.image_normalization_fn(image)
    
    def get_config_image(self, * args, ** kwargs):
        return {
            'input_size' : self.input_size,
            'image_normalization'   : self.image_normalization
        }
