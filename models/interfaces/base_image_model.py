
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
import pandas as pd
import tensorflow as tf

from hparams import HParams
from utils import infer_downsampling_factor, infer_upsampling_factor, pad_to_multiple
from utils.image import load_image, augment_image, pad_image, get_image_augmentation_config
from models.interfaces.base_model import BaseModel

_default_augmentations = ['hue', 'brightness', 'saturation', 'contrast', 'noise']

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
    means   = np.reshape(means, [1, 1, 1, -1])
    std     = np.reshape(std,   [1, 1, 1, -1])
    return lambda image: (image - means) / std

_no_normalization   = (None, 'null', '01', 'identity')
_image_normalization_styles = {
    None    : lambda image: image,
    "null"  : lambda image: image,
    '01'    : normalize_01,
    "mean"  : lambda image: (image - tf.reduce_mean(image)) / tf.maximum(1e-3, tf.math.reduce_std(image)),
    "identity"  : lambda image: image,
    'tanh'  : lambda image: image * 2. - 1.,
    'vgg'   : lambda image: tf.keras.applications.vgg16.preprocess_input(image),
    'vgg16' : lambda image: tf.keras.applications.vgg16.preprocess_input(image),
    'vgg19' : lambda image: tf.keras.applications.vgg19.preprocess_input(image),
    'vggface'   : lambda image: image[...,::-1] - tf.reshape(_vggface_vals, [1, 1, 1, 3]) / 255.,
    'mobilenet' : lambda image: tf.keras.applications.mobilenet.preprocess_input(image),
    'clip'      : build_mean_normalize(_clip_means, _clip_std),
    'east'      : build_mean_normalize(_east_means, _east_std),
    'easyocr'   : build_mean_normalize(0.5, 0.5)
}

ImageTrainingHParams    = HParams(
    image_augmentation_methods = []
)

class BaseImageModel(BaseModel):
    def _init_image(self,
                    input_size,
                    max_image_size  = None,
                    resize_method   = 'resize',
                    resize_kwargs   = {},
                    image_normalization     = None,
                    image_normalization_fn  = None,
                    ** kwargs
                   ):
        if image_normalization_fn is None and image_normalization not in _image_normalization_styles:
            raise ValueError('Unknown normalization style !\n  Accepted : {}\n  Got : {}'.format(
                tuple(_image_normalization_styles.keys()), image_normalization
            ))

        if resize_method not in ('resize', 'pad'):
            raise ValueError('Unknown resizing method !\n  Accepted : (resize, pad)\n  Got : {}'.format(resize_method))
        
        if not isinstance(input_size, (list, tuple)): input_size = (input_size, input_size, 3)
        
        self.input_size = tuple(input_size)
        self.resize_method  = resize_method
        self.resize_kwargs  = resize_kwargs
        self.max_image_size = None if not self.has_variable_input_size else max_image_size
        self.image_normalization    = image_normalization
        
        if image_normalization_fn is None:
            image_normalization_fn = _image_normalization_styles[image_normalization]
        self.image_normalization_fn = image_normalization_fn
        self._downsampling_factor   = None
        self._upsampling_factor     = None
    
    @property
    def max_image_shape(self):
        if not self.has_variable_input_size or self.max_image_size in (-1, None): return None
        if isinstance(self.max_image_size, (list, tuple)): return self.max_image_size
        return (self.max_image_size, self.max_image_size)
    
    @property
    def has_fixed_input_size(self):
        return any(size not in (None, -1) for size in self.input_size[:2])
    
    @property
    def has_variable_input_size(self):
        return any(size in (None, -1) for size in self.input_size)
    
    @property
    def should_pad_to_multiple(self):
        if not self.has_variable_input_size: return False
        return isinstance(self.upsampling_factor, np.ndarray) and np.any(self.upsampling_factor > 1)
    
    @property
    def upsampling_factor(self):
        if self._upsampling_factor is None:
            self._upsampling_factor = infer_upsampling_factor(self.model)
        return self._upsampling_factor

    @property
    def downsampling_factor(self):
        if self._downsampling_factor is None:
            self._downsampling_factor = infer_downsampling_factor(self.model)
        return self._downsampling_factor

    @property
    def downscale_factor(self):
        return self.downsampling_factor // self.upsampling_factor
    
    @property
    def image_signature(self):
        return tf.TensorSpec(shape = (None, ) + self.input_size, dtype = tf.float32)
    
    @property
    def default_image_augmentation(self):
        return _default_augmentations if self.input_size[-1] == 3 else ['noise']

    @property
    def image_augmentation_config(self):
        return {
            k : getattr(self, k) for k, v in self.get_image_augmentation_config().items()
            if hasattr(self, k)
        }
    
    @property
    def training_hparams_image(self):
        config = {}
        if self.has_variable_input_size:
            config['max_image_size'] = None
        
        config['image_augmentation_methods'] = self.default_image_augmentation
        return ImageTrainingHParams(** config)

    def _str_image(self):
        if self.max_image_shape:
            des = "- Image size : {}\n".format('({})'.format(', '.join(
                '<= {}'.format(max_s) if s is None else str(s)
                for s, max_s in zip(self.input_size, self.max_image_shape)
            )))
        else:
            des = "- Image size : {}\n".format(self.input_size)
        if self.resize_kwargs: des += "- Resize config : {}\n".format(self.resize_kwargs)
        des += '- Normalization style : {}\n'.format(self.image_normalization)
        return des
    
    def get_image_augmentation_config(self, hparams = None):
        if hparams is not None:
            methods = hparams.image_augmentation_methods
        else:
            methods = self.image_augmentation_methods
        return {
            k : v for k, v in get_image_augmentation_config(methods).items()
            if k not in ('clip', 'seed')
        }
    
    def pad_to_multiple(self, image):
        factors = self.downsampling_factor
        return pad_to_multiple(image, factors, axis = list(range(1, len(factors) + 1)))
    
    def get_image(self, filename, use_box = False, ** kwargs):
        """ Calls `utils.image.load_image` on the given `filename` """
        if isinstance(filename, (list, tuple)):
            return [self.get_image(f, use_box, ** kwargs) for f in filename]
        elif isinstance(filename, pd.DataFrame):
            return [self.get_image(row, use_box, ** kwargs) for idx, row in filename.iterrows()]
        
        bbox_kwargs = {}
        if isinstance(filename, (dict, pd.Series)):
            if use_box and 'box' in filename:
                bbox_kwargs['bbox'] = filename['box']
                if 'box_mode' in filename: bbox_kwargs['box_mode'] = filename['box_mode']
            filename = filename['image'] if 'image' in filename else filename['filename']
        
        tar_shape, tar_max_shape, tar_mul_shape = None, None, None
        if self.resize_method == 'resize':
            if self.has_variable_input_size:
                if any(s not in (-1, None) for s in self.input_size[:-1]):
                    tar_shape   = tuple(s if s is not None else -1 for s in self.input_size[:2])
                tar_max_shape   = self.max_image_shape
                if self.should_pad_to_multiple:
                    tar_mul_shape   = self.downsampling_factor
            else:
                tar_shape       = self.input_size
        else:
            if self.has_fixed_input_size:
                tar_max_shape   = self.input_size
            else:
                tar_max_shape   = self.max_image_shape

        return load_image(
            filename,
            dtype   = tf.float32,
            mode    = 'gray' if self.input_size[-1] == 1 else None,
            target_shape    = tar_shape,
            target_max_shape    = tar_max_shape,
            target_multiple_shape   = tar_mul_shape,
            resize_kwargs   = self.resize_kwargs,
            ** bbox_kwargs,
            ** kwargs
        )
    
    def get_image_with_box(self, filename, ** kwargs):
        return self.get_image(filename, use_box = True, ** kwargs)

    def normalize_image(self, image, ** kwargs):
        """ Normalizes a (batch of) image by calling the normalization schema """
        return self.image_normalization_fn(image)

    def filter_image(self, image):
        return tf.reduce_all(tf.shape(image) > 0)
    
    def augment_image(self, image, ** kwargs):
        if self.has_variable_input_size:
            kwargs.setdefault('resize_kwargs',      self.resize_kwargs)
            kwargs.setdefault('target_max_shape',   self.max_image_shape)
            if self.should_pad_to_multiple:
                kwargs.setdefault('target_multiple_shape', self.downsampling_factor)
        elif self.resize_method != 'resize':
            kwargs.setdefault('target_max_shape', self.input_size)

        return augment_image(
            image,
            self.image_augmentation_methods,
            self.augment_prct / len(self.image_augmentation_methods),
            ** {** self.image_augmentation_config, ** kwargs}
        )
    
    def preprocess_image(self, image, ** kwargs):
        """ Normalizes a (batch of) image by calling the normalization schema """
        image = self.normalize_image(image, ** kwargs)
        if self.resize_method == 'pad':
            image = pad_image(
                image,
                target_shape    = self.input_size if not self.has_variable_input_size else None,
                target_multiple_shape   = self.downsampling_factor if self.should_pad_to_multiple else None,
                ** self.resize_kwargs
            )
        
        return image
    
    def get_config_image(self, * args, ** kwargs):
        config = {
            'input_size' : self.input_size,
            'resize_method' : self.resize_method,
            'resize_kwargs' : self.resize_kwargs,
            'image_normalization'   : self.image_normalization
        }
        if self.has_variable_input_size:
            config['max_image_size'] = self.max_image_size
        
        return config
