
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
from utils.image import load_image, augment_image, augment_box, pad_image, get_image_augmentation_config, get_image_normalization_fn
from models.interfaces.base_model import BaseModel

_default_augmentations = ['hue', 'brightness', 'saturation', 'contrast', 'noise']

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
        if resize_method not in ('resize', 'pad'):
            raise ValueError('Unknown resizing method !\n  Accepted : (resize, pad)\n  Got : {}'.format(resize_method))
        
        if not isinstance(input_size, (list, tuple)): input_size = (input_size, input_size, 3)
        
        self.input_size = tuple(input_size)
        self.resize_method  = resize_method
        self.resize_kwargs  = resize_kwargs
        self.max_image_size = None if not self.has_variable_input_size else max_image_size
        self.image_normalization    = image_normalization
        
        if image_normalization_fn is None:
            image_normalization_fn = get_image_normalization_fn(image_normalization)
        
        self.image_normalization_fn = image_normalization_fn
        self._downsampling_factor   = None
        self._upsampling_factor     = None
    
    @property
    def target_image_shape(self):
        return tuple(s if s is not None else -1 for s in self.input_size)
    
    @property
    def max_image_shape(self):
        if not self.has_variable_input_size or self.max_image_size in (-1, None): return None
        max_shape = self.max_image_size
        if not isinstance(max_shape, (list, tuple)): max_shape = (max_shape, max_shape)
        max_shape = [
            max_s if s in (None, -1) else s for s, max_s in zip(self.input_size, max_shape)
        ]
        return tuple(s if s is not None else -1 for s in max_shape)
    
    @property
    def has_fixed_input_size(self):
        return any(size not in (None, -1) for size in self.input_size[:2])
    
    @property
    def has_variable_input_size(self):
        return any(size in (None, -1) for size in self.input_size[:2])
    
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
                if 'box_mode' in filename:
                    bbox_kwargs['box_mode'] = filename['box_mode']
                if 'dezoom_factor' in filename:
                    bbox_kwargs['dezoom_factor'] = filename['dezoom_factor']
            
            filename = filename['image' if 'image' in filename else 'filename']
        
        # Case 1 : not has_variable_input_size -> tar_shape = input_size
        # Case 2 : all input sizes are variable -> max_shape = max_image_shape
        # Case 3 : input size is semi-variable : tar_shape = input_size, max_shape = max_image_shape
        tar_shape, tar_max_shape, tar_mul_shape = None, None, None
        if self.resize_method == 'resize':
            tar_shape = self.target_image_shape
            max_shape = self.max_image_shape
            if self.should_pad_to_multiple:
                tar_mul_shape = self.downsampling_factor
        else:
            max_shape = self.max_image_shape

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
        img_shape   = tf.shape(image)
        
        good_shape  = True
        if self.has_fixed_input_size and self.resize_method == 'resize':
            target_shape = tf.cast(self.target_image_shape[:2], img_shape.dtype)
            good_shape = tf.reduce_all(tf.logical_or(
                target_shape == -1, img_shape[:2] == target_shape[:2]
            ))

        small_enough = True
        if self.has_variable_input_size and self.max_image_shape is not None:
            max_shape       = tf.cast(self.max_image_shape, img_shape.dtype)
            small_enough    = tf.reduce_all(tf.logical_or(
                max_shape == -1, img_shape[:2] <= max_shape
            ))
        
        return tf.logical_and(
            tf.reduce_all(img_shape > 0),
            tf.logical_and(small_enough, good_shape)
        )
    
    def augment_box(self, data, ** kwargs):
        if 'box' in data:
            data['box'] = tf.cond(
                tf.random.uniform((), 0., 1.) <= self.augment_prct,
                lambda: augment_box(data['box'], ** kwargs),
                lambda: data['box']
            )
        return data
    
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
                target_shape    = self.target_image_shape if self.has_fixed_input_size else None,
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
