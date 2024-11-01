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

import numpy as np

from functools import cached_property

from utils import *
from utils.image import *
from utils.keras_utils import TensorSpec, ops
from models.utils import infer_downsampling_factor, infer_upsampling_factor
from .base_model import BaseModel

_default_image_augmentation = ['noise', 'brightness', 'contrast', 'hue', 'saturation']

TrainingHParamsImage    = HParams(
    augmentation_method = None
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
    
    @property
    def target_image_shape(self):
        return tuple(s if s is not None else -1 for s in self.input_size)
    
    @property
    def max_image_shape(self):
        if not self.has_variable_input_size or self.max_image_size is None: return None
        max_shape = self.max_image_size
        if not isinstance(max_shape, (list, tuple)): max_shape = (max_shape, max_shape)
        return tuple(s if s is not None else -1 for s in max_shape)
    
    @property
    def has_fixed_input_size(self):
        return all(self.input_size)
    
    @property
    def has_variable_input_size(self):
        return not self.has_fixed_input_size
    
    @property
    def should_pad_to_multiple(self):
        if not self.has_variable_input_size: return False
        return isinstance(self.upsampling_factor, np.ndarray) and np.any(self.upsampling_factor > 1)
    
    @cached_property
    def upsampling_factor(self):
        return infer_upsampling_factor(self.model)

    @cached_property
    def downsampling_factor(self):
        return infer_downsampling_factor(self.model)

    @property
    def downscale_factor(self):
        return self.downsampling_factor // self.upsampling_factor
    
    @property
    def image_signature(self):
        return TensorSpec(shape = (None, ) + self.input_size, dtype = 'float')
    
    @property
    def default_image_augmentation(self):
        return _default_image_augmentation if self.input_size[-1] == 3 else ['noise']
    
    @property
    def training_hparams_image(self):
        config = {'augmentation_method' : self.default_image_augmentation}
        if self.has_variable_input_size: config['max_image_size'] = None
        
        return TrainingHParamsImage(** config)

    def _str_image(self):
        if self.has_variable_input_size and self.max_image_shape is not None:
            des = "- Image size : {}\n".format('({})'.format(', '.join(
                '<= {}'.format(max_s) if s is None else str(s)
                for s, max_s in zip(self.input_size, self.max_image_shape)
            )))
        else:
            des = "- Image size : {}\n".format(self.input_size)
        if self.resize_kwargs: des += "- Resize config : {}\n".format(self.resize_kwargs)
        des += '- Normalization schema : {}\n'.format(self.image_normalization)
        return des
    
    def pad_to_multiple(self, image):
        factors = self.downsampling_factor
        return pad_to_multiple(image, factors, axis = list(range(1, len(factors) + 1)))
    
    def get_image(self, filename, use_box = False, ** kwargs):
        """ Calls `utils.image.load_image` on the given `filename` """
        if isinstance(filename, (list, tuple)):
            return [self.get_image(f, use_box, ** kwargs) for f in filename]
        elif is_dataframe(filename):
            return [self.get_image(row, use_box, ** kwargs) for idx, row in filename.iterrows()]
        
        bbox_kwargs = {}
        if isinstance(filename, dict):
            if use_box and 'boxes' in filename:
                for key in ('boxes', 'source', 'dezoom_format'):
                    if key in filename: bbox_kwargs[key] = filename[key]
            
            filename = filename['image' if 'image' in filename else 'filename']
        
        # Case 1 : not has_variable_input_size -> tar_shape = input_size
        # Case 2 : all input sizes are variable -> max_shape = max_image_shape
        # Case 3 : input size is semi-variable : tar_shape = input_size, max_shape = max_image_shape
        tar_shape, tar_max_shape, tar_mul_shape = None, None, None
        if self.resize_method == 'resize':
            tar_shape = self.target_image_shape[:2]
            max_shape = self.max_image_shape
            if self.should_pad_to_multiple:
                tar_mul_shape = self.downsampling_factor
        else:
            max_shape = self.max_image_shape

        resize_kwargs = {
            'target_max_shape'  : tar_max_shape,
            'target_multiple_shape' : tar_mul_shape,
            ** self.resize_kwargs,
            'target_shape'  : tar_shape
        }
        return load_image(
            filename,
            dtype   = 'float',
            mode    = 'gray' if self.input_size[-1] == 1 else None,
            ** resize_kwargs,
            ** bbox_kwargs,
            ** kwargs
        )
    
    def get_image_with_box(self, filename, ** kwargs):
        return self.get_image(filename, use_box = True, ** kwargs)

    def filter_image(self, image):
        img_shape   = ops.convert_to_numpy(ops.shape(image)[:2])
        target_shape    = ops.convert_to_numpy(self.target_image_shape[:2])
        
        good_shape  = ops.logical_or(
            ops.equal(target_shape, -1), ops.equal(target_shape, img_shape)
        )

        if self.has_variable_input_size and self.max_image_shape is not None:
            max_shape   = ops.convert_to_numpy(self.max_image_shape)[:2]
            small_enough    = ops.logical_or(
                ops.equal(max_shape, -1), ops.less_equal(img_shape, max_shape)
            )
            good_shape  = ops.logical_and(good_shape, small_enough)
        
        good_shape = ops.logical_and(img_shape > 0, good_shape)
        return ops.all(good_shape)
    
    def augment_image(self, image, ** kwargs):
        return augment_image(
            image, method = self.augmentation_method, prct = self.augment_prct, ** kwargs
        )
    
    def augment_box(self, data, ** kwargs):
        if 'boxes' in data:
            data['boxes'] = ops.cond(
                ops.random.uniform((), 0., 1.) <= self.augment_prct,
                lambda: augment_box(data['boxes'], ** kwargs),
                lambda: data['boxes']
            )
        return data

    def normalize_image(self, image, ** kwargs):
        """ Normalizes a (batch of) image by calling the normalization schema """
        return self.image_normalization_fn(image)

    def process_image(self, image, ** kwargs):
        """ Normalizes a (batch of) image by calling the normalization schema """
        image = self.normalize_image(image, ** kwargs)
        
        if self.resize_method == 'pad':
            image = pad_image(
                image,
                target_shape    = self.target_image_shape,
                target_multiple_shape   = self.downsampling_factor if self.should_pad_to_multiple else None,
                ** self.resize_kwargs
            )
        
        return image
    
    def get_config_image(self):
        config = {
            'input_size' : self.input_size,
            'resize_method' : self.resize_method,
            'resize_kwargs' : self.resize_kwargs,
            'image_normalization'   : self.image_normalization
        }
        if self.has_variable_input_size:
            config['max_image_size'] = self.max_image_size
        
        return config
    
    @staticmethod
    def get_image_data(data):
        if isinstance(data, dict):
            data = get_entry(data, ('tf_image', 'image', 'image_copy', 'filename'))

        if isinstance(data, str):
            return load_image(data, to_tensor = False, dtype = None, run_eagerly = True)
        return data
