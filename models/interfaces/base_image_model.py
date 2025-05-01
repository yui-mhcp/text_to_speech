# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
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

from functools import cached_property

from utils import is_dataframe
from utils.image import load_image, get_image_normalization_fn
from utils.keras import TensorSpec, ops
from ..utils import infer_downsampling_factor, infer_upsampling_factor
from .base_model import BaseModel

class BaseImageModel(BaseModel):
    def _init_image(self,
                    input_size,
                    
                    resize_kwargs   = {},
                    
                    image_normalization     = None,
                    image_normalization_fn  = None,
                    
                    ** _
                   ):
        if not isinstance(input_size, (list, tuple)): input_size = (input_size, input_size, 3)
        
        self.input_size = tuple(input_size)
        self.resize_kwargs  = resize_kwargs
        self.image_normalization    = image_normalization
        
        if image_normalization_fn is None:
            image_normalization_fn = get_image_normalization_fn(image_normalization)
        
        self.image_normalization_fn = image_normalization_fn
        
        if 'target_multiple_shape' in resize_kwargs:
            resize_kwargs['multiples'] = resize_kwargs.pop('target_multiple_shape')
            resize_kwargs.pop('manually_compute_ratio', None)
        if 'method' in resize_kwargs:
            resize_kwargs['interpolation'] = resize_kwargs.pop('method')
    
    @property
    def color_mode(self):
        return 'gray' if self.input_size[-1] == 1 else 'rgb'
    
    @property
    def has_fixed_input_size(self):
        return any(s for s in self.input_size[:2])
    
    @property
    def has_variable_input_size(self):
        return any(s is None for s in self.input_size[:2])
    
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
    
    def _str_image(self):
        des = "- Image size : {}\n".format(self.input_size)
        if self.resize_kwargs: des += "- Resize config : {}\n".format(self.resize_kwargs)
        des += '- Normalization schema : {}\n'.format(self.image_normalization)
        return des
    
    def get_image(self, filename, use_box = False, ** kwargs):
        """ Calls `utils.image.load_image` on the given `filename` """
        if isinstance(filename, (list, tuple)):
            return [self.get_image(f, use_box, ** kwargs) for f in filename]
        elif is_dataframe(filename):
            return [self.get_image(row, use_box, ** kwargs) for idx, row in filename.iterrows()]
        
        box = None
        if isinstance(filename, dict):
            if use_box and 'boxes' in filename:
                box = filename['boxes']
                if not isinstance(box, dict) and 'format' in filename:
                    box = {'boxes' : box, 'format' : filename['format']}
            
            filename = filename['image' if 'image' in filename else 'filename']
        
        if self.should_pad_to_multiple and 'multiples' not in self.resize_kwargs:
            kwargs['multiples'] = self.downsampling_factor
        
        return load_image(
            filename,
            size    = self.input_size[:2] if self.has_fixed_input_size else None,
            
            boxes   = box,
            
            dtype   = 'float32',
            to_tensor   = True,
            channels    = self.input_size[-1],
            ** {** self.resize_kwargs, ** kwargs}
        )
    
    def get_image_with_box(self, filename, ** kwargs):
        return self.get_image(filename, use_box = True, ** kwargs)

    def normalize_image(self, image, ** kwargs):
        """ Normalizes a (batch of) image by calling the normalization schema """
        return self.image_normalization_fn(image) if self.image_normalization_fn else image

    process_image = normalize_image
    
    def get_config_image(self):
        return {
            'input_size' : self.input_size,
            'resize_kwargs' : self.resize_kwargs,
            'image_normalization'   : self.image_normalization
        }
    
    @staticmethod
    def get_image_data(data):
        filename = None
        if isinstance(data, dict):
            for k in ('tf_image', 'image', 'image_copy', 'filename'):
                if k in data:
                    if 'filename' in data: filename = data['filename']
                    data = data[k]
                    break

        if isinstance(data, str):
            return data, load_image(data, dtype = None, to_tensor = False)
        return filename, data
