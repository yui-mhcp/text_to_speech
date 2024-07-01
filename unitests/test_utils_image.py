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

import os
import keras
import logging
import unittest
import numpy as np
import keras.ops as K
import tensorflow as tf

from absl.testing import parameterized

from utils.image import *
from utils.image.image_normalization import _clip_means, _clip_std, _east_means, _east_std
from unitests import CustomTestCase, data_dir

filename = os.path.join(data_dir, 'lena.jpg')

@unittest.skipIf(not os.path.exists(filename), '{} does not exist'.format(filename))
class TestImageIO(CustomTestCase, parameterized.TestCase):
    @parameterized.parameters(
        (None, None, None, (512, 512, 3)),
        ('gray', None, None, (512, 512, 1)),
        (None, (256, 256), None, (256, 256, 3)),
        ('gray', (256, 256), None, (256, 256, 1)),
        (None, (256, 512), None, (256, 512, 3)),
        (None, None, (256, 256), (256, 256, 3))
    )
    def test_load_image(self, mode, target_shape, max_shape, expected_shape):
        if mode and keras.backend.backend() != 'tensorflow':
            self.skipTest('The `mode` argument is currently only supported with tensorflow backend')
        
        image_tensor = load_image(
            filename, mode = mode, target_shape = target_shape, target_max_shape = max_shape
        )
        self.assertEqual(image_tensor.shape, expected_shape)
        self.assertTensor(image_tensor)

    def test_size(self):
        self.assertEqual(get_image_size(filename), (512, 512))
        self.assertEqual(get_image_size(load_image(filename)), (512, 512))

class TestImageUtils(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.fake_image = np.random.uniform(size = (512, 512, 3))
    
    @parameterized.parameters(
        {'target' : None, 'min' : None, 'max' : None, 'mul' : None, 'expected' : (512, 512)},
        
        {'target' : (256, 256), 'min' : None, 'max' : None, 'mul' : None, 'expected' : (256, 256)},
        {'target' : (600, 600), 'min' : None, 'max' : None, 'mul' : None, 'expected' : (600, 600)},
        {'target' : (-1, 256), 'min' : None, 'max' : None, 'mul' : None, 'expected' : (512, 256)},
        {'target' : (256, -1), 'min' : None, 'max' : None, 'mul' : None, 'expected' : (256, 512)},
        
        {'target' : None, 'min' : (256, 256), 'max' : None, 'mul' : None, 'expected' : (512, 512)},
        {'target' : None, 'min' : (600, 600), 'max' : None, 'mul' : None, 'expected' : (600, 600)},
        {'target' : None, 'min' : (-1, 600), 'max' : None, 'mul' : None, 'expected' : (512, 600)},
        {'target' : None, 'min' : (600, -1), 'max' : None, 'mul' : None, 'expected' : (600, 512)},

        {'target' : None, 'min' : None, 'max' : (256, 256), 'mul' : None, 'expected' : (256, 256)},
        {'target' : None, 'min' : None, 'max' : (600, 256), 'mul' : None, 'expected' : (512, 256)},
        {'target' : None, 'min' : None, 'max' : (-1, 256), 'mul' : None, 'expected' : (512, 256)},
        {'target' : None, 'min' : None, 'max' : (256, -1), 'mul' : None, 'expected' : (256, 512)},

        {'target' : None, 'min' : None, 'max' : None, 'mul' : (256, 256), 'expected' : (512, 512)},
        {'target' : None, 'min' : None, 'max' : None, 'mul' : (100, 100), 'expected' : (500, 500)},
        {'target' : None, 'min' : None, 'max' : None, 'mul' : (100, -1), 'expected' : (500, 512)},
        {'target' : None, 'min' : None, 'max' : None, 'mul' : (64, -1), 'expected' : (512, 512)},
    )
    def test_resized_shape(self, target, min, max, mul, expected):
        self.assertEqual(get_resized_shape(
            self.fake_image,
            shape    = target,
            max_shape    = max,
            min_shape    = min,
            multiples   = mul
        ), expected)

    def _test_resize(self):
        self.assertEqual(
            get_image_size(load_image(self.image, target_shape = self.res_size)), self.res_size
        )
        self.assertEqual(
            get_image_size(load_image(self.image, target_max_shape = self.res_size)), self.res_size
        )
        self.assertEqual(get_image_size(
            load_image(self.image, target_shape = self.res_size, preserve_aspect_ratio = True)
        ), self.res_size)
        self.assertEqual(get_image_size(
            load_image(self.image, target_max_shape = self.res_size, preserve_aspect_ratio = True)
        ), (100, 100))
        
        self.assertEqual(
            load_image(self.image, target_shape = (200, 100), preserve_aspect_ratio = True)[100:],
            np.zeros((100, 100, 3), dtype = np.float32)
        )

    def _test_color_mask(self):
        threshold = 50 / 255
        
        self.assertEqual(
            K.sum(K.abs(self.image - self.mask_color / 255.), axis = -1, keepdims = True) <= threshold,
            create_color_mask(self.image, self.mask_color, threshold = threshold, per_channel = False)
        )
        self.assertEqual(
            K.all(K.abs(self.image - self.mask_color / 255.) <= threshold, axis = -1, keepdims = True),
            create_color_mask(self.image, self.mask_color, threshold = threshold, per_channel = True)
        )

@parameterized.named_parameters(
    ('array',  np.random.uniform(size = (256, 256, 3)).astype(np.float32)),
    ('tensor', keras.random.uniform(shape = (256, 256, 3))),
)
class TestImageNormalization(CustomTestCase, parameterized.TestCase):
    def _test_normalization_schema(self, image, name, target):
        if isinstance(target, tuple):
            mean, std = target
            target = K.divide(
                K.convert_to_tensor(image) - K.reshape(K.convert_to_tensor(mean, image.dtype), [1, 1, 3]),
                K.reshape(K.convert_to_tensor(std, image.dtype), [1, 1, 3])
            )
        target = K.convert_to_numpy(target).astype(image.dtype.name)
        
        normalized = get_image_normalization_fn(name)(image)
        self.assertEqual(normalized, target)
        if isinstance(image, np.ndarray):
            self.assertArray(normalized)
        else:
            self.assertTensor(normalized)
            self.assertGraphCompatible(get_image_normalization_fn(name), image, target = target)
        
    def test_normalization_01(self, image):
        target = image - ops.min(image)
        target = target / ops.max(target)
        self._test_normalization_schema(image, '01', target)
    
    def test_normalization_normal(self, image):
        target = ops.divide_no_nan(image - ops.mean(image), ops.std(image))
        self._test_normalization_schema(image, 'mean', target)

    def test_normalization_none(self, image):
        self._test_normalization_schema(image, None, image)

    def test_normalization_clip(self, image):
        self._test_normalization_schema(image, 'clip', (_clip_means, _clip_std))
    
    def test_normalization_east(self, image):
        self._test_normalization_schema(image, 'east', (_east_means, _east_std))
    
    def test_normalization_easyocr(self, image):
        self._test_normalization_schema(image, 'east', (image - 0.5) / 0.5)

class TestImageNormalizationKeras(CustomTestCase, parameterized.TestCase):
    @parameterized.product(
        image   = (
            np.random.uniform(size = (256, 256, 3)),
            keras.random.uniform(shape = (256, 256, 3))
        ),
        model   = (
            'vgg16', 'vgg19', 'mobilenet'
        )
    )
    def test_image_normalization(self, image, model):
        copy    = image.copy() if isinstance(image, np.ndarray) else image
        target      = getattr(keras.applications, model).preprocess_input(copy)
        normalized  = get_image_normalization_fn(model)(image)
        
        self.assertEqual(normalized, target)
        if isinstance(image, np.ndarray):
            self.assertArray(normalized)
        else:
            self.assertTensor(normalized)
            self.assertGraphCompatible(get_image_normalization_fn(model), image, target = target)

