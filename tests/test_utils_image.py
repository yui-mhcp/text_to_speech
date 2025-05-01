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

import os
import keras
import logging
import unittest
import numpy as np
import keras.ops as K

from absl.testing import parameterized

from utils.image import *
from . import CustomTestCase, data_dir
from utils.image.image_normalization import _clip_means, _clip_std, _east_means, _east_std

filename = os.path.join(data_dir, 'lena.jpg')

@unittest.skipIf(not os.path.exists(filename), '{} does not exist'.format(filename))
class TestImageIO(CustomTestCase, parameterized.TestCase):
    def test_load_image(self):
        image = load_image(filename)
        self.assertEqual((512, 512, 3), image.shape)
        self.assertEqual((512, 512, 3), load_image(image).shape)
        
        self.assertEqual((256, 256, 3), load_image(image, size = (256, 256)).shape)

    def test_size(self):
        self.assertEqual((512, 512), get_image_size(filename))
        self.assertEqual((512, 512), get_image_size(load_image(filename)))

class TestImageProcessing(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.fake_image = np.random.uniform(size = (512, 512, 3))
    
    @parameterized.parameters(
        {'size' : None, 'target' : (512, 512)},
        {'size' : (256, 256), 'target' : (256, 256)},
        {'size' : (256, None), 'target' : (256, 512)},
        {'size' : (None, 256), 'target' : (512, 256)},
        {'size' : (None, None), 'target' : (512, 512)}
    )
    def test_get_output_size_simple(self, size, target):
        self.assertEqual(target, get_output_size(self.fake_image, size))

    @parameterized.parameters([
        (None, 64, (512, 512)),
        (None, 100, (600, 600)),
        ((256, None), 100, (256, 600)),
        ((None, 64), 100, (600, 64))
    ])
    def test_get_output_size_multiples(self, size, multiples, target):
        self.assertEqual(target, get_output_size(self.fake_image, size, multiples = multiples))

    @parameterized.parameters([
        (None, None, (512, 512)),
        ((256, 256), None, (256, 256)),
        
        ((256, None), None, (256, 256)),
        ((None, 500), None, (500, 500)),
        
        (None, 64, (512, 512)),
        (None, 100, (600, 600)),
        ((256, None), 100, (256, 300)),
        ((None, 64), 100, (100, 64))
    ])
    def test_get_output_size_preserve_ratio(self, size, multiples, target):
        self.assertEqual(target, get_output_size(
            self.fake_image, size, multiples = multiples, preserve_aspect_ratio = True
        ))

    def test_resize(self):
        self.assertTrue(resize_image(self.fake_image, run_eagerly = True) is self.fake_image)
        self.assertTrue(
            resize_image(self.fake_image, self.fake_image.shape[:2], run_eagerly = True) is self.fake_image
        )
        self.assertEqual((256, 256, 3), tuple(resize_image(self.fake_image, (256, 256)).shape))
        self.assertEqual((256, 512, 3), tuple(resize_image(
            self.fake_image, (256, None)
        ).shape))
        
        self.assertEqual((256, 256, 3), tuple(resize_image(
            self.fake_image, (256, None), preserve_aspect_ratio = True
        ).shape))

        padded = resize_image(self.fake_image, (256, 512), preserve_aspect_ratio = True)
        self.assertEqual((256, 512, 3), tuple(padded.shape))
        self.assertEqual(np.zeros((256, 256, 3), 'float32'), padded[:, 256 :])
        
        padded = resize_image(
            self.fake_image, (256, 512), preserve_aspect_ratio = True, pad_value = 1.
        )
        self.assertEqual((256, 512, 3), tuple(padded.shape))
        self.assertEqual(np.ones((256, 256, 3), 'float32'), padded[:, 256 :])
        

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
        self.assertEqual(target, normalized)
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
        self._test_normalization_schema(image, 'normal', target)

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
