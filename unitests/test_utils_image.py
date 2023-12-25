# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
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
import logging
import unittest
import numpy as np
import tensorflow as tf

from utils.image import *
from utils.image.image_normalization import _clip_means, _clip_std, _east_means, _east_std
from unitests import CustomTestCase, data_dir

filename = os.path.join(data_dir, 'lena.jpg')

@unittest.skipIf(not os.path.exists(filename), '{} does not exist'.format(filename))
class TestImage(CustomTestCase):
    def setUp(self):
        self.image      = load_image(filename)
        self.resized    = load_image(filename, target_shape = (100, 200))
        self.img_size   = (512, 512)
        self.res_size   = (100, 200)
        
        self.mask_color = 255
        self.mask_box   = [
            self.image.shape[1] // 4, self.image.shape[0] // 4,
            self.image.shape[1] // 2, self.image.shape[0] // 2
        ]

    def test_boxes(self):
        self.assertEqual(
            convert_box_format(self.mask_box, BoxFormat.XYWH), self.mask_box
        )
        self.assertEqual(convert_box_format(
            self.mask_box, BoxFormat.XYWH, dezoom_factor = 2., image = self.image
        ), [0, 0, self.image.shape[1], self.image.shape[0]])
        self.assertEqual(
            convert_box_format(self.mask_box, BoxFormat.CORNERS),
            self.mask_box[:2] + [3 * self.image.shape[1] // 4, 3 * self.image.shape[0] // 4]
        )
        self.assertEqual(
            convert_box_format(self.mask_box, BoxFormat.DICT),{
                'xmin' : self.mask_box[0], 'ymin' : self.mask_box[1],
                'xmax' : 3 * self.image.shape[1] // 4, 'ymax' : 3 * self.image.shape[0] // 4,
                'width' : self.mask_box[2], 'height' : self.mask_box[3]
            }
        )


        for shape in ('rectangle', 'ellipse', 'circle'):
            with self.subTest(shape = shape):
                self.assertReproductible(draw_boxes(
                    self.image, self.mask_box, color = 'blue', shape = shape
                ), 'draw_boxes-{}.npy'.format(shape))

        
    def test_load(self):
        self.assertEqual(self.image.shape, self.img_size + (3, ))
        self.assertEqual(
            load_image(filename, mode = 'gray').shape, self.img_size + (1, )
        )

    def test_size(self):
        self.assertEqual(get_image_size(filename), self.img_size)
        self.assertEqual(get_image_size(self.image), self.img_size)
    
    def test_resized_shape(self):
        self.assertEqual(
            get_resized_shape(self.image), self.img_size
        )
        self.assertEqual(
            get_resized_shape(self.image, min_shape = (128, 128)), self.img_size
        )
        self.assertEqual(get_resized_shape(
            self.image, min_shape = (-1, 600)
        ), (self.img_size[0], 600))
        self.assertEqual(get_resized_shape(
            self.image, min_shape = (-1, 600), preserve_aspect_ratio = True
        ), (600, 600))

        self.assertEqual(
            get_resized_shape(self.image, max_shape = self.res_size), self.res_size
        )
        self.assertEqual(
            get_resized_shape(self.image, max_shape = (-1, -1)), self.img_size
        )
        self.assertEqual(get_resized_shape(
            self.image, max_shape = (-1, 256)
        ), (self.img_size[0], 256))
        self.assertEqual(get_resized_shape(
            self.image, max_shape = (256, -1), preserve_aspect_ratio = True
        ), (256, 256))

    
    def test_resize(self):
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
            np.zeros((100, 100, 3), dtype = np.float32),
            load_image(self.image, target_shape = (200, 100), preserve_aspect_ratio = True)[100:]
        )


    def test_image_augmentation(self):
        for i, method in enumerate(augment_image.methods.keys()):
            with self.subTest(method = method):
                self.assertReproductible(augment_image(
                    self.image, method, 1., seed = i
                ), 'augment_image-{}.npy'.format(method))

    def test_image_normalization(self):
        self.assertEqual(np.max(get_image_normalization_fn('01')(self.image)), 1.)
        self.assertEqual(np.min(get_image_normalization_fn('01')(self.image)), 0.)
        self.assertEqual(
            tf.keras.applications.vgg16.preprocess_input(self.image),
            get_image_normalization_fn('vgg')(self.image)
        )
        self.assertEqual(
            tf.keras.applications.vgg16.preprocess_input(self.image),
            get_image_normalization_fn('vgg16')(self.image)
        )
        self.assertEqual(
            tf.keras.applications.vgg19.preprocess_input(self.image),
            get_image_normalization_fn('vgg19')(self.image)
        )
        self.assertEqual(
            tf.keras.applications.mobilenet.preprocess_input(self.image),
            get_image_normalization_fn('mobilenet')(self.image)
        )
        self.assertEqual(
            (self.image - np.reshape(_clip_means, [1, 1, 3])) / np.reshape(_clip_std, [1, 1, 3]),
            get_image_normalization_fn('clip')(self.image)
        )
        self.assertEqual(
            (self.image - np.reshape(_east_means, [1, 1, 3])) / np.reshape(_east_std, [1, 1, 3]),
            get_image_normalization_fn('east')(self.image)
        )
        self.assertEqual(
            (self.image - np.mean(self.image)) / np.std(self.image),
            get_image_normalization_fn('mean')(self.image)
        )
        self.assertEqual(
            (self.image - 0.5) / 0.5, get_image_normalization_fn('easyocr')(self.image)
        )
        self.assertEqual(
            self.image, get_image_normalization_fn(None)(self.image)
        )

    def test_masking(self):
        threshold = 50 / 255
        
        self.assertEqual(
            np.sum(np.abs(self.image - self.mask_color / 255.), axis = -1, keepdims = True) <= threshold,
            create_color_mask(self.image, self.mask_color, threshold = threshold, per_channel = False)
        )
        self.assertEqual(
            np.all(np.abs(self.image - self.mask_color / 255.) <= threshold, axis = -1, keepdims = True),
            create_color_mask(self.image, self.mask_color, threshold = threshold, per_channel = True)
        )
        
        for shape in ('rectangle', 'ellipse', 'circle'):
            with self.subTest(shape = shape):
                self.assertReproductible(box_as_mask(
                    self.image, self.mask_box, shape = 'rectangle'
                ), 'box_as_mask-{}.npy'.format(shape))

        mask = box_as_mask(self.image, self.mask_box)
        
        for method in apply_mask.methods.keys():
            if method == 'replace': continue
            with self.subTest(method = method):
                self.assertReproductible(apply_mask(
                    self.image, mask, method = method
                ), 'apply_mask-{}.npy'.format(method))

        self.assertReproductible(apply_mask(
            self.image, mask, method = 'blur', on_background = True
        ), 'apply_mask_background.npy')
