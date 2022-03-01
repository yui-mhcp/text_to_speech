
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

import os

from unitest import Test, assert_function, assert_equal, assert_smaller

import os
import tensorflow as tf

from utils.image import load_image, augment_image
from utils.image.mask_utils import create_color_mask
from utils.image.box_utils import *

_filename = os.path.join('test', '__datas', 'lena.jpg')

_image = None

def maybe_load_image():
    global _image
    if _image is None:
        _image = load_image(_filename)
    
    return _image

@Test
def image_io():
    image = maybe_load_image()
    grayscale = load_image(_filename, mode = 'gray')
    
    assert_equal((512, 512, 3), image.shape)
    assert_equal((512, 512, 1), grayscale.shape)
    assert_equal((512, 512), get_image_size, _filename)
    assert_equal(lambda: get_image_size(_filename), lambda: get_image_size(image))
    

@Test(contains_randomness = True)
def test_image_augmentation():
    image = maybe_load_image()
    
    for i, transform in enumerate(['color', 'flip_horizontal', 'flip_vertical', 'noise', 'hue', 'saturation', 'brightness', 'contrast']):
        assert_function(augment_image, image, transform, 1., seed = i)

@Test
def test_image_mask():
    image = maybe_load_image()

    color = (225, 225, 225)
    box   = [len(image) // 4, len(image) // 3, len(image) // 2, len(image) // 2]
    
    assert_function(create_color_mask, image, color, threshold = 50)
    
    assert_function(draw_boxes, image, box)
    assert_function(draw_boxes, image, box, shape = ELLIPSE)
    assert_function(draw_boxes, image, box, shape = CIRCLE)

    assert_function(box_as_mask, image, box, shape = OVALE, dezoom_factor = 1.)

    mask = box_as_mask(image, box, shape = OVALE, dezoom_factor = 1)
    
    assert_function(apply_mask, image, mask, transform = 'keep')
    assert_function(apply_mask, image, mask, transform = 'remove', on_background = True)
    
    assert_function(apply_mask, image, mask, transform = 'keep', smooth = True, smooth_size = 0.75)
    assert_function(apply_mask, image, mask, transform = 'blur', on_background = True, smooth = True)
