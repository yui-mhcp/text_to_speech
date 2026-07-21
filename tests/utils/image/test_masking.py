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

""" Pure-``pytest`` tests for :mod:`utils.image.masking`.

``apply_mask`` forces its inputs to tensors (``ops.convert_to_tensor``) and the
``blur`` transformation relies on OpenCV, so those cases are marked ``keras`` /
``cv2`` (auto-skipped when the dependency is missing). ``create_color_mask`` runs
on pure numpy (``utils.keras.ops`` dispatches numpy) and stays keras-free.

``box_as_mask`` (from ``bounding_box.visualization``) is used as the mask source,
mirroring the original ``test_utils_boxes`` masking tests.
"""

import numpy as np
import pytest

from utils.keras import ops
from utils.image import apply_mask, create_color_mask, create_poly_mask, smooth_mask, box_as_mask


@pytest.fixture
def image():
    """ Deterministic (seeded) ``float32`` image, shape (64, 64, 3). """
    return np.random.default_rng(0).uniform(size = (64, 64, 3)).astype('float32')

@pytest.fixture
def box():
    return [10, 10, 30, 30]   # xywh


# --- `apply_mask` ------------------------------------------------------------------

@pytest.mark.cv2
@pytest.mark.keras
@pytest.mark.parametrize('on_background', [False, True])
@pytest.mark.parametrize('method', ['keep', 'remove', 'blur'])
def test_apply_mask(image, box, asserts, method, on_background):
    import cv2

    box_mask = box_as_mask(image, box, source = 'xywh')
    # `on_background` is applied *inside* `apply_mask` ; replicate it here for the target
    mask = box_mask if not on_background else ~box_mask

    if method == 'keep':      target = np.where(mask, image, 0.)
    elif method == 'remove':  target = np.where(mask, 0., image)
    else:                     target = np.where(mask, cv2.blur(image, (21, 21)), image)

    asserts.assert_equal(
        target, apply_mask(image, box_mask, method = method, on_background = on_background)
    )


@pytest.mark.keras
def test_apply_mask_unknown_method_raises(image, box):
    box_mask = box_as_mask(image, box, source = 'xywh')
    with pytest.raises(ValueError):
        apply_mask(image, box_mask, method = 'does_not_exist')


@pytest.mark.keras
def test_apply_mask_shape_mismatch_raises(image):
    bad_mask = np.ones((8, 8, 1), dtype = 'float32')
    with pytest.raises(ValueError):
        apply_mask(image, bad_mask, method = 'keep')


# --- `create_color_mask` (keras-free) ----------------------------------------------

def test_create_color_mask(asserts):
    image = np.zeros((8, 8, 3), dtype = 'float32')
    image[2: 5, 2: 5] = [1., 0., 0.]   # a red square

    expected = np.zeros((8, 8, 1), dtype = bool)
    expected[2: 5, 2: 5] = True

    asserts.assert_equal(expected, create_color_mask(image, color = (1., 0., 0.), threshold = 0.1))


def test_create_color_mask_with_mask(asserts):
    """ When a `mask` is given, its values are kept only where the color matches. """
    image = np.zeros((4, 4, 3), dtype = 'float32')
    image[0, 0] = [1., 0., 0.]
    src_mask = np.full((4, 4, 1), 7., dtype = 'float32')

    out = create_color_mask(image, color = (1., 0., 0.), threshold = 0.1, mask = src_mask)
    expected = np.zeros((4, 4, 1), dtype = 'float32')
    expected[0, 0] = 7.
    asserts.assert_equal(expected, out)


# --- `create_poly_mask` (cv2) ------------------------------------------------------

@pytest.mark.cv2
def test_create_poly_mask():
    image  = np.zeros((8, 8, 3), dtype = 'float32')
    square = [[2, 2], [5, 2], [5, 5], [2, 5]]

    mask = create_poly_mask(image, [square])
    assert mask.shape == (8, 8), 'Got {}'.format(mask.shape)
    assert mask[3, 3] != 0, 'The polygon interior should be filled'
    assert mask[0, 0] == 0, 'The polygon exterior should stay empty'


# --- `smooth_mask` (conv2d -> keras) -----------------------------------------------

@pytest.mark.keras
def test_smooth_mask_shape_and_range():
    mask = np.zeros((16, 16, 1), dtype = 'float32')
    mask[4: 12, 4: 12] = 1.

    smoothed = ops.convert_to_numpy(smooth_mask(mask, smooth_size = 3))
    assert smoothed.shape == mask.shape, 'Got {}'.format(smoothed.shape)
    assert smoothed.min() >= 0. and smoothed.max() <= 1., \
        'Smoothed values must stay in [0, 1], got [{}, {}]'.format(smoothed.min(), smoothed.max())
