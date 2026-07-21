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

""" Pure-``pytest`` tests for :mod:`utils.image.bounding_box.visualization`.

``draw_boxes`` operates on a ``np.ndarray`` image (keras-free) but relies on
``cv2`` -> marked ``cv2`` (auto-skipped when OpenCV is missing). ``cv2`` is
imported lazily inside the tests so collection never fails when it is absent.

Note : ``draw_boxes`` mutates the numpy image in place (``cv2`` draws in place),
so every reference / call gets its own ``image.copy()``.
"""

import numpy as np
import pytest

from utils.image.bounding_box import draw_boxes, box_as_mask
from utils.image.bounding_box.visualization import normalize_color


@pytest.fixture
def image():
    """ Deterministic (seeded) ``float32`` image, shape (720, 1024, 3). """
    return np.random.default_rng(0).uniform(size = (720, 1024, 3)).astype('float32')

@pytest.fixture
def box():
    return [250, 200, 500, 500]


@pytest.mark.cv2
def test_draw_boxes_rectangle(image, box, asserts):
    import cv2
    x, y, w, h = box
    color = normalize_color('blue', dtype = image.dtype.name).tolist()
    asserts.assert_equal(
        cv2.rectangle(image.copy(), (x, y), (x + w, y + h), color, 3),
        draw_boxes(image.copy(), box, shape = 'rectangle', color = 'blue', thickness = 3, source = 'xywh')
    )


@pytest.mark.cv2
def test_draw_boxes_circle(image, box, asserts):
    import cv2
    x, y, w, h = box
    color = normalize_color('blue', dtype = image.dtype.name).tolist()
    asserts.assert_equal(
        cv2.circle(image.copy(), (x + w // 2, y + h // 2), min(w, h) // 2, color, 3),
        draw_boxes(image.copy(), box, shape = 'circle', color = 'blue', thickness = 3, source = 'xywh')
    )


@pytest.mark.cv2
def test_draw_boxes_ellipse(image, box, asserts):
    import cv2
    x, y, w, h = box
    color = normalize_color('blue', dtype = image.dtype.name).tolist()
    asserts.assert_equal(
        cv2.ellipse(
            image.copy(),
            angle      = 0,
            startAngle = 0,
            endAngle   = 360,
            center     = (x + w // 2, y + h // 2),
            thickness  = 3,
            axes       = (w // 2, int(h / 1.5)),
            color      = color
        ),
        draw_boxes(image.copy(), box, shape = 'ellipse', color = 'blue', thickness = 3, source = 'xywh')
    )


# --- `box_as_mask` -----------------------------------------------------------------

@pytest.mark.cv2
@pytest.mark.parametrize('shape', ['rectangle', 'ellipse', 'circle'])
def test_box_as_mask(image, box, asserts, shape):
    """ The mask is the (any-channel) non-zero footprint of a filled-box drawing. """
    expected = np.any(draw_boxes(
        np.zeros(image.shape, dtype = image.dtype), box,
        shape = shape, thickness = -1, source = 'xywh'
    ) != 0, axis = -1, keepdims = True)

    mask = box_as_mask(image, box, shape = shape, source = 'xywh')

    asserts.assert_equal(expected, mask)
    assert mask.dtype == bool, 'Expected a boolean mask, got {}'.format(mask.dtype)
    assert mask.shape == image.shape[:2] + (1, ), 'Got {}'.format(mask.shape)
    assert mask.any(), 'The mask should be non-empty inside the box'
