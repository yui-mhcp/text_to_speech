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

""" Pure-``pytest`` tests for :mod:`utils.image.bounding_box.processing`.

Covers ``sort_boxes`` (every criterion, incl. ``top`` / ``score``), ``crop_box``
and ``select_boxes``. Shared box fixtures come from the local ``conftest.py``.

Backend handling mirrors ``test_distances`` : ``numpy`` stays keras-free, while
``tensor`` inputs are marked ``keras`` through the shared ``backend`` fixture.
"""

import numpy as np
import pytest

from utils.keras import ops
from utils.image.bounding_box import sort_boxes, crop_box, select_boxes

_FORMATS = ('xywh', 'xyxy')

def _to_tensor(x):
    return ops.convert_to_tensor(x)


@pytest.fixture
def image():
    """ Small deterministic 4x4 image used for crop tests. """
    return np.arange(16).reshape(4, 4)


# --- `sort_boxes` : coordinate-based criteria --------------------------------------

@pytest.mark.parametrize('source', _FORMATS)
@pytest.mark.parametrize('method,expected', [
    ('x',      [0, 1, 2]),
    ('y',      [0, 1, 2]),
    ('w',      [0, 2, 1]),
    ('h',      [0, 2, 1]),
    ('area',   [0, 2, 1]),
    ('center', [1, 0, 2]),
    ('corner', [0, 1, 2]),
])
def test_sort_boxes(backend, asserts, relative_boxes, absolute_boxes, image_shape,
                    source, method, expected):
    kwargs = {'return_indices': True, 'image_shape': image_shape, 'source': source}
    rel, abs_ = relative_boxes[source], absolute_boxes[source]
    if backend == 'tensor':
        rel, abs_ = _to_tensor(rel), _to_tensor(abs_)

    asserts.assert_equal(expected, sort_boxes(rel, method = method, ** kwargs))
    asserts.assert_equal(expected, sort_boxes(abs_, method = method, ** kwargs))


def test_sort_boxes_top(asserts):
    """ ``top`` groups boxes into rows (tolerant) then reads each row left-to-right. """
    # two rows of two boxes ; input order differs from the expected reading order
    boxes = np.array([
        [10,  0, 20, 10],   # 0 : top row, right
        [ 0,  0, 10, 10],   # 1 : top row, left
        [ 0, 20, 10, 30],   # 2 : bottom row, left
        [10, 20, 20, 30],   # 3 : bottom row, right
    ], dtype = 'float32')
    asserts.assert_equal(
        [1, 0, 2, 3], sort_boxes(boxes, method = 'top', source = 'xyxy', return_indices = True)
    )


def test_sort_boxes_score(asserts):
    """ ``score`` sorts in decreasing score order (requires a `dict` with ``scores``). """
    boxes = {
        'boxes'  : np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], dtype = 'float32'),
        'scores' : np.array([0.1, 0.9, 0.5], dtype = 'float32'),
    }
    asserts.assert_equal(
        [1, 2, 0], sort_boxes(boxes, method = 'score', source = 'xyxy', return_indices = True)
    )


# --- `crop_box` --------------------------------------------------------------------

def test_crop_empty_box(image):
    assert crop_box(image, [], source = 'xyxy')[1] is None


def test_crop_single_box(image, asserts):
    asserts.assert_equal(image[1: 4, :1], crop_box(image, [0, 1, 1, 3], source = 'xywh')[1])
    asserts.assert_equal(image[1: 3, :1], crop_box(image, [0, 1, 1, 3], source = 'xyxy')[1])


def test_crop_multiple_boxes(image, asserts):
    asserts.assert_equal(
        [image[1: 4, :1], image[2: 4, 2: 3]],
        crop_box(image, [[0, 1, 1, 3], [2, 2, 1, 2]], source = 'xywh')[1]
    )


# --- `select_boxes` ----------------------------------------------------------------

def test_select_boxes(asserts):
    boxes = np.arange(12).reshape(3, 4)
    asserts.assert_equal(boxes[[2, 0, 1]], select_boxes(boxes, [2, 0, 1]))


def test_select_boxes_dict(asserts):
    """ On a `dict`, boxes are gathered on axis -2 and per-box keys on axis -1. """
    boxes = {
        'boxes'  : np.arange(12).reshape(3, 4),
        'scores' : np.array([0.1, 0.2, 0.3]),
    }
    selected = select_boxes(boxes, [2, 0])
    asserts.assert_equal(boxes['boxes'][[2, 0]], selected['boxes'])
    asserts.assert_equal(boxes['scores'][[2, 0]], selected['scores'])
