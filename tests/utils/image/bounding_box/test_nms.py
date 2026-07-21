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

""" Pure-``pytest`` tests for :mod:`utils.image.bounding_box.non_max_suppression`
and :mod:`utils.image.bounding_box.locality_aware_nms`.

``nms`` forces its inputs to tensors (``box_converter_wrapper(force_tensor=True)``)
so the whole module needs a backend -> marked ``keras``. The ``tensorflow`` NMS
method additionally requires TF -> that single parameter is marked ``tensorflow``
(auto-skipped when TF is missing).
"""

import numpy as np
import pytest

from utils.keras import ops
from utils.image.bounding_box import nms


@pytest.fixture
def boxes():
    """ 4 overlapping boxes ; consecutive ones overlap, every-other ones do not. """
    return np.array([
        [0,   0,   0.2, 0.2],
        [0.1, 0.1, 0.3, 0.3],
        [0.2, 0.2, 0.4, 0.4],
        [0.3, 0.3, 0.5, 0.5],
    ], dtype = 'float32')


def _selected(result):
    out_boxes, _, valids = result
    return ops.convert_to_numpy(out_boxes)[ops.convert_to_numpy(valids)]


# --- standard NMS variants ---------------------------------------------------------

@pytest.mark.keras
@pytest.mark.parametrize('run_eagerly', [True, False])
@pytest.mark.parametrize('method', [
    pytest.param('tensorflow', marks = pytest.mark.tensorflow),
    'nms', 'fast', 'padded',
])
def test_nms_standard(boxes, asserts, method, run_eagerly):
    """ With a 0.1 threshold only the non-overlapping boxes 0 and 2 survive. """
    selected = _selected(nms(
        boxes, nms_threshold = 0.1, source = 'xyxy', method = method, run_eagerly = run_eagerly
    ))
    asserts.assert_equal(boxes[[0, 2]], selected)


# --- locality-aware NMS (lanms) ----------------------------------------------------

@pytest.mark.keras
def test_lanms_union(boxes, asserts):
    """ Default ``union`` merge : overlapping boxes are merged into their bounding union. """
    selected = _selected(nms(
        boxes, nms_threshold = 0.1, merge_threshold = 0.1,
        source = 'xyxy', method = 'lanms', run_eagerly = True
    ))
    asserts.assert_equal(
        np.array([[0, 0, 0.3, 0.3], [0.2, 0.2, 0.5, 0.5]], dtype = 'float32'), selected
    )


@pytest.mark.keras
def test_lanms_average(boxes, asserts):
    """ ``average`` merge averages the coordinates of the merged boxes. """
    selected = _selected(nms(
        boxes, nms_threshold = 0.1, merge_threshold = 0.1, merge_method = 'average',
        source = 'xyxy', method = 'lanms', run_eagerly = True
    ))
    asserts.assert_equal(
        np.array([[0.05, 0.05, 0.25, 0.25], [0.25, 0.25, 0.45, 0.45]], dtype = 'float32'), selected
    )


@pytest.mark.keras
def test_lanms_low_nms_threshold(boxes, asserts):
    """ A very low ``nms_threshold`` suppresses one of the two merged boxes. """
    selected = _selected(nms(
        boxes, nms_threshold = 0.01, merge_threshold = 0.1,
        source = 'xyxy', method = 'lanms', run_eagerly = True
    ))
    asserts.assert_equal(np.array([[0, 0, 0.3, 0.3]], dtype = 'float32'), selected)


@pytest.mark.keras
def test_lanms_is_iterative(boxes, asserts):
    """ A low ``merge_threshold`` lets the merge cascade over all boxes (iterative). """
    selected = _selected(nms(
        boxes, nms_threshold = 0.1, merge_threshold = 0.01,
        source = 'xyxy', method = 'lanms', run_eagerly = True
    ))
    asserts.assert_equal(
        np.array([[0, 0, 0.5, 0.5]], dtype = 'float32'), selected,
        msg = 'The LANMS should be iterative'
    )
