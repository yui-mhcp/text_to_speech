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

""" Pure-``pytest`` tests for :mod:`utils.image.bounding_box.metrics`.

``compute_iou`` / ``compute_ioa`` run on pure ``np.ndarray`` inputs :
``utils.keras.ops`` dispatches numpy without importing keras, so the whole module
stays keras-free (unmarked). Tolerant comparisons go through the ``asserts`` fixture.
"""

import numpy as np

from utils.image.bounding_box import compute_iou, compute_ioa


# --- `compute_iou` -----------------------------------------------------------------

def test_compute_iou_single(asserts):
    box1 = np.array([[100, 101, 200, 201]])
    box2 = box1 + 1
    # areas of box1 and box2 are each 100 * 100 = 10000
    # intersection area is 99 * 99 = 9801
    # iou = 9801 / (2 * 10000 - 9801) = 0.96097656633
    asserts.assert_equal([0.96097656633], compute_iou(box1[0], box2[0], source = 'xyxy'))
    asserts.assert_equal([0.96097656633], compute_iou(box1, box2, source = 'xyxy'))


def test_compute_iou_matrix_and_aligned(asserts):
    bb1               = [100, 101, 200, 201]
    bb1_off_by_1_pred = [101, 102, 201, 202]
    iou_bb1_bb1_off   = 0.96097656633
    top_left_box      = [0, 2, 1, 3]
    far_away_box      = [1300, 1400, 1500, 1401]
    far_away_pred     = [1000, 1400, 1200, 1401]

    # rows = predictions, columns = ground truths
    expected = np.array(
        [[iou_bb1_bb1_off, 0., 0.], [0., 1., 0.], [0., 0., 0.]], dtype = np.float32
    )
    y_true = np.array([bb1, top_left_box, far_away_box], dtype = 'int32')
    y_pred = np.array([bb1_off_by_1_pred, top_left_box, far_away_pred], dtype = 'int32')

    asserts.assert_equal(expected, compute_iou(y_true, y_pred, source = 'xyxy', as_matrix = True))
    # the aligned (non-matrix) mode is the matrix diagonal
    asserts.assert_equal(np.diagonal(expected), compute_iou(y_true, y_pred, source = 'xyxy'))


def test_compute_iou_batched(asserts):
    bb1               = [100, 101, 200, 201]
    bb1_off_by_1_pred = [101, 102, 201, 202]
    iou_bb1_bb1_off   = 0.96097656633
    top_left_box      = [0, 2, 1, 3]
    far_away_box      = [1300, 1400, 1500, 1401]
    far_away_pred     = [1000, 1400, 1200, 1401]

    expected = np.array(
        [[iou_bb1_bb1_off, 0., 0.], [0., 1., 0.], [0., 0., 0.]], dtype = np.float32
    )
    y_true = np.array([bb1, top_left_box, far_away_box], dtype = 'int32')
    y_pred = np.array([bb1_off_by_1_pred, top_left_box, far_away_pred], dtype = 'int32')

    batch_y_true = np.stack([y_true, y_true[::-1]], axis = 0)
    batch_y_pred = np.stack([y_pred, y_pred[::-1]], axis = 0)
    batch_matrix = np.stack([expected, expected[::-1, ::-1]], axis = 0)

    asserts.assert_equal(
        batch_matrix, compute_iou(batch_y_true, batch_y_pred, source = 'xyxy', as_matrix = True)
    )
    asserts.assert_equal(
        np.stack([np.diagonal(batch_matrix[0]), np.diagonal(batch_matrix[1])], axis = 0),
        compute_iou(batch_y_true, batch_y_pred, source = 'xyxy')
    )


# --- `compute_ioa` -----------------------------------------------------------------

def test_compute_ioa(asserts):
    box1 = np.array([[1, 1, 5, 10]])
    box2 = box1 * 2
    box3 = np.array([[0, 0, 2, 2]])

    asserts.assert_equal([36 / 50], compute_ioa(box1[0], box2[0], source = 'xywh'))
    asserts.assert_equal(np.array([36 / 50], dtype = 'float32'), compute_ioa(box1, box2, source = 'xywh'))

    boxes = np.concatenate([box1, box2, box3], axis = 0)
    asserts.assert_equal(
        np.array([[1., 36 / 50, 1 / 50], [36 / 200, 1, 0], [1 / 4, 0, 1]], dtype = 'float32'),
        compute_ioa(boxes, source = 'xywh', as_matrix = True)
    )
    asserts.assert_equal(
        np.array([[1, 1 / 50], [36 / 200, 0], [1 / 4, 1]], dtype = 'float32'),
        compute_ioa(boxes, boxes[[0, 2]], source = 'xywh', as_matrix = True)
    )
    asserts.assert_equal(
        np.array([[1, 36 / 50, 1 / 50], [1 / 4, 0, 1]], dtype = 'float32'),
        compute_ioa(boxes[[0, 2]], boxes, source = 'xywh', as_matrix = True)
    )
