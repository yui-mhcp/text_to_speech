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

""" Pure-``pytest`` tests for :mod:`utils.image.bounding_box.combination`.

These cases were built from real images : ``EAST`` detections were combined and
the result validated visually, then frozen here as regression goldens. Everything
runs on ``np.ndarray`` (keras-free), so the module stays unmarked.

Each test feeds raw boxes to ``combine_boxes_horizontal`` (lines) then chains
``combine_boxes_vertical`` (paragraphs), checking both the produced boxes and the
index groups at every stage.
"""

import numpy as np

from utils.image.bounding_box import combine_boxes_horizontal, combine_boxes_vertical


def test_combine_simple(asserts):
    boxes = np.array([[0.2052, 0.8635, 0.2501, 0.8865],
     [0.2443, 0.8626, 0.2930, 0.8862],
     [0.2856, 0.8623, 0.3319, 0.8881],
     [0.3280, 0.8642, 0.4000, 0.8893],
     [0.4100, 0.8613, 0.4525, 0.8883],
     [0.4459, 0.8607, 0.5001, 0.8888],
     [0.4944, 0.8579, 0.5490, 0.8902]])
    target_groups_h  = [[0, 1, 2, 3, 4, 5, 6]]
    target_boxes_h   = np.array([[0.2052, 0.8579, 0.5490, 0.8902]])
    target_groups_hv = [[[0, 1, 2, 3, 4, 5, 6]]]
    target_boxes_hv  = np.array([[0.2052, 0.8579, 0.5490, 0.8902]])

    combined_boxes, groups, _ = combine_boxes_horizontal(boxes, source = 'xyxy')
    asserts.assert_equal(target_groups_h, groups)
    asserts.assert_equal(target_boxes_h, combined_boxes)

    combined_boxes, groups, _ = combine_boxes_vertical(combined_boxes, groups, source = 'xyxy')
    asserts.assert_equal(target_groups_hv, groups)
    asserts.assert_equal(target_boxes_hv, combined_boxes)


def test_combine_with_lots_of_boxes(asserts):
    boxes = np.array([[0.0059, -0.0021, 0.1044, 0.0433],
     [0.0928, 0.0073, 0.1775, 0.0390],
     [0.5240, 0.0227, 0.5985, 0.0611],
     [0.6827, 0.0191, 0.7810, 0.0605],
     [0.8794, 0.0234, 0.9272, 0.0582],
     [0.0706, 0.1227, 0.1325, 0.1479],
     [0.0954, 0.1940, 0.1870, 0.2216],
     [0.8001, 0.1787, 0.9473, 0.2582],
     [0.9333, 0.2030, 0.9884, 0.2337],
     [0.9157, 0.2174, 1.0020, 0.2555],
     [0.0770, 0.2688, 0.1245, 0.2913],
     [0.1055, 0.3409, 0.1740, 0.3679],
     [0.4107, 0.3608, 0.4474, 0.3856],
     [0.4438, 0.3606, 0.4659, 0.3850],
     [0.4619, 0.3620, 0.5260, 0.3881],
     [0.5198, 0.3619, 0.5587, 0.3853],
     [0.5567, 0.3621, 0.6071, 0.3868],
     [0.7319, 0.3521, 0.8585, 0.4079],
     [0.8707, 0.3613, 0.9840, 0.4017],
     [0.4048, 0.3824, 0.4693, 0.4132],
     [0.4598, 0.3856, 0.5490, 0.4130],
     [0.5468, 0.3863, 0.6125, 0.4135],
     [0.6088, 0.3883, 0.6387, 0.4123],
     [0.0751, 0.4117, 0.1411, 0.4365],
     [0.4113, 0.4139, 0.4509, 0.4341],
     [0.4472, 0.4120, 0.4867, 0.4345],
     [0.0482, 0.4661, 0.0937, 0.4930],
     [0.0515, 0.4873, 0.0928, 0.5094],
     [0.1079, 0.4823, 0.1621, 0.5103],
     [0.7133, 0.5003, 0.7741, 0.5582],
     [0.8054, 0.4933, 0.9250, 0.5577],
     [0.7058, 0.5397, 0.7511, 0.6063],
     [0.4489, 0.6093, 0.4903, 0.6381],
     [0.1553, 0.8783, 0.1828, 0.8985],
     [0.0399, 0.8824, 0.0725, 0.9075],
     [0.0650, 0.8840, 0.0987, 0.9069],
     [0.1520, 0.8949, 0.2012, 0.9180],
     [0.6727, 0.9371, 0.7438, 0.9671],
     [0.7816, 0.9403, 0.8281, 0.9712],
     [0.8236, 0.9410, 0.8687, 0.9684]])
    target_groups_h = [[0, 1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12, 13, 14, 15, 16], [17, 18], [19, 20, 21, 22], [23], [24, 25], [26], [27, 28], [29], [30], [31], [32], [33], [34, 35], [36], [37], [38, 39]]
    target_boxes_h  = np.array([[0.0059, -0.0021, 0.1775, 0.0433],
     [0.5240, 0.0227, 0.5985, 0.0611],
     [0.6827, 0.0191, 0.7810, 0.0605],
     [0.8794, 0.0234, 0.9272, 0.0582],
     [0.0706, 0.1227, 0.1325, 0.1479],
     [0.0954, 0.1940, 0.1870, 0.2216],
     [0.8001, 0.1787, 0.9473, 0.2582],
     [0.9333, 0.2030, 0.9884, 0.2337],
     [0.9157, 0.2174, 1.0020, 0.2555],
     [0.0770, 0.2688, 0.1245, 0.2913],
     [0.1055, 0.3409, 0.1740, 0.3679],
     [0.4107, 0.3606, 0.6071, 0.3881],
     [0.7319, 0.3521, 0.9840, 0.4079],
     [0.4048, 0.3824, 0.6387, 0.4135],
     [0.0751, 0.4117, 0.1411, 0.4365],
     [0.4113, 0.4120, 0.4867, 0.4345],
     [0.0482, 0.4661, 0.0937, 0.4930],
     [0.0515, 0.4823, 0.1621, 0.5103],
     [0.7133, 0.5003, 0.7741, 0.5582],
     [0.8054, 0.4933, 0.9250, 0.5577],
     [0.7058, 0.5397, 0.7511, 0.6063],
     [0.4489, 0.6093, 0.4903, 0.6381],
     [0.1553, 0.8783, 0.1828, 0.8985],
     [0.0399, 0.8824, 0.0987, 0.9075],
     [0.1520, 0.8949, 0.2012, 0.9180],
     [0.6727, 0.9371, 0.7438, 0.9671],
     [0.7816, 0.9403, 0.8687, 0.9712]])
    target_groups_hv = [[[0, 1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8], [9]], [[10]], [[11]], [[12, 13, 14, 15, 16], [19, 20, 21, 22], [24, 25]], [[17, 18]], [[23]], [[26], [27, 28]], [[29], [31]], [[30]], [[32]], [[33], [36]], [[34, 35]], [[37]], [[38, 39]]]
    target_boxes_hv  = np.array([[0.0059, -0.0021, 0.1775, 0.0433],
     [0.5240, 0.0227, 0.5985, 0.0611],
     [0.6827, 0.0191, 0.7810, 0.0605],
     [0.8794, 0.0234, 0.9272, 0.0582],
     [0.0706, 0.1227, 0.1325, 0.1479],
     [0.0954, 0.1940, 0.1870, 0.2216],
     [0.8001, 0.1787, 0.9473, 0.2582],
     [0.9157, 0.2030, 1.0020, 0.2555],
     [0.0770, 0.2688, 0.1245, 0.2913],
     [0.1055, 0.3409, 0.1740, 0.3679],
     [0.4048, 0.3606, 0.6387, 0.4345],
     [0.7319, 0.3521, 0.9840, 0.4079],
     [0.0751, 0.4117, 0.1411, 0.4365],
     [0.0482, 0.4661, 0.1621, 0.5103],
     [0.7058, 0.5003, 0.7741, 0.6063],
     [0.8054, 0.4933, 0.9250, 0.5577],
     [0.4489, 0.6093, 0.4903, 0.6381],
     [0.1520, 0.8783, 0.2012, 0.9180],
     [0.0399, 0.8824, 0.0987, 0.9075],
     [0.6727, 0.9371, 0.7438, 0.9671],
     [0.7816, 0.9403, 0.8687, 0.9712]])

    combined_boxes, groups, _ = combine_boxes_horizontal(boxes, source = 'xyxy', h_factor = 1.)
    asserts.assert_equal(target_groups_h, groups)
    asserts.assert_equal(target_boxes_h, combined_boxes)

    combined_boxes, groups, _ = combine_boxes_vertical(combined_boxes, groups, source = 'xyxy')
    asserts.assert_equal(target_groups_hv, groups)
    asserts.assert_equal(target_boxes_hv, combined_boxes)


def test_combine_line_with_space(asserts):
    boxes = np.array([[0.2068, 0.8587, 0.2526, 0.8852],
     [0.2439, 0.8601, 0.2884, 0.8887],
     [0.2797, 0.8627, 0.3152, 0.8864],
     [0.3085, 0.8611, 0.3480, 0.8861],
     [0.3483, 0.8620, 0.3831, 0.8867],
     [0.4098, 0.8625, 0.4582, 0.8884],
     [0.4514, 0.8603, 0.4895, 0.8880],
     [0.4900, 0.8578, 0.5558, 0.8900],
     [0.5470, 0.8594, 0.5819, 0.8881],
     [0.5796, 0.8598, 0.6640, 0.8875],
     [0.6485, 0.8607, 0.6948, 0.8873],
     [0.6811, 0.8577, 0.7734, 0.8864]])
    target_groups_h  = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    target_boxes_h   = np.array([[0.2068, 0.8577, 0.7734, 0.8900]])
    target_groups_hv = [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]]
    target_boxes_hv  = np.array([[0.2068, 0.8577, 0.7734, 0.8900]])

    combined_boxes, groups, _ = combine_boxes_horizontal(boxes, source = 'xyxy')
    asserts.assert_equal(target_groups_h, groups)
    asserts.assert_equal(target_boxes_h, combined_boxes)

    combined_boxes, groups, _ = combine_boxes_vertical(combined_boxes, groups, source = 'xyxy')
    asserts.assert_equal(target_groups_hv, groups)
    asserts.assert_equal(target_boxes_hv, combined_boxes)


def test_combine_multi_line_with_space(asserts):
    boxes = np.array([[0.1979, 0.8680, 0.2281, 0.8898],
     [0.2171, 0.8691, 0.2532, 0.8976],
     [0.2514, 0.8622, 0.3301, 0.8916],
     [0.3213, 0.8648, 0.4020, 0.8908],
     [0.3898, 0.8635, 0.4325, 0.8925],
     [0.4589, 0.8659, 0.5293, 0.8940],
     [0.5216, 0.8642, 0.5752, 0.8915],
     [0.5701, 0.8632, 0.6514, 0.8923],
     [0.6444, 0.8681, 0.7024, 0.8898],
     [0.6945, 0.8693, 0.7250, 0.8885],
     [0.7205, 0.8676, 0.7792, 0.8924],
     [0.1978, 0.8904, 0.2939, 0.9214]])
    target_groups_h  = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]]
    target_boxes_h   = np.array([[0.1979, 0.8622, 0.7792, 0.8976],
     [0.1978, 0.8904, 0.2939, 0.9214]])
    target_groups_hv = [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]]]
    target_boxes_hv  = np.array([[0.1978, 0.8622, 0.7792, 0.9214]])

    combined_boxes, groups, _ = combine_boxes_horizontal(boxes, source = 'xyxy')
    asserts.assert_equal(target_groups_h, groups)
    asserts.assert_equal(target_boxes_h, combined_boxes)

    combined_boxes, groups, _ = combine_boxes_vertical(combined_boxes, groups, source = 'xyxy')
    asserts.assert_equal(target_groups_hv, groups)
    asserts.assert_equal(target_boxes_hv, combined_boxes)
