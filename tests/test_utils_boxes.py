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
import cv2
import unittest
import numpy as np

from absl.testing import parameterized

from utils.keras import ops
from utils.image import load_image
from utils.image.bounding_box import *
from utils.image.bounding_box.visualization import _normalize_color as normalize_color
from . import CustomTestCase, data_dir, is_tensorflow_available

_box_formats    = ('xywh', 'xyxy')

class TestBoxes(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.image_h, self.image_w = 720, 1024
        
        self.relative_boxes_xywh = np.array([
            [0, 0, 1, 1], [0.25, 0.2, 0.1, 0.2], [0.5, 0.5, 0.5, 0.5]
        ], dtype = np.float32)
        self.relative_boxes_xyxy = np.array([
            [0, 0, 1, 1], [0.25, 0.2, 0.35, 0.4], [0.5, 0.5, 1,1]
        ], dtype = np.float32)

        factor  = np.array([[self.image_w, self.image_h, self.image_w, self.image_h]])
        self.absolute_boxes_xywh = np.array([
            [0, 0, 1, 1], [0.25, 0.2, 0.1, 0.2], [0.5, 0.5, 0.5, 0.5]
        ], dtype = np.float32) * factor
        self.absolute_boxes_xyxy = np.array([
            [0, 0, 1, 1], [0.25, 0.2, 0.35, 0.4], [0.5, 0.5, 1,1]
        ], dtype = np.float32) * factor
        
        self.absolute_boxes_xywh = self.absolute_boxes_xywh.astype(np.int32)
        self.absolute_boxes_xyxy = self.absolute_boxes_xyxy.astype(np.int32)

class TestBoxesConvertion(TestBoxes):
    @parameterized.product(
        source = _box_formats, target = _box_formats, to_tensor = (True, False)
    )
    def test_converter(self, source, target, to_tensor):
        rel_in_boxes    = getattr(self, 'relative_boxes_{}'.format(source))
        rel_out_boxes   = getattr(self, 'relative_boxes_{}'.format(target))
        abs_in_boxes    = getattr(self, 'absolute_boxes_{}'.format(source))
        abs_out_boxes   = getattr(self, 'absolute_boxes_{}'.format(target))

        if to_tensor:
            rel_in_boxes    = ops.convert_to_tensor(rel_in_boxes)
            rel_out_boxes   = ops.convert_to_tensor(rel_out_boxes)
            abs_in_boxes    = ops.convert_to_tensor(abs_in_boxes)
            abs_out_boxes   = ops.convert_to_tensor(abs_out_boxes)

        if source == target:
            self.assertTrue(
                convert_box_format(rel_in_boxes, source = source, target = target) is rel_in_boxes,
                'The function should return the same instance when `source == target`'
            )
            self.assertTrue(
                convert_box_format(
                    rel_in_boxes, source = source, target = target, normalize_mode = 'relative'
                ) is rel_in_boxes,
                'The function should return the same instance when `source == target`'
            )
            self.assertTrue(
                convert_box_format(abs_in_boxes, source = source, target = target) is abs_in_boxes,
                'The function should return the same instance when `source == target`'
            )
            self.assertTrue(
                convert_box_format(
                    abs_in_boxes, source = source, target = target, normalize_mode = 'absolute'
                ) is abs_in_boxes,
                'The function should return the same instance when `source == target`'
            )


        self.assertEqual(
            rel_out_boxes, convert_box_format(rel_in_boxes, source = source, target = target)
        )
        self.assertEqual(ops.unstack(rel_out_boxes, axis = -1, num = 4), convert_box_format(
            rel_in_boxes, source = source, target = target, as_list = True
        ))

        self.assertEqual(abs_out_boxes, convert_box_format(
            abs_in_boxes, source = source, target = target,
        ))
        
        self.assertEqual(abs_out_boxes, convert_box_format(
            rel_in_boxes,
            source = source,
            target = target,
            normalize_mode  = 'absolute',
            image_h = self.image_h,
            image_w = self.image_w
        ))
        self.assertEqual(rel_out_boxes, convert_box_format(
            abs_in_boxes,
            source = source,
            target = target,
            normalize_mode  = 'relative',
            image_h = self.image_h,
            image_w = self.image_w
        ), max_err = 5e-4)
    
    def test_dezoom(self):
        kwargs = {'source' : 'xywh'}
        
        self.assertEqual(
            np.array([[0.25, 0.25, 0.5, 0.5]], dtype = 'float64'),
            convert_box_format([0., 0., 1., 1.], dezoom_factor = 0.5, ** kwargs)
        )
        self.assertEqual(
            np.array([[0, 0, 1, 1]], dtype = 'float64'),
            convert_box_format([0., 0., 1., 1.], dezoom_factor = 2, ** kwargs)
        )
        
        self.assertEqual(
            np.array([[0, 0, 1, 1]], dtype = 'float64'),
            convert_box_format([0.25, 0.25, .5, .5], dezoom_factor = 2, ** kwargs)
        )
        self.assertEqual(
            np.array([[0.25, 0.25, .75, .75]], dtype = 'float64'),
            convert_box_format([0.5, 0.5, .5, .5], dezoom_factor = 2, ** kwargs)
        )


class TestMetrics(CustomTestCase):
    def test_single_iou(self):
        box1 = np.array([[100, 101, 200, 201]])
        box2 = box1 + 1
        # area of bb1 and bb1_off_by_1 are each 10000.
        # intersection area is 99*99=9801
        # iou=9801/(2*10000 - 9801)=0.96097656633
        self.assertEqual(
            [0.96097656633], compute_iou(box1[0], box2[0], source = "xyxy")
        )
        self.assertEqual(
             [0.96097656633], compute_iou(box1, box2, source = "xyxy")
        )

    def test_iou(self):
        bb1 = [100, 101, 200, 201]
        bb1_off_by_1_pred = [101, 102, 201, 202]
        iou_bb1_bb1_off = 0.96097656633
        top_left_bounding_box = [0, 2, 1, 3]
        far_away_box = [1300, 1400, 1500, 1401]
        another_far_away_pred = [1000, 1400, 1200, 1401]

        # Rows represent predictions, columns ground truths
        expected_matrix_result = np.array(
            [[iou_bb1_bb1_off, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32
        )

        sample_y_true = np.array([bb1, top_left_bounding_box, far_away_box], dtype = 'int32')
        sample_y_pred = np.array(
            [bb1_off_by_1_pred, top_left_bounding_box, another_far_away_pred], dtype = 'int32'
        )
        
        self.assertEqual(
            expected_matrix_result,
            compute_iou(sample_y_true, sample_y_pred, source = "xyxy", as_matrix = True)
        )
        self.assertEqual(
            np.diagonal(expected_matrix_result),
            compute_iou(sample_y_true, sample_y_pred, source = "xyxy")
        )
        
        
        batch_y_true = np.stack([
            sample_y_true, sample_y_true[::-1]
        ], axis = 0)
        batch_y_pred = np.stack([
            sample_y_pred, sample_y_pred[::-1]
        ], axis = 0)
        batch_matrix    = np.stack([
            expected_matrix_result, expected_matrix_result[::-1, ::-1]
        ], axis = 0)
        
        self.assertEqual(
            batch_matrix, compute_iou(batch_y_true, batch_y_pred, source = "xyxy", as_matrix = True)
        )
        self.assertEqual(
            np.stack([np.diagonal(batch_matrix[0]), np.diagonal(batch_matrix[1])], axis = 0),
            compute_iou(batch_y_true, batch_y_pred, source = "xyxy")
        )

    def test_ioa(self):
        box1 = np.array([[1, 1, 5, 10]])
        box2 = box1 * 2
        box3 = np.array([[0, 0, 2, 2]])

        self.assertEqual(
            [36 / 50], compute_ioa(box1[0], box2[0], source = "xywh")
        )
        self.assertEqual(
            np.array([36 / 50], dtype = 'float32'), compute_ioa(box1, box2, source = "xywh")
        )
        
        boxes = np.concatenate([box1, box2, box3], axis = 0)
        self.assertEqual(
            np.array([
                [1., 36 / 50, 1 / 50],
                [36 / 200, 1, 0],
                [1 / 4, 0, 1]
            ], dtype = 'float32'),
            compute_ioa(boxes, source = "xywh", as_matrix = True)
        )
        self.assertEqual(
            np.array([
                [1, 1 / 50],
                [36 / 200, 0],
                [1 / 4, 1]
            ], dtype = 'float32'),
            compute_ioa(boxes, boxes[[0, 2]], source = "xywh", as_matrix = True)
        )
        self.assertEqual(
            np.array([
                [1, 36 / 50, 1 / 50],
                [1 / 4, 0, 1]
            ], dtype = 'float32'),
            compute_ioa(boxes[[0, 2]], boxes, source = "xywh", as_matrix = True)
        )


class TestBoxProcessing(TestBoxes):
    def setUp(self):
        super().setUp()
        self.image = np.arange(16).reshape(4, 4)
    
    @parameterized.parameters(
        {'method' : 'x', 'expected' : [0, 1, 2]},
        {'method' : 'y', 'expected' : [0, 1, 2]},
        {'method' : 'w', 'expected' : [0, 2, 1]},
        {'method' : 'h', 'expected' : [0, 2, 1]},
        {'method' : 'area', 'expected' : [0, 2, 1]},
        {'method' : 'center', 'expected' : [1, 0, 2]},
        {'method' : 'corner', 'expected' : [0, 1, 2]},
    )
    def test_sort(self, method, expected):
        for to_tensor in (False, True):
            with self.subTest(to_tensor = to_tensor):
                for source in _box_formats:
                    kwargs = {
                        'return_indices' : True,
                        'image_shape' : (self.image_h, self.image_w),
                        'source' : source
                    }
                    rel_boxes    = getattr(self, 'relative_boxes_{}'.format(source))
                    abs_boxes    = getattr(self, 'absolute_boxes_{}'.format(source))

                    if to_tensor:
                        rel_boxes    = ops.convert_to_tensor(rel_boxes)
                        abs_boxes    = ops.convert_to_tensor(abs_boxes)

                    self.assertEqual(
                        expected, sort_boxes(rel_boxes, method = method, ** kwargs)
                    )
                    self.assertEqual(
                        expected, sort_boxes(abs_boxes, method = method, ** kwargs)
                    )

    def test_crop_empty_box(self):
        self.assertEqual(None, crop_box(self.image, [], source = 'xyxy')[1])

    def test_crop_single_box(self):
        self.assertEqual(
            self.image[1: 4, :1], crop_box(self.image, [0, 1, 1, 3], source = 'xywh')[1]
        )
        self.assertEqual(
            self.image[1 : 3, :1], crop_box(self.image, [0, 1, 1, 3], source = 'xyxy')[1]
        )

    def test_crop_multiple_boxes(self):
        self.assertEqual(
            [self.image[1:4, :1], self.image[2:4, 2:3]],
            crop_box(self.image, [[0, 1, 1, 3], [2, 2, 1, 2]], source = 'xywh')[1]
        )


class TestDrawing(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.image = np.random.uniform(size = (720, 1024, 3)).astype('float32')
        self.mask_box = [250, 200, 500, 500]
        self.kwargs = {'color' : 'blue', 'thickness' : 3, 'source' : 'xywh'}
    
    def test_rectangle(self):
        x, y, w, h = self.mask_box
        color = normalize_color('blue', dtype = self.image.dtype.name).tolist()
        
        self.assertEqual(
            cv2.rectangle(
                self.image.copy(), (x, y), (x + w, y + h), color, 3
            ),
            draw_boxes(
                self.image.copy(), self.mask_box, shape = 'rectangle', ** self.kwargs
            )
        )
        
    def test_circle(self):
        x, y, w, h = self.mask_box
        color = normalize_color('blue', dtype = self.image.dtype.name).tolist()

        self.assertEqual(
            cv2.circle(
                ops.convert_to_numpy(self.image), (x + w // 2, y + h // 2),
                min(w, h) // 2, color, 3
            ),
            draw_boxes(
                self.image, self.mask_box, shape = 'circle', ** self.kwargs
            )
        )
    
    def test_ellipse(self):
        x, y, w, h = self.mask_box
        color = normalize_color('blue', dtype = self.image.dtype.name).tolist()

        self.assertEqual(
            cv2.ellipse(
                ops.convert_to_numpy(self.image),
                angle       = 0,
                startAngle  = 0,
                endAngle    = 360, 
                center      = (x + w // 2, y + h // 2),
                thickness   = 3,
                axes    = (w // 2, int(h / 1.5)),
                color   = color
            ),
            draw_boxes(
                self.image, self.mask_box, color = 'blue', shape = 'ellipse', thickness = 3, source = 'xywh'
            )
        )

    @parameterized.parameters('rectangle', 'ellipse', 'circle')
    def _test_box_to_mask(self, shape):
        self.assertEqual(
            box_as_mask(self.image, self.mask_box, shape = shape),
            draw_boxes(
                np.zeros(self.image.shape), self.mask_box, shape = shape, thickness = -1
            )[..., :1] != 0
        )

    def _test_box_masking(self):
        box_mask = box_as_mask(self.image, self.mask_box)
        
        for method in apply_mask.methods.keys():
            if method == 'replace': continue
            for on_background in (False, True):
                with self.subTest(method = method, on_background = on_background):
                    mask = box_mask if not on_background else ~box_mask
                    if method == 'keep':
                        target = np.where(mask, self.image, 0.)
                    elif method == 'remove':
                        target = np.where(mask, 0., self.image)
                    elif method == 'blur':
                        target = np.where(
                            mask,
                            cv2.blur(ops.convert_to_numpy(self.image), (21, 21)),
                            self.image
                        )

                    self.assertEqual(apply_mask(
                        self.image, box_mask, method = method, on_background = on_background
                    ), target)




class TestCombination(CustomTestCase, parameterized.TestCase):
    """
        These tests have been built based on real images, based on pretrained EAST detection
        Then, the method has been executed and validated based on visual evaluation
    """
    def test_simple(self):
        boxes = np.array([[0.2052, 0.8635, 0.2501, 0.8865],
         [0.2443, 0.8626, 0.2930, 0.8862],
         [0.2856, 0.8623, 0.3319, 0.8881],
         [0.3280, 0.8642, 0.4000, 0.8893],
         [0.4100, 0.8613, 0.4525, 0.8883],
         [0.4459, 0.8607, 0.5001, 0.8888],
         [0.4944, 0.8579, 0.5490, 0.8902]])
        target_groups_h = [[0, 1, 2, 3, 4, 5, 6]]
        target_boxes_h  = np.array([[0.2052, 0.8579, 0.5490, 0.8902]])
        target_groups_hv = [[[0, 1, 2, 3, 4, 5, 6]]]
        target_boxes_hv  = np.array([[0.2052, 0.8579, 0.5490, 0.8902]])

        combined_boxes, groups, _ = combine_boxes_horizontal(boxes, source = 'xyxy')
        self.assertEqual(target_groups_h, groups)
        self.assertEqual(target_boxes_h, combined_boxes)
        
        combined_boxes, groups, _ = combine_boxes_vertical(combined_boxes, groups, source = 'xyxy')
        self.assertEqual(target_groups_hv, groups)
        self.assertEqual(target_boxes_hv, combined_boxes)

    def test_with_lots_of_boxes(self):
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
        self.assertEqual(target_groups_h, groups)
        self.assertEqual(target_boxes_h, combined_boxes)
        
        combined_boxes, groups, _ = combine_boxes_vertical(combined_boxes, groups, source = 'xyxy')
        self.assertEqual(target_groups_hv, groups)
        self.assertEqual(target_boxes_hv, combined_boxes)

    def test_line_with_space(self):
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
        target_groups_h = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        target_boxes_h  = np.array([[0.2068, 0.8577, 0.7734, 0.8900]])
        target_groups_hv = [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]]
        target_boxes_hv  = np.array([[0.2068, 0.8577, 0.7734, 0.8900]])

        combined_boxes, groups, _ = combine_boxes_horizontal(boxes, source = 'xyxy')
        self.assertEqual(target_groups_h, groups)
        self.assertEqual(target_boxes_h, combined_boxes)
        
        combined_boxes, groups, _ = combine_boxes_vertical(combined_boxes, groups, source = 'xyxy')
        self.assertEqual(target_groups_hv, groups)
        self.assertEqual(target_boxes_hv, combined_boxes)

    def test_multi_line_with_space(self):
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
        target_groups_h = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]]
        target_boxes_h  = np.array([[0.1979, 0.8622, 0.7792, 0.8976],
         [0.1978, 0.8904, 0.2939, 0.9214]])
        target_groups_hv = [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]]]
        target_boxes_hv  = np.array([[0.1978, 0.8622, 0.7792, 0.9214]])

        combined_boxes, groups, _ = combine_boxes_horizontal(boxes, source = 'xyxy')
        self.assertEqual(target_groups_h, groups)
        self.assertEqual(target_boxes_h, combined_boxes)
        
        combined_boxes, groups, _ = combine_boxes_vertical(combined_boxes, groups, source = 'xyxy')
        self.assertEqual(target_groups_hv, groups)
        self.assertEqual(target_boxes_hv, combined_boxes)

class TestNonMaxSuppression(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.boxes = np.array([
            [0, 0, 0.2, 0.2],
            [0.1, 0.1, 0.3, 0.3],
            [0.2, 0.2, 0.4, 0.4],
            [0.3, 0.3, 0.5, 0.5]
        ], dtype = 'float32')

    @parameterized.parameters('tensorflow', 'nms', 'fast', 'padded')
    def test_standard_nms(self, method):
        if method == 'tensorflow' and not is_tensorflow_available():
            self.skip('Tensorflow is not available, skipping the `tensorflow` nms method')
            return
        
        boxes, scores, valids = nms(
            self.boxes, nms_threshold = 0.1, source = 'xyxy', method = method, run_eagerly = True
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(boxes, self.boxes[[0, 2]])
        
        boxes, scores, valids = nms(
            self.boxes, nms_threshold = 0.1, source = 'xyxy', method = method, run_eagerly = False
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(boxes, self.boxes[[0, 2]])
    
    def test_locality_aware_nms(self):
        boxes, scores, valids = nms(
            self.boxes,
            nms_threshold   = 0.1,
            merge_threshold = 0.1,
            source  = 'xyxy',
            method  = 'lanms',
            run_eagerly = True
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(
            boxes, np.array([[0, 0, 0.3, 0.3], [0.2, 0.2, 0.5, 0.5]], dtype = 'float32')
        )
        
        boxes, scores, valids = nms(
            self.boxes,
            nms_threshold   = 0.1,
            merge_threshold = 0.1,
            merge_method    = 'average',
            source  = 'xyxy',
            method  = 'lanms',
            run_eagerly = True
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(
            boxes, np.array([[0.05, 0.05, 0.25, 0.25], [0.25, 0.25, 0.45, 0.45]], dtype = 'float32')
        )


        boxes, scores, valids = nms(
            self.boxes,
            nms_threshold   = 0.01,
            merge_threshold = 0.1,
            source  = 'xyxy',
            method  = 'lanms',
            run_eagerly = True
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(boxes, np.array([[0, 0, 0.3, 0.3]], dtype = 'float32'))

        boxes, scores, valids = nms(
            self.boxes,
            nms_threshold   = 0.1,
            merge_threshold = 0.01,
            source  = 'xyxy',
            method  = 'lanms',
            run_eagerly = True
        )
        boxes = ops.convert_to_numpy(boxes)[ops.convert_to_numpy(valids)]
        self.assertEqual(
            boxes, np.array([[0, 0, 0.5, 0.5]], dtype = 'float32'), 'The LANMS should be iterative'
        )
