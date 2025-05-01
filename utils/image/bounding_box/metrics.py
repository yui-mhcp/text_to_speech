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

import numpy as np

from loggers import timer
from ...keras import ops
from .converter import box_converter_wrapper

@timer
@box_converter_wrapper('xyxy', as_dict = False, as_list = True, dtype = 'float')
def compute_iou(boxes1, boxes2 = None, *, as_matrix = None, ** kwargs):
    if as_matrix is None:   as_matrix = boxes2 is None
    if boxes2 is None:      boxes2 = boxes1
    if len(boxes1) == 0 or len(boxes2) == 0:
        return ops.zeros((0, 0), dtype = 'float32')

    if as_matrix:
        boxes1 = [b[..., None] for b in boxes1]
        boxes2 = [b[..., None, :] for b in boxes2]

    xmin_1, ymin_1, xmax_1, ymax_1 = boxes1
    xmin_2, ymin_2, xmax_2, ymax_2 = boxes2

    areas_1 = (ymax_1 - ymin_1) * (xmax_1 - xmin_1)
    areas_2 = (ymax_2 - ymin_2) * (xmax_2 - xmin_2)

    xmin, ymin = ops.maximum(xmin_1, xmin_2), ops.maximum(ymin_1, ymin_2)
    xmax, ymax = ops.minimum(xmax_1, xmax_2), ops.minimum(ymax_1, ymax_2)

    _zero = ops.convert_to_numpy(0, xmax.dtype)
    inter_w, inter_h = ops.maximum(_zero, xmax - xmin), ops.maximum(_zero, ymax - ymin)

    # making the `- inter` in the middle reduces value overflow when using `float16` computation
    inter = inter_w * inter_h
    union = areas_1 - inter + areas_2

    return ops.divide_no_nan(inter, union)

@timer
@box_converter_wrapper('xyxy', as_dict = False, as_list = True, dtype = 'float')
def compute_ioa(boxes1, boxes2 = None, *, as_matrix = None, ** kwargs):
    if as_matrix is None:   as_matrix = boxes2 is None
    if boxes2 is None:      boxes2 = boxes1
    
    xmin_1, ymin_1, xmax_1, ymax_1 = boxes1
    areas_1 = (ymax_1 - ymin_1) * (xmax_1 - xmin_1)

    if as_matrix:
        boxes1 = [b[..., None] for b in boxes1]
        boxes2 = [b[..., None, :] for b in boxes2]
        areas_1 = areas_1[..., None]
    
    xmin_1, ymin_1, xmax_1, ymax_1 = boxes1
    xmin_2, ymin_2, xmax_2, ymax_2 = boxes2

    xmin, ymin = ops.maximum(xmin_1, xmin_2), ops.maximum(ymin_1, ymin_2)
    xmax, ymax = ops.minimum(xmax_1, xmax_2), ops.minimum(ymax_1, ymax_2)

    _zero = ops.convert_to_numpy(0, xmax.dtype)
    inter_w, inter_h = ops.maximum(_zero, xmax - xmin), ops.maximum(_zero, ymax - ymin)

    return ops.divide_no_nan(inter_w * inter_h, areas_1)

