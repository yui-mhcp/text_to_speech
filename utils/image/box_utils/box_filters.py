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

import logging
import numpy as np
import tensorflow as tf

from loggers import timer
from utils.image.box_utils.box_functions import NORMALIZE_01, BoxFormat, convert_box_format, bbox_iou

logger  = logging.getLogger(__name__)

def combine_box_filters(filters):
    def filter_intersect(boxes, indices, rows, ** kwargs):
        intersect = set(range(len(boxes)))
        for f in filters:
            intersect.intersection_update(f(
                boxes = boxes, indices = indices, rows = rows, ** kwargs
            ))
        return list(intersect)
    if not isinstance(filters, (list, tuple)): return filters
    elif len(filters) == 1: return filters[0]
    return filter_intersect

class BoxFilter:
    def __call__(self, boxes, indices, rows, ** kwargs):
        self.start()
        res = self.filter(boxes = boxes, indices = indices, rows = rows)
        self.finish()
        return res

    def start(self):    pass
    def finish(self):   pass
    def filter(self, box, indices, rows, ** kwargs):
        raise NotImplementedError()

class StreamFilter(BoxFilter):
    def __init__(self, threshold = 0.8, n_repeat = 2):
        self.n_repeat  = n_repeat
        self.threshold = threshold
        
        self.memory    = []
        self.new_boxes  = []
        self.waiting_boxes = []
        self.waiting_time  = []
        self.updated       = []
    
    def __len__(self):
        return len(self.memory) + len(self.waiting_boxes)
    
    def start(self):
        self.new_boxes  = []
        self.updated    = [False] * len(self.waiting_boxes)
    
    def finish(self):
        self.waiting_boxes = [b for b, up in zip(self.waiting_boxes, self.updated) if up]
        self.waiting_time  = [t for t, up in zip(self.waiting_time, self.updated) if up]
        
        if self.new_boxes:
            self.waiting_boxes.extend(self.new_boxes)
            self.waiting_time.extend([1] * len(self.new_boxes))
    
    def select_rows(self, list_rows, length):
        indices = [i for i, rows in enumerate(list_rows) if len(rows) == length]
        if not indices: return [], []
        return indices, np.concatenate([list_rows[idx] for idx in indices], axis = 0)

    def get_memory_boxes(self, length):
        return self.select_rows(self.memory, length)
    
    def get_waiting_boxes(self, length):
        return self.select_rows(self.waiting_boxes, length)

    def check_memory(self, rows):
        _, boxes = self.get_memory_boxes(len(rows))
        return len(get_rows_iou(rows, boxes, threshold = self.threshold))

    def check_waiting(self, rows):
        if self.n_repeat == 1: return True
        
        indices, boxes = self.get_waiting_boxes(len(rows))
        valids  = get_rows_iou(rows, boxes, self.threshold)
        if len(valids) == 0:
            self.new_boxes.append(rows)
        elif len(valids) == 1:
            idx = indices[valids[0]]
            self.waiting_time[idx] += 1
            self.updated[idx] = True
            if self.waiting_time[idx] >= self.n_repeat:
                self.waiting_time.pop(idx)
                self.waiting_boxes.pop(idx)
                return True
        else:
            logger.error('Multiple matches : {} !'.format(valids))
        return False

    @timer
    def filter(self, rows, ** kwargs):
        if isinstance(rows, list):
            return [i for i, rows_i in enumerate(rows) if self.filter(rows = rows_i)]
        
        if self.check_memory(rows): return False
        if self.check_waiting(rows):
            self.memory.append(rows)
            return True
        return False
    

class RegionFilter(BoxFilter):
    def __init__(self, region, mode = 'overlap', ** kwargs):
        self.mode   = mode
        self.x_min, self.y_min, self.x_max, self.y_max = convert_box_format(
            np.array(region),
            BoxFormat.CORNERS,
            as_list = True,
            normalize_mode = NORMALIZE_01,
            ** kwargs
        )
    
    def filter(self, boxes, ** kwargs):
        if len(boxes) == 0: return []
        
        if self.mode == 'overlap':
            mask = np.logical_and(
                np.logical_and(boxes[:, 0] < self.x_max, boxes[:, 2] > self.x_min),
                np.logical_and(boxes[:, 1] < self.y_max, boxes[:, 3] > self.y_min)
            )
        elif self.mode == 'center':
            center  = (boxes[:, :2] + boxes[:, 2:]) / 2.
            mask    = np.logical_and(
                np.logical_and(center[:, 0] > self.x_min, boxes[:, 0] < self.x_max),
                np.logical_and(center[:, 1] > self.y_min, boxes[:, 1] < self.y_max)
            )
        else:
            raise ValueError('Unknown mode for region filtering : {} !'.format(self.mode))
        
        return np.where(mask)[0]

def get_rows_iou(rows, boxes, threshold = 0.8, verbose = False):
    if len(boxes) == 0: return []

    ious = bbox_iou(
        rows, boxes, as_matrix = True, use_graph = False, box_mode = BoxFormat.CORNERS
    )
    if verbose: logger.info(np.around(ious, decimals = 3))
    ious = ious[0] if len(rows) == 1 else np.max(ious, axis = 0)
    ious = np.reshape(ious, [-1, len(rows)])
    return np.where(np.all(ious > threshold, axis = 1))[0]
