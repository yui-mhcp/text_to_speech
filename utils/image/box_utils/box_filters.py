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

from dataclasses import dataclass

from utils.image.box_utils.box_functions import NORMALIZE_01, BoxFormat, convert_box_format, bbox_iou

logger  = logging.getLogger(__name__)

def combine_box_filters(filters):
    """
        Combines the result of multiple filters (i.e. returns only the boses that all filters have accepted)
        A filter must be a `callable` returning a boolean mask (i.e. 1D `np.ndarray` with same length as `boxes`)
    """
    def filter_intersect(boxes, indices, rows, ** kwargs):
        valid_indexes = list(range(len(boxes)))
        for f in filters:
            if not valid_indexes: break
            keep_indexes    = f(boxes = boxes, indices = indices, rows = rows)
            valid_indexes   = [valid_indexes[idx] for idx in keep_indexes]
            
            boxes   = boxes[keep_indexes]
            rows    = [rows[idx] for idx in keep_indexes]
            indices = [indices[idx] for idx in keep_indexes]
        
        return valid_indexes
    
    if not isinstance(filters, (list, tuple)): return filters
    elif len(filters) == 1: return filters[0]
    return filter_intersect

class BoxFilter:
    def __call__(self, boxes, indices, rows, ** kwargs):
        self.start()
        res = self.filter(boxes = boxes, indices = indices, rows = rows)
        self.finish()
        
        if isinstance(res, np.ndarray) and res.dtype == bool:
            res = np.where(res)[0]
        
        return res

    def start(self):    pass
    def finish(self):   pass
    def filter(self, box, indices, rows, ** kwargs):
        raise NotImplementedError()

@dataclass
class BoxItem:
    index      : int
    box        : np.ndarray
    rows       : np.ndarray
    indices    : list
    unseen     : int  = 0
    updated    : bool = True
    repetition : int  = 0

class RepetitionFilter(BoxFilter):
    """
        Box filtering method to only keep boxes that have been seen multiple times (`n_repeat`) at the same place (`iou_threshold`)
        The filter also filters out boxes that have already been emitted to avoid repeating them
    """
    def __init__(self, iou_threshold = 0.6, n_repeat = 2, max_unseen = 2, filter_memory = True):
        self.n_repeat       = n_repeat
        self.max_unseen     = max_unseen
        self.iou_threshold  = iou_threshold
        self.filter_memory  = filter_memory

        self.index     = 0
        
        self.memory    = []
        self.waiting   = []
        self.new_boxes = []
    
    def __len__(self):
        return len(self.memory) + len(self.waiting_boxes)
    
    def start(self):
        self.index += 1
        self.new_boxes  = []
        for b in self.waiting: b.updated = False
    
    def finish(self):
        for b in self.waiting:
            b.unseen = 0 if b.updated else b.unseen + 1
        self.waiting = [b for b in self.waiting if b.unseen < self.max_unseen] + self.new_boxes
    
    def select_candidates(self, candidates, rows, indices, nested_check = False, ** kwargs):
        def _get_length(indices):
            if isinstance(indices, int): return 1
            return [len(idx) if not isinstance(idx, int) else 1 for idx in indices]
        
        indices = [
            i for i, cand in enumerate(candidates)
            if _get_length(cand.indices) == _get_length(indices) and
            (not nested_check or _get_length(cand.indices) == _get_length(indices))
        ]
        if not indices: return [], []
        return indices, np.concatenate([candidates[idx].rows for idx in indices], axis = 0)

    def get_memory_boxes(self, * args, ** kwargs):
        return self.select_candidates(self.memory, * args, nested_check = True, ** kwargs)
    
    def get_waiting_boxes(self, * args, ** kwargs):
        return self.select_candidates(self.waiting, * args, nested_check = True, ** kwargs)

    def check_memory(self, rows, ** kwargs):
        indices, boxes = self.get_memory_boxes(rows = rows, ** kwargs)
        valids = get_rows_iou(rows, boxes, threshold = self.iou_threshold)
        if len(valids) and logger.isEnabledFor(logging.DEBUG):
            idx = indices[valids[0]]
            logger.debug('Index {} : match between {} and {}'.format(
                self.index, self.memory[idx], {** kwargs, 'rows' : rows}
            ))
        return len(valids)

    def check_waiting(self, rows, ** kwargs):
        if self.n_repeat == 1: return True
        
        indices, boxes = self.get_waiting_boxes(rows = rows, ** kwargs)
        valids  = get_rows_iou(rows, boxes, threshold = self.iou_threshold)
        if len(valids) == 0:
            self.new_boxes.append(BoxItem(index = self.index, rows = rows, ** kwargs))
        elif len(valids) == 1:
            idx = indices[valids[0]]
            self.waiting[idx].repetition += 1
            self.waiting[idx].updated = True
            if self.waiting[idx].repetition >= self.n_repeat:
                if self.filter_memory:
                    self.memory.append(self.waiting.pop(idx))
                return True
        else:
            logger.error('Multiple matches : {} !'.format(valids))
        return False

    def filter(self, boxes, rows, indices):
        if isinstance(rows, list):
            return np.array([
                self.filter(box, rows_i, indices_i)
                for box, rows_i, indices_i in zip(boxes, rows, indices)
            ], dtype = bool)

        # Filters out boxes that have already been emitted
        if self.check_memory(rows = rows, box = boxes, indices = indices): return False
        # Check if the box was already seen at least `n_repeat - 1` times
        return self.check_waiting(rows = rows, box = boxes, indices = indices)

class RegionFilter(BoxFilter):
    """ Filter out boxes that are not in the given region (`[x_min, y_min, x_max, y_max]`) """
    def __init__(self, region, mode = 'overlap', ** kwargs):
        self.mode   = mode
        self.x_min, self.y_min, self.x_max, self.y_max = convert_box_format(
            np.array(region) if not isinstance(region, dict) else region,
            BoxFormat.CORNERS,
            as_list     = True,
            extended    = False,
            normalize_mode  = NORMALIZE_01,
            ** kwargs
        )
    
    def filter(self, boxes, ** kwargs):
        if self.mode == 'overlap':
            return np.logical_and(
                np.logical_and(boxes[:, 0] < self.x_max, boxes[:, 2] > self.x_min),
                np.logical_and(boxes[:, 1] < self.y_max, boxes[:, 3] > self.y_min)
            )
        elif self.mode == 'center':
            center  = (boxes[:, :2] + boxes[:, 2:]) / 2.
            return np.logical_and(
                np.logical_and(self.x_min <= center[:, 0], center[:, 0] <= self.x_max),
                np.logical_and(self.y_min <= center[:, 1], center[:, 1] <= self.y_max)
            )
        else:
            raise ValueError('Unknown mode for region filtering : {} !'.format(self.mode))

class SizeFilter(BoxFilter):
    """ Filter out boxes that do not respect some size thresholds """
    def __init__(self, min_h = None, min_w = None, max_h = None, max_w = None, min_area = None, max_area = None, ** kwargs):
        self.min_h, self.max_h = min_h, max_h
        self.min_w, self.max_w = min_w, max_w
        self.min_area, self.max_area = min_area, max_area
    
    def filter(self, boxes, ** kwargs):
        valids = np.ones((len(boxes), ), dtype = bool)

        h = boxes[:, 3] - boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        a = h * w
        
        if self.min_h: valids = np.logical_and(valids, h >= self.min_h)
        if self.max_h: valids = np.logical_and(valids, h <= self.max_h)
        if self.min_w: valids = np.logical_and(valids, w >= self.min_w)
        if self.max_w: valids = np.logical_and(valids, w <= self.max_w)
        if self.min_area: valids = np.logical_and(valids, a >= self.min_area)
        if self.max_area: valids = np.logical_and(valids, a <= self.max_area)

        return valids

def get_rows_iou(rows, boxes, threshold = 0.8, verbose = False):
    if len(boxes) == 0: return []

    ious = bbox_iou(
        rows, boxes, as_matrix = True, use_graph = False, box_mode = BoxFormat.CORNERS
    )
    if verbose: logger.info(np.around(ious, decimals = 3))
    ious = ious[0] if len(rows) == 1 else np.max(ious, axis = 0)
    ious = np.reshape(ious, [-1, len(rows)])
    return np.where(np.all(ious > threshold, axis = 1))[0]
