# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
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

from dataclasses import dataclass

from .metrics import compute_iou
from .converter import NORMALIZE_01, BoxFormat, convert_box_format

logger  = logging.getLogger(__name__)

def filter_boxes(filters, boxes, ** kwargs):
    if callable(filters): return filters(boxes, ** kwargs)
    
    valid_indexes = np.arange(len(boxes))
    for f in filters:
        if not valid_indexes: return []
        keep_indexes    = f(boxes = boxes, ** kwargs)
        valid_indexes   = valid_indexes[keep_indexes]
        
        boxes   = boxes[keep_indexes]
        rows    = [rows[idx] for idx in keep_indexes]
        indices = [indices[idx] for idx in keep_indexes]
        
    return valid_indexes

class BoxFilter:
    """ Abstract class representing a box filtering strategy """
    def __call__(self, boxes, indices, rows, ** kwargs):
        """
            Filtering method that takes the boxes / indices / rows (typically returned by `combine_boxes`) and returns the indices to keep
            
            Arguments :
                - boxes : 2-D `np.ndarray` with shape `(n_boxes, 4)`, the original boxes
                - indices   : list of combination indexes (mainly not used)
                - rows  : individual rows composing a box, `list` with `n_boxes` items, each item being a 2-D `np.ndarray` with shape `(n_rows, 4)`
            Return :
                - keep_indexes  : list of index(es) to keep
            
        """
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
    def __init__(self, iou_threshold = 0.5, n_repeat = 2, max_unseen = 2, use_memory = False):
        """
            This filter keeps boxes that are repeated `n_repeat` consecutive frames at the same position (up to `iou_threshold` tolerance)
            This allows to ensure the box is repeated multiple times, and is a fiable prediction
            
            Arguments :
                - iou_threshold : the threshold to determine that 2 boxes are at the same position
                - n_repeat  : the number of times the box has to be emitted at the same position
                - max_unseen    : drops boxes that are not seen this number of frames
                - use_memory    : whether to filter boxes that match a previously emitted position
        """
        self.n_repeat       = n_repeat
        self.max_unseen     = max_unseen
        self.iou_threshold  = iou_threshold
        self.use_memory     = use_memory

        self.index     = 0
        
        self.memory    = []
        self.waiting   = []
        self.new_boxes = None
    
    def __len__(self):
        return len(self.memory) + len(self.waiting_boxes)
    
    def start(self):
        self.index += 1
        self.new_boxes  = []
        for b in self.waiting: b.updated = False
    
    def finish(self):
        for b in self.waiting:
            b.unseen = 0 if b.updated else b.unseen + 1
        self.waiting = [b for b in self.waiting if b.unseen <= self.max_unseen] + self.new_boxes
    
    def select_candidates(self, candidates, rows, indices, nested_check = False, ** kwargs):
        def _get_length(indices):
            if isinstance(indices, int): return 1
            return [len(idx) if isinstance(idx, list) else 1 for idx in indices]
        
        indices = [
            i for i, cand in enumerate(candidates)
            if len(cand.indices) == len(indices) and
            (not nested_check or _get_length(cand.indices) == _get_length(indices))
        ]
        if not indices: return [], []
        return indices, np.concatenate([candidates[idx].rows for idx in indices], axis = 0)

    def get_memory_boxes(self, * args, ** kwargs):
        return self.select_candidates(self.memory, * args, nested_check = True, ** kwargs)
    
    def get_waiting_boxes(self, * args, ** kwargs):
        return self.select_candidates(self.waiting, * args, nested_check = True, ** kwargs)

    def is_in_memory(self, rows, ** kwargs):
        indices, boxes = self.get_memory_boxes(rows = rows, ** kwargs)
        valids = get_rows_iou(rows, boxes, threshold = self.iou_threshold)
        if len(valids) and logger.isEnabledFor(logging.DEBUG):
            idx = indices[valids[0]]
            logger.debug('Index {} : match between {} and {}'.format(
                self.index, self.memory[idx], {** kwargs, 'rows' : rows}
            ))
        return len(valids)

    def check_waiting(self, rows, ** kwargs):
        if self.n_repeat <= 1: return True
        
        indices, boxes = self.get_waiting_boxes(rows = rows, ** kwargs)
        matches  = get_rows_iou(rows, boxes, threshold = self.iou_threshold)
        if len(matches) == 0:
            self.new_boxes.append(BoxItem(index = self.index, rows = rows, ** kwargs))
        else:
            if len(matches) > 1:
                logger.warning('Multiple matches detected ! Maybe your threshold is too low')
            
            for idx in matches[::-1]:
                idx = indices[idx]
                self.waiting[idx].repetition += 1
                self.waiting[idx].updated = True
                if self.waiting[idx].repetition >= self.n_repeat:
                    item = self.waiting.pop(idx)
                    if self.use_memory: self.memory.append(item)
                    return True
        return False

    def filter(self, boxes, rows, indices, ** _):
        if isinstance(rows, list):
            return np.array([
                self.filter(box, rows_i, indices_i)
                for box, rows_i, indices_i in zip(boxes, rows, indices)
            ], dtype = bool)

        # Filters out boxes that have already been emitted
        if self.use_memory and self.is_in_memory(rows = rows, box = boxes, indices = indices):
            return False
        # Check if the box was already seen at least `n_repeat - 1` times
        return self.check_waiting(rows = rows, box = boxes, indices = indices)

class RegionFilter(BoxFilter):
    """ Filter out boxes that are not in the given region (`[x_min, y_min, x_max, y_max]`) """
    def __init__(self, region, mode = 'overlap', ** kwargs):
        self.mode   = mode
        self.x_min, self.y_min, self.x_max, self.y_max = convert_box_format(
            np.array(region) if not isinstance(region, dict) else region,
            BoxFormat.XYXY,
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
    def __init__(self,
                 min_h = None,
                 min_w = None,
                 max_h = None,
                 max_w = None,
                 min_area = None,
                 max_area = None,
                 ** kwargs
                ):
        self.min_h, self.max_h = min_h, max_h
        self.min_w, self.max_w = min_w, max_w
        self.min_area, self.max_area = min_area, max_area
    
    def filter(self, boxes, ** kwargs):
        valids = np.ones((len(boxes), ), dtype = bool)
        
        h = boxes[:, 3] - boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        for _min, _max, val in [
            (self.min_h, self.max_h, h),
            (self.min_w, self.max_w, w),
            (self.min_area, self.max_area, h * w)
        ]:
            if _min is not None: valids[val < _min] = False
            if _max is not None: valids[val >= _max] = False

        return valids

def get_rows_iou(rows, boxes, threshold = 0.8):
    if len(boxes) == 0: return []

    # ious has shape `(len(rows), len(boxes))`
    ious = compute_iou(
        rows, boxes, as_matrix = True, use_graph = False, box_mode = BoxFormat.CORNERS
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('IoU between rows and candidates :\n'.format(np.around(ious, decimals = 3)))
    if len(rows) == 1: return np.where(ious[0] > threshold)[0]
    # computes the best IoU for each target box
    ious = np.max(ious, axis = 0)
    # a match requires that all rows from target boxes are matching a row in `rows`
    ious = np.reshape(ious, [-1, len(rows)])
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Reshaped IoU :\n'.format(np.around(ious, decimals = 3)))
    return np.where(np.all(ious > threshold, axis = 1))[0]
