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

import logging
import numpy as np

from loggers import timer
from .converter import box_converter_wrapper
from .metrics import compute_ioa

logger = logging.getLogger(__name__)

def _compute_groups_np(mask):
    edges = {i : set([i]) for i in range(len(mask))}
    for s, e in zip(* np.where(mask)):
        edges[s].add(e)
        edges[e].add(s)
    
    components, q, seen = [], [], set()
    for i in range(len(mask)):
        if i in seen: continue
        component = set()
        q.append(i)
        while q:
            idx = q.pop(0)
            if idx in seen: continue
            seen.add(idx)
            
            neighbors = edges[idx]

            component.update(neighbors)
            q.extend([n for n in neighbors if n not in seen])
        
        components.append(component)
    return components

def _compute_groups_networkx(mask):
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(zip(* np.where(mask)))
    return nx.connected_components(g)

compute_groups = _compute_groups_np

def _merge_lists(lists):
    if len(lists) == 1: return lists[0]
    union = []
    for l in lists: union.extend(l)
    return union

def _merge_rows(rows, indices):
    """
        Merge boxes in `rows` that are on the same line. This is useful to ensure that, within a paragraph, each box in `rows` correspond to the entire line
    """
    if isinstance(rows, list):
        return list(zip(* [_merge_rows(r, i) for r, i in zip(rows, indices)]))
    elif len(rows) < 2:
        return rows, indices
    
    overlap_y   = np.maximum(0., (
        np.minimum(rows[:, None, 3], rows[None, :, 3]) -
        np.maximum(rows[:, None, 1], rows[None, :, 1])
    )) / np.mean(rows[:, 3] - rows[:, 1])

    groups  = sorted([list(grp) for grp in compute_groups(overlap_y >= 0.5)], key = min)
    rows    = np.array([compute_union(rows[grp]) for grp in groups])
    indices = [
        _merge_lists([indices[idx] for idx in grp])
        for grp in groups
    ]

    return rows, indices

def _align_rows(rows, mode, center_threshold = 1e-2, ** _):
    """ Aligns `rows` on left/right border """
    if len(rows) == 1: return rows
    
    if mode == 'auto':
        x_center = (rows[:, 0] + rows[:, 2]) / 2.
        is_centered = np.all(np.abs(x_center - np.mean(x_center)) < center_threshold)
        if not is_centered: mode = 'left'
    
    if mode == 'left':
        rows[:, 0] = np.min(rows[:, 0])
    elif mode == 'right':
        rows[:, 2] = np.max(rows[:, 2])
    
    return rows

def compute_union(boxes):
    if len(boxes) == 1: return boxes[0]
    return np.concatenate([
        np.min(boxes[:, :2], axis = 0), np.max(boxes[:, 2:], axis = 0)
    ], axis = 0)

def group_boxes(boxes,
                indices,
                groups,
                *,
                sort    = None,
                
                rows    = None,
                
                check_rows  = None,
                align_borders   = None,

                ** kwargs
               ):
    if rows is not None:
        if check_rows is None:      check_rows = True
        if align_borders is None:   align_borders = 'auto'
    
    res_boxes, res_indices, individuals = [], [], []
    for group in groups:
        group = list(group)
        if sort is not None and len(group) > 1:
            group = sorted(group, key = lambda idx: boxes[idx, sort])
        else:
            group = list(group)

        boxes_group = boxes[group]
        res_boxes.append(compute_union(boxes_group))
        
        if rows is None:
            res_indices.append([indices[idx] for idx in group])
            individuals.append(boxes_group)
        elif len(group) == 1:
            res_indices.append(indices[group[0]])
            individuals.append(rows[group[0]])
        else:
            res_indices.append(_merge_lists([indices[idx] for idx in group]))
            individuals.append(np.concatenate([rows[idx] for idx in group], axis = 0))
    
    if check_rows:
        individuals, res_indices = _merge_rows(individuals, res_indices)
    
    if align_borders:
        individuals = [_align_rows(rows, align_borders, ** kwargs) for rows in individuals]
    
    return np.array(res_boxes), res_indices, individuals

@timer
@box_converter_wrapper('xyxy', normalize_mode = 'relative', force_np = True, as_dict = False)
def combine_boxes_horizontal(boxes,
                             *,
                             
                             indices    = None,
                             
                             h_factor   = 1.5,
                             x_threshold    = None,
                             y_overlap_threshold    = 0.55,
                             
                             ** kwargs
                            ):
    """
        Combines a list of boxes according to the following criteria :
            1. The distance between the right-side of box i and the left-side of box j is less than `x_threshold` (if the two boxes overlap on the x axis, the distance is set to 0)
            2. The y-overlap of box i and j divided by the maximal height (of boxes i and j) is greater than `y_overlap_threshold`
        
        The 1st criterion ensures proximity between the two boxes
        The 2nd criterion ensures that both boxes are on the same line with equivalent heights
        
        If `x_threshold` is not provided, it is defined as `median(heights) * h_factor`
        The intuition is that a standard space should be approximately equal to the median height
        
        Arguments :
            - boxes : 2-D `np.ndarray` with shape `(n_boxes, 4)`
            - indices   : a list of indices (default to `range(n_boxes)`)
            - x_threshold   : the threshold for the distance between right/left sides
            - overlap_threshold : fraction of the hgieht that should overlap between the 2 boxes
            - h_factor  : multiplies the median height to define adaptive `x_threshold`
        Return :
            - combined_boxes    : `np.ndarray` with shape `(n_res_boxes, 4)`, the result of the combination
            - combined_indices  : a `list` with length `n_res_boxes`, where item `i` corresponds to values in `indices` used to create `combined_boxes[i]`
            - rows  : a `list` of `np.ndarray` where each item is the concatenation of `boxes` that have been combined together
        
        Example :
        ```python
        boxes = [
            [0, 0, 1, 1],
            [1, 0, 2, 1],
            [3, 3, 4, 4]
        ]
        combined_boxes, groups, rows = combine_boxes_horizontal(boxes)
        # the 1st box is the union of the 2 first boxes
        print(combined_boxes) # [[0, 0, 2, 1], [3, 3, 4, 4]]
        # the 1st groups contains [0, 1] as they have been combined
        print(groups) # [[0, 1], [2]]
        # the 1st group contains the 2 boxes that have been combined
        print(rows) # [array([[0, 0, 1, 1], [1, 0, 2, 1]]), array([[3, 3, 4, 4]])]
        ```
    """
    if indices is None: indices = list(range(len(boxes)))
    if len(boxes) <= 1: return boxes, [indices], [boxes]

    h   = boxes[:, 3] - boxes[:, 1]
    if x_threshold is None:
        x_threshold = np.median(h) * h_factor
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('X threshold : {:.3f}'.format(x_threshold))

    max_h   = np.maximum(h[:, None], h[None, :])
    diff_border = np.abs(boxes[:, None, 2] - boxes[None, :, 0])
    
    overlap_x   = (
        np.minimum(boxes[:, None, 2], boxes[None, :, 2]) -
        np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    ) > 0.
    diff_border = diff_border * ~overlap_x

    overlap_y   = np.maximum(0., (
        np.minimum(boxes[:, None, 3], boxes[None, :, 3]) -
        np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    ))

    should_combine_horizontal = np.logical_and(
        diff_border <= x_threshold,
        overlap_y / max_h >= y_overlap_threshold
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Horizontal combination :\nX distance :\n{}\n% y overlap :\n{}'.format(
            np.around(diff_border, decimals = 3),
            np.around(overlap_y / max_h, decimals = 2)
        ))
    
    return group_boxes(boxes, indices, compute_groups(should_combine_horizontal), sort = 0)

@timer
@box_converter_wrapper('xyxy', normalize_mode = 'relative', force_np = True, as_dict = False)
def combine_boxes_vertical(boxes,
                           *,
                           
                           indices  = None,
                           
                           y_threshold  = None,
                           h_threshold  = 0.02,
                           x_overlap_threshold    = 0.,
                           shift_factor = 0.5,
                           
                           ** kwargs
                          ):
    """
        Combines a list of boxes according to the following criteria :
            1. The distance between the bottom-side of box i and the top-side of box j is less than `y_threshold` or the two boxes overlap on the y axis
            2. The difference in heights between boxes i and j is less than `h_threshold`
            3. The x-overlap of box i and j divided by the maximal width (of boxes i and j) is greater than `x_overlap_threshold` (default to 0, meaning that they simply have to overlap)
        
        The 1st criterion ensures proximity between the two boxes
        The 2nd criterion ensures that both boxes have approximately equivalent heights
        The 3rd criterion ensures that both boxes are overlapping on the x axis
        
        If `y_threshold` is not provided, it is defined as `median(heights) / 4`
        The intuition is that a standard space should be approximately equal to 25% of the median height
        
        The `shift_factor` allows to control in which fraction of the height a box should be
        The intuition is that lines in a paragraph are aligned on the left side
        Therefore, if box i overlaps with box j but is completely on the right side, they may not belong to the same paragraph
        
        Arguments :
            - boxes : 2-D `np.ndarray` with shape `(n_boxes, 4)`
            - indices   : a list of indices (default to `range(n_boxes)`)
            - y_threshold   : the threshold for the distance between bottom/top sides
            - h_threshold   : the threshold for the difference in heights
            - overlap_threshold : fraction of the (shifted) width that should overlap between the 2 boxes
            - shift_factor  : shift the right sode to the left before computing overlap
        Return :
            - combined_boxes    : `np.ndarray` with shape `(n_res_boxes, 4)`, the result of the combination
            - combined_indices  : a `list` with length `n_res_boxes`, where item `i` corresponds to values in `indices` used to create `combined_boxes[i]`
            - rows  : a `list` of `np.ndarray` where each item is the concatenation of `boxes` that have been combined together
        
        Example :
        ```python
        boxes = [
            [0, 0, 1, 1],
            [0, 1, 1, 2],
            [3, 3, 4, 4]
        ]
        combined_boxes, groups, rows = combine_boxes_horizontal(boxes)
        # the 1st box is the union of the 2 first boxes
        print(combined_boxes) # [[0, 0, 1, 2], [3, 3, 4, 4]]
        # the 1st groups contains [0, 1] as they have been combined
        print(groups) # [[0, 1], [2]]
        # the 1st group contains the 2 boxes that have been combined
        print(rows) # [array([[0, 0, 1, 1], [0, 1, 1, 2]]), array([[3, 3, 4, 4]])]
        ```
    """
    if indices is None: indices = list(range(len(boxes)))
    if len(boxes) <= 1: return boxes, [indices], [boxes]
    
    h = boxes[:, 3] - boxes[:, 1]
    if y_threshold is None: y_threshold = max(np.median(h) / 4., 1e-2)
    
    h_diff  = np.abs(h[:, None] - h[None, :])
    diff_border = np.abs(boxes[:, None, 3] - boxes[None, :, 1])

    overlap_y   = (
        np.minimum(boxes[:, None, 3], boxes[None, :, 3]) -
        np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    ) > 0
    diff_border = diff_border * ~overlap_y

    shifted_w   = (boxes[:, 2] - boxes[:, 0]) * shift_factor
    min_w       = np.minimum(shifted_w[:, None], shifted_w[None, :])
    
    shifted_x_max   = boxes[:, 2] - shifted_w
    overlap_x       = (
        np.minimum(shifted_x_max[:, None], shifted_x_max[None, :]) -
        np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    )
    if x_overlap_threshold == 0.:
        x_overlap_mask = overlap_x > 0.
    else:
        overlap_x = overlap_x / np.minimum(shifted_w[:, None], shifted_w[None, :])
        x_overlap_mask = overlap_x > x_overlap_threshold

    should_combine_vertical = np.logical_and(
        np.logical_and(diff_border <= y_threshold, h_diff <= h_threshold),
        x_overlap_mask
    )

    return group_boxes(boxes, indices, compute_groups(should_combine_vertical), sort = 1)

@timer
@box_converter_wrapper('xyxy', normalize_mode = 'relative', force_np = True, as_dict = False)
def combine_boxes_overlap(boxes, indices = None, overlap_threshold = 0.5, ** kwargs):
    is_overlapping = compute_ioa(
        boxes, source = 'xyxy', as_matrix = True
    ) > overlap_threshold
    return group_boxes(boxes, indices, compute_groups(is_overlapping), sort = 1, ** kwargs)

@timer
@box_converter_wrapper('xyxy', normalize_mode = 'relative', force_np = True, as_dict = False)
def combine_boxes(boxes, indices = None, ** kwargs):
    """
        Combines `boxes` (list of e.g., single-word boxes) by creating horizontal then vertical combinations.
        This enables, as an example, to combine the individual words detected by `EAST` to sentences (horizontal lines), then paragraphs (vertically grouping lines)
        
        This method calls sequentially
            1) {combine_boxes_horizontal}
            2) {combine_boxes_vertical}
            3) {combine_boxes_overlap}
    """
    combined_boxes, combined_indices, _ = combine_boxes_horizontal(
        boxes, indices = indices, source = 'xyxy', ** kwargs
    )
    combined_boxes, combined_indices, rows = combine_boxes_vertical(
        combined_boxes, indices = combined_indices, source = 'xyxy', ** kwargs
    )
    combined_boxes, combined_indices, rows = combine_boxes_overlap(
        combined_boxes, indices = combined_indices, rows = rows, source = 'xyxy', ** kwargs
    )

    return combined_boxes, combined_indices, rows
