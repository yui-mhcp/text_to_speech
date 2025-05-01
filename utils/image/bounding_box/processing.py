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

import warnings
import numpy as np

from ...keras import ops
from ..image_io import load_image
from .converter import _keys_to_convert, box_converter_wrapper, convert_box_format
from .visualization import draw_boxes

def sort_boxes(boxes,
               method = 'top',
               
               threshold    = 0.5,
               columns      = 10,
               
               return_indices = False,
               ** kwargs
              ):
    """
        Sorts the `boxes` based on `method` sorting criterion
        
        Arguments :
            - boxes : the boxes to sort (any format supported by `convert_box_format`)
            - method    : the sorting criterion
                - x / y     : sorts based on the `x` or `y` coordinate of the top-left corner
                - corner    : sorts based on `x + y` coordinate
                - top       : sorts from top to bottom by computing "rows" with tolerance (i.e. 2 boxes may be on the same "line" but not exactly on the same pixel line)
                - left      : sorts from left to right by splitting the image in "columns"
                - score     : sorts in decreasing order of scores (`scores` must be provided)
            
            - threshold : the tolerance threshold for the "top" sorting method
            - columns   : the number of columns for the "left" sorting method
            
            - return_indices    : whether to return the sorted boxes or the sorted indices
            - kwargs    : ignored
        Return :
            If `return_indices == True`:
                - sorted_indices    : list of sorted indices
            Else :
                - sorted_boxes  : same box format as `boxes` sorted
    """
    _boxes = boxes['boxes'] if isinstance(boxes, dict) else boxes
    if len(_boxes) == 0:
        return [] if return_indices else boxes

    x, y, w, h = None, None, None, None
    if method != 'scores':
        _boxes = convert_box_format(boxes, target = 'xywh', as_list = True, ** kwargs)
        if isinstance(_boxes, dict): _boxes = _boxes['boxes']
        if method in ('center', 'left', 'top'):
            _boxes = [ops.cast(coord, 'float32') for coord in _boxes]
        x, y, w, h = _boxes
    elif not isinstance(boxes, dict) or 'scores' not in boxes:
        warnings.warn('`method = "scores"` does not have any effect if scores are not provided')
        return boxes
    
    if method == 'x':       indices = ops.argsort(x, axis = -1)
    elif method == 'y':     indices = ops.argsort(y, axis = -1)
    elif method == 'w':     indices = ops.flip(ops.argsort(w, axis = -1), axis = -1)
    elif method == 'h':     indices = ops.flip(ops.argsort(h, axis = -1), axis = -1)
    elif method == 'corner':    indices = ops.argsort(x + y, axis = -1)
    elif method == 'center':    indices = ops.argsort((x + w / 2.) + (y + h / 2.), axis = -1)
    elif method == 'left':  indices = ops.argsort(ops.round(x * columns) + y, axis = -1)
    elif method == 'score': indices = ops.flip(ops.argsort(boxes['scores'], axis = -1), axis = -1)
    elif method == 'area':  indices = ops.flip(ops.argsort(w * h, axis = -1), axis = -1)
    elif method == 'top':
        x, y, w, h  = [ops.convert_to_numpy(coord) for coord in (x, y, w, h)]
        y_center    = y + h / 2.
        same_rows   = ops.abs(y_center[None, :] - y_center[:, None]) <= h[:, None] * threshold / 2.
        same_rows   = ops.logical_or(same_rows, same_rows.T)
        
        indices = []
        to_set  = np.full((len(x), ), True)
        for idx in ops.argsort(y_center):
            if not to_set[idx]: continue
            
            row_indices = np.where(
                np.logical_and(same_rows[idx], to_set)
            )[0]
            indices.extend(row_indices[ops.argsort(x[row_indices])])
            to_set[row_indices] = False
        indices = np.array(indices, dtype = np.int32)
    else:
        raise ValueError('Unsupported sorting criterion : {}'.format(method))
    
    return indices if return_indices else select_boxes(boxes, indices)

def select_boxes(boxes, indices, axis = -2):
    if not ops.is_array(indices): indices = ops.convert_to_numpy(indices)
    if isinstance(boxes, dict):
        boxes = boxes.copy()
        boxes['boxes'] = select_boxes(boxes['boxes'], indices, axis = -2)
        for k in _keys_to_convert[1:]:
            if k in boxes: boxes[k] = select_boxes(boxes[k], indices, axis = -1)
        return boxes
    
    if len(ops.shape(indices)) == 1: return ops.gather(boxes, indices, axis = 0)
    if axis == -2: indices = indices[..., None]
    return ops.take_along_axis(boxes, indices, axis = axis)

@box_converter_wrapper(
    'xyxy', normalize_mode = 'absolute', as_dict = False, as_list = True, force_np = True
)
def crop_box(image, boxes, show = False, ** kwargs):
    """
        Returns a tuple `(box_image, box_pos)` box `box_pos` is a tuple (x, y, w, h, label, score)
        `box` can be a list of boxes then the result will be a list of the above (single) result
    """
    if isinstance(image, str): image = load_image(image)
    
    if len(boxes) == 0 or len(boxes[0]) == 0:
        return None, None
    
    elif len(boxes[0]) > 1:
        if ops.is_tensorflow_graph():
            raise NotImplementedError('Extracting a list of boxes is not supported in graph-mode')
        
        x0, y0, x1, y1 = boxes
        return boxes, [image[y0[i] : y1[i], x0[i] : x1[i]] for i in range(len(x0))]
    else:
        x0, y0, x1, y1 = [c[0] for c in boxes]
        return boxes, image[y0 : y1, x0 : x1]
