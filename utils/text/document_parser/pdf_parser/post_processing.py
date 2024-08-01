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

from loggers import timer, time_logger
from .combination import combine_boxes_horizontal, combine_boxes_vertical, compute_union

logger = logging.getLogger(__name__)

def _is_valid_line(line):
    count = 0
    for c in line:
        count += c.isalnum()
        if count >= 3: return True
    return False

@timer
def split_in_columns(boxes, indices):
    r_col_mask = boxes[:, 0] > 0.5
    if np.count_nonzero(r_col_mask) < len(boxes) * 0.25:
        return (indices, [], [])

    middle_mask = np.abs((boxes[:, 0] + boxes[:, 2]) / 2. - 0.5) <= 0.05
    middle_mask = np.logical_and(
        boxes[:, 0] < 0.3,
        np.abs((boxes[:, 0] + boxes[:, 2]) / 2. - 0.5) <= 0.1
    )
    
    if not np.any(middle_mask[:5]):
        return (indices[~r_col_mask], [], indices[r_col_mask])

    middle_bottom = np.max(boxes[middle_mask, 3])
    middle_mask   = np.logical_or(middle_mask, boxes[:, 1] < middle_bottom)
    l_col_mask = np.logical_and(~r_col_mask, ~middle_mask)
    r_col_mask = np.logical_and(r_col_mask, ~middle_mask)
    return (indices[l_col_mask], indices[middle_mask], indices[r_col_mask])

@timer
def group_words(blocks, boxes, indices):
    if len(indices) == 0: return [], []

    boxes = boxes[indices]
    
    h_half = np.mean(boxes[:, 3] - boxes[:, 1]) / 2.
    line_boxes, groups, group_word_boxes = combine_boxes_horizontal(
        boxes, indices = indices, x_threshold = 0.1
    )

    with time_logger.timer('grouping words'):
        lines = []
        for line_box, row, word_boxes in zip(line_boxes, groups, group_word_boxes):
            if len(row) == 1:
                lines.append(blocks[row[0]])
                continue
            
            middle = np.median(word_boxes[:, 1]) + h_half
            
            text   = ''
            for idx, word_box in zip(row, word_boxes):
                word = blocks[idx]['text']
                if word[0].isalnum():
                    if word_box[1] >= middle - 1e-3:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('Index detected : {} - {}'.format(text, word))
                        text += '_'
                    elif word_box[3] <= middle + 1e-3:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('Exp detected : {} - {}'.format(text, word))
                        text += '^'
                    else:
                        text += ' '
                text += word
            
            lines.append({'text' : text, 'box' : line_box})

    return lines, line_boxes

@timer
def group_lines(lines, boxes):
    if len(lines) == 0: return []
    
    para_boxes, groups, group_para_boxes = combine_boxes_vertical(boxes)

    col_left = 1.
    if len(lines) > 5:
        col_left = np.mean(boxes[:, 0]) + 1e-3
    
    with time_logger.timer('grouping lines'):
        paragraphs = []
        for para_box, row, line_boxes in zip(para_boxes, groups, group_para_boxes):
            if len(row) == 1:
                paragraphs.append(lines[row[0]])
                continue

            texts, cut_idx = [], 0
            for i, (idx, line_box) in enumerate(zip(row, line_boxes)):
                if line_box[0] > col_left:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('Indentation detected : {}'.format(lines[idx]['text']))
                    if i:
                        paragraphs.append({
                            'text'  : '\n'.join(texts),
                            'box'   : compute_union(line_boxes[cut_idx : i])
                        })
                    texts, cut_idx = [], i
                
                texts.append(lines[idx]['text'])

            if cut_idx: para_box = compute_union(line_boxes[cut_idx :])
            paragraphs.append({'text' : '\n'.join(texts), 'box' : para_box})

    paragraphs = [p for p in paragraphs if _is_valid_line(p['text'])]
    return paragraphs

def group_column(blocks, boxes, indices):
    lines, line_boxes = group_words(blocks, boxes, indices)
    return group_lines(lines, line_boxes)

@timer
def combine_blocks(blocks):
    page_w, page_h = blocks[0]['page_w'], blocks[0]['page_h']
    boxes = np.array([l['box'] for l in blocks], dtype = np.float32)
    boxes = boxes / np.array([[page_w, page_h, page_w, page_h]], dtype = np.float32)

    l_col, m_col, r_col = split_in_columns(boxes, np.arange(len(boxes)))

    m_lines = group_column(blocks, boxes, m_col)
    l_lines = group_column(blocks, boxes, l_col)
    r_lines = group_column(blocks, boxes, r_col)
    return m_lines + l_lines + r_lines
