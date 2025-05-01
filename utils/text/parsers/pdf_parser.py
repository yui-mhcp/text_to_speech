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

from loggers import Timer, timer
from .parser import Parser
from .combination import combine_boxes_horizontal, combine_boxes_vertical, compute_union

logger = logging.getLogger(__name__)

class PdfParser(Parser):
    __extensions__ = 'pdf'
    
    def __new__(cls, * args, method = 'pypdfium2', ** kwargs):
        return Pypdfium2Parser(* args, ** kwargs)

class Pypdfium2Parser(Parser):
    def get_text(self, *, pagenos = None, ** kwargs):
        import pypdfium2
        import pypdfium2.raw as pypdfium_c

        with Timer('pdf processing'):
            pdf = pypdfium2.PdfDocument(self.filename)

        if pagenos is None: pagenos = range(len(pdf))

        paragraphs = []
        for page_index in pagenos:
            with Timer('page processing'):
                page = pdf.get_page(page_index)
                
                paragraphs.append({
                    'page' : page_index, 'text' : page.get_textpage().get_text_bounded()
                })
        
        return paragraphs

    def get_paragraphs(self, *, image_folder = None, pagenos = None, ** kwargs):
        """
            Extract texts and images from `filename` with `pdfium2` library

            Arguments :
                - filename  : the `.pdf` document filename
                - pagenos   : list of page numbers to parse
                - image_folder  : where to store the images (with format `image_{i}.jpg`)
            Return :
                - document  : `dict` of pages `{page_index : list_of_paragraphs}`

                A `paragraph` is a `dict` containing the following keys :
                    Text paragraphs :
                    - text  : the paragraph text
                    Image paragraphs :
                    - image : the image path
                    - height    : the image height
                    - width     : the image width
        """
        import pypdfium2
        import pypdfium2.raw as pypdfium_c

        with Timer('pdf processing'):
            pdf = pypdfium2.PdfDocument(self.filename)

        if pagenos is None: pagenos = range(len(pdf))

        filters = (pypdfium_c.FPDF_PAGEOBJ_TEXT, ) if not image_folder else ()

        pages = {}
        for page_index in pagenos:
            with Timer('page processing'):
                page = pdf.get_page(page_index)
                text = page.get_textpage()
                page_w, page_h = int(page.get_width()), int(page.get_height())

                img_num = 0
                paragraphs = []
                for obj in page.get_objects(filters):
                    with Timer('object extraction'):
                        box = obj.get_pos()
                        scaled_box = [int(c) for c in box]
                        scaled_box[1], scaled_box[3] = page_h - scaled_box[3], page_h - scaled_box[1]
                        if obj.type == pypdfium_c.FPDF_PAGEOBJ_TEXT:
                            paragraphs.append({
                                'page' : page_index,
                                'text' : text.get_text_bounded(* box),
                                'box'  : scaled_box,
                                'page_w'    : page_w,
                                'page_h'    : page_h
                            })
                        elif obj.type == pypdfium_c.FPDF_PAGEOBJ_IMAGE and image_folder:
                            if not os.path.exists(image_folder):
                                os.makedirs(image_folder)

                            image_path = os.path.join(
                                image_folder, 'image_{}_{}.jpg'.format(page_index, img_num)
                            )
                            obj.extract(image_path)
                            paragraphs.append({
                                'page'  : page_index,
                                'image' : image_path,
                                'height': obj.height,
                                'width' : obj.width,
                                'box'   : scaled_box
                            })
                            img_num += 1
                
                pages[page_index] = paragraphs
        
        paragraphs = []
        for index, page in pages.items():
            paragraphs.extend(combine_blocks(page, ** kwargs))

        return paragraphs
    

@timer
def combine_blocks(blocks):
    if not blocks: return []
    
    page_w, page_h = blocks[0]['page_w'], blocks[0]['page_h']
    boxes = np.array([l['box'] for l in blocks], dtype = np.float32)
    boxes = boxes / np.array([[page_w, page_h, page_w, page_h]], dtype = np.float32)

    l_col, m_col, r_col = split_in_columns(boxes, np.arange(len(boxes)))

    m_lines = group_column(blocks, boxes, m_col)
    l_lines = group_column(blocks, boxes, l_col)
    r_lines = group_column(blocks, boxes, r_col)
    return m_lines + l_lines + r_lines

@timer
def split_in_columns(boxes, indices):
    r_col_mask = boxes[:, 0] > 0.5
    if np.count_nonzero(r_col_mask) < len(boxes) * 0.25:
        return (indices, [], [])

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

def group_column(blocks, boxes, indices):
    lines, line_boxes = group_words(blocks, boxes, indices)
    return group_lines(lines, line_boxes)

@timer
def group_words(blocks, boxes, indices):
    if len(indices) == 0: return [], []

    boxes = boxes[indices]
    
    h_half = np.mean(boxes[:, 3] - boxes[:, 1]) / 2.
    line_boxes, groups, group_word_boxes = combine_boxes_horizontal(
        boxes, indices = indices, x_threshold = 0.1
    )

    with Timer('grouping words'):
        lines = []
        for line_box, row, word_boxes in zip(line_boxes, groups, group_word_boxes):
            if len(row) == 1:
                lines.append(blocks[row[0]])
                continue
            
            middle = np.median(word_boxes[:, 1]) + h_half
            
            text   = ''
            for idx, word_box in zip(row, word_boxes):
                word = blocks[idx]['text']
                if not word: continue
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
def group_lines(lines, boxes, h_threshold = 1e-2, y_threshold = 75):
    if len(lines) <= 1: return lines
    
    def _is_same_block(i):
        same = abs(h[i] - h[i - 1]) < h_threshold and boxes[i, 1] - boxes[i - 1, 3] < y_threshold_
        if logger.isEnabledFor(logging.DEBUG) and not same:
            logger.debug('New block detected at index #{} : {}'.format(i, lines[i]['text']))
        return same
    
    def _is_indented(i):
        if (x_center[i] - x_center[i - 1]) < 1e-3: return False
        indent = boxes[i, 0] > boxes[i - 1, 0] + 1e-3
        if logger.isEnabledFor(logging.DEBUG) and indent:
            print(np.around(boxes[i - 1 : i + 1], decimals = 3))
            logger.debug('Indentation detected at index #{} : {}'.format(i, lines[i]['text']))
        return indent
    
    indexes = np.argsort(boxes[:, 1])
    boxes   = boxes[indexes]
    lines   = [lines[idx] for idx in indexes]
    
    h = boxes[:, 3] - boxes[:, 1]
    x_center    = (boxes[:, 0] + boxes[:, 2]) / 2.
    
    with Timer('grouping lines'):
        paragraphs, current = [], [0]
        for i, (line, box) in enumerate(zip(lines[1:], boxes[1:]), start = 1):
            y_threshold_ = y_threshold
            if y_threshold_ > 1: y_threshold_ = h[i] * y_threshold_ / 100.
            
            if not _is_same_block(i) or _is_indented(i):
                paragraphs.append({
                    'text'  : '\n'.join([lines[idx]['text'] for idx in current]),
                    'box'   : compute_union(boxes[current])
                })
                current = []
            
            current.append(i)
        
        if current:
            paragraphs.append({
                'text'  : '\n'.join([lines[idx]['text'] for idx in current]),
                'box'   : compute_union(boxes[current])
            })

    paragraphs = [p for p in paragraphs if _is_valid_line(p['text'])]
    return paragraphs

def _is_valid_line(line):
    count = 0
    for c in line:
        count += c.isalnum()
        if count >= 3: return True
    return False
