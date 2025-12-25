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
import logging
import numpy as np

from loggers import Timer, timer
from .parser import Parser

logger = logging.getLogger(__name__)

class PdfParser(Parser):
    __extension__ = 'pdf'
    
    def __new__(cls, * args, method = 'pypdfium2', ** kwargs):
        if method == 'pypdfium2':
            return Pypdfium2Parser(* args, ** kwargs)
        else:
            raise NotImplementedError('The pdf parser {} does not exist !'.format(method))

class Pypdfium2Parser(Parser):
    def get_text(self, *, pages = None, ** kwargs):
        import pypdfium2
        import pypdfium2.raw as pypdfium_c

        with Timer('pdf processing'):
            pdf = pypdfium2.PdfDocument(self.filename)

        if pages is None:               pages = range(len(pdf))
        elif isinstance(pages, int):    pages = [pages]

        paragraphs = []
        for page_index in pages:
            with Timer('page processing'):
                page = pdf.get_page(page_index)
                
                paragraphs.append({
                    'page' : page_index, 'text' : page.get_textpage().get_text_bounded()
                })
        
        return paragraphs

    def get_paragraphs(self,
                       *,
                       
                       pages    = None,
                       raw_content  = None,
                       image_folder = None,
                       header_threshold = 0.1,
                       ** kwargs
                      ):
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
            pdf = pypdfium2.PdfDocument(self.filename if raw_content is None else raw_content)

        if pages is None:               pages = range(len(pdf))
        elif isinstance(pages, int):    pages = [pages]

        filters = (pypdfium_c.FPDF_PAGEOBJ_TEXT, ) if not image_folder else ()

        _pages = {}
        for page_index in pages:
            with Timer('page processing'):
                page    = pdf.get_page(page_index)
                text    = page.get_textpage()
                page_h  = page.get_height()
                page_w  = page.get_width()
                
                img_num = 0
                page_infos = {'paragraphs' : [], 'height' : page_h, 'width' : page_w}
                for obj in page.get_objects(filters):
                    with Timer('object extraction'):
                        box = obj.get_bounds()
                        relative_box = [
                            box[0] / page_w,            # left
                            (page_h - box[3]) / page_h, # top
                            box[2] / page_w,            # right
                            (page_h - box[1]) / page_h  # bottom
                        ]
                        
                        if obj.type == pypdfium_c.FPDF_PAGEOBJ_TEXT:
                            txt = text.get_text_bounded(* box).strip()
                            if (not txt) or (len(txt) == 1 and ord(txt) <= 10):
                                continue
                            
                            page_infos['paragraphs'].append({
                                'text' : txt,
                                'box'  : relative_box,
                                'font_size' : obj.get_font_size()
                            })
                        elif obj.type == pypdfium_c.FPDF_PAGEOBJ_IMAGE and image_folder:
                            if img_num == 0 and not os.path.exists(image_folder):
                                os.makedirs(image_folder)

                            image_path = os.path.join(
                                image_folder, 'image_{}_{}.png'.format(page_index, img_num)
                            )
                            obj.extract(image_path[:-4])
                            
                            page_infos['paragraphs'].append({
                                'type'  : 'image',
                                'image' : image_path,
                                'height': box[3] - box[1],
                                'width' : box[2] - box[0],
                                'box'   : relative_box
                            })
                            img_num += 1
                
                _pages[page_index] = page_infos
        
        paragraphs = []
        for index, page in _pages.items():
            content = combine_blocks(page['paragraphs'], ** kwargs)
            
            font_size = sorted(p['font_size'] for p in content if 'font_size' in p)
            font_size = font_size[len(font_size) // 2]
            for i, para in enumerate(content):
                if i and 'font_size' in para and not para.get('is_footnote', False):
                    if (
                        (font_size - para['font_size'] > 1.5)
                        and (i == len(content) - 1 or para['box'][1] > content[i + 1]['box'][1])
                    ):
                        para['is_footnote'] = True
                
                if 'text' in para and para['box'][1] <= header_threshold and '\n' not in para['text']:
                    para['is_header'] = True
                
                para.update({
                    'page' : index, 'page_h' : page['height'], 'page_w' : page['width']
                })
            
            if content[-1].get('text', '').isdigit():
                content[-1]['is_page_number'] = True
            
            content = sorted(
                content, key = lambda p: _get_paragraph_order_weight(p)
            )
            
            paragraphs.extend(content)
        
        return paragraphs

@timer
def combine_blocks(blocks, ** kwargs):
    if not blocks: return []
    
    lines = group_blocks_in_lines(blocks, ** kwargs)
    paragraphs = group_lines_in_paragraphs(lines, ** kwargs)
    return paragraphs

@timer
def group_blocks_in_lines(blocks, *, factor = 0.6, space_threshold = 0.35, ** _):
    boxes = np.array([b['box'] for b in blocks], dtype = np.float32)

    if 'text' in blocks[0]:
        groups, group, group_indexes = [], [blocks[0]], [0]
    else:
        groups, group, group_indexes = [([blocks[0]], boxes[:1])], [], []
    for i, block in enumerate(blocks[1:], start = 1):
        if (group) and ('text' not in block or not any(_overlap_y(block['box'], g['box']) for g in group)):
            groups.append((group, boxes[group_indexes]))
            group, group_indexes = [], []
        
        group.append(block)
        group_indexes.append(i)
    
    if group: groups.append((group, boxes[group_indexes]))
    
    
    lines = []
    for group, group_boxes in groups:
        if len(group) == 1:
            lines.append(group[0])
            continue
        
        # if 'text' not in blocks, the group is single-item and enters in the above condition
        is_text     = [any(c.isalnum() for c in block['text']) for block in group]
        text_boxes  = group_boxes[is_text]
        if len(text_boxes) == 0: text_boxes = group_boxes
        
        group_h = np.mean(text_boxes[:, 3] - text_boxes[:, 1])
        exp_threshold   = np.median(text_boxes[:, 1]) + group_h * factor
        ind_threshold   = np.median(text_boxes[:, 1]) + group_h * (1 - factor)
        
        text, infos, prev_box, last_is_special = '', {}, None, False
        for block, box in zip(group, group_boxes):
            txt = block['text']
            if txt[0].isalnum():
                # if the top of the box is lower than the line middle, it is an index (e.g., c_i)
                if box[1] >= ind_threshold and ' ' not in txt:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('[Parser] Index detected : {}_{}'.format(text, txt))
                    text += '_'
                    last_is_special = True
                elif box[3] <= exp_threshold and ' ' not in txt:
                    if text.endswith(txt) and txt[-1].isdigit():
                        txt = ''.join(c for c in txt if c.isdigit())
                        text = text[:-len(txt)]

                    if logger.isEnabledFor(logging.DEBUG):
                        if text:
                            logger.debug('[Parser] Exp detected : {}^{}'.format(text, txt))
                        else:
                            logger.debug('[Parser] Footnote detected : {}'.format(txt))
                    
                    if not text:
                        infos.update({'is_footnote' : True, 'footnote_index' : txt})
                    else:
                        infos.setdefault('footnotes', []).append(txt)
                    
                    text += '^'
                    last_is_special = True
                elif (
                    (prev_box is not None) and (
                        last_is_special
                        or not text[-1].isalnum()
                        or abs(box[0] - prev_box[2]) > (prev_box[3] - prev_box[1]) * space_threshold
                    )
                ):
                    text += ' '
                    last_is_special = False
                else:
                    last_is_special = False
            elif txt[0] not in ('.', ',', ')'):
                text += ' '
                
            text += txt
            prev_box = box

        lines.append({
            'text'  : text,
            'box'   : compute_union(group_boxes),
            'font_size' : max(b['font_size'] for b in group),
            ** infos
        })

    return lines

@timer
def group_lines_in_paragraphs(lines, *, indent_threshold = 0.008, y_threshold = 0.8, ** _):
    if len(lines) <= 1: return lines
    
    def _is_indented(box, left):
        return box[0] - left > indent_threshold
    
    boxes = np.array([l['box'] for l in lines], dtype = np.float32)
    
    dist_to_prev = boxes[1:, 1] - boxes[:-1, 3]
    close_to_prev = np.logical_and(dist_to_prev > 0, dist_to_prev < y_threshold)
    
    if 'text' in lines[0]:
        groups, group, group_indexes = [], [lines[0]], [0]
    else:
        groups, group, group_indexes = [([lines[0]], boxes[:1])], [], []
    
    for i, (line, is_close) in enumerate(zip(lines[1:], close_to_prev), start = 1):
        if (group) and (
            'text' not in line
            or abs(line['font_size'] - group[-1]['font_size']) > 1
            or line['box'][1] - group[-1]['box'][3] > (line['box'][3] - line['box'][1]) * y_threshold
            
        ):
            groups.append((group, boxes[group_indexes]))
            group, group_indexes = [], []
        
        group.append(line)
        group_indexes.append(i)
    
    if group: groups.append((group, boxes[group_indexes]))
    
    paragraphs = []
    for i, (group, group_boxes) in enumerate(groups):
        if len(group) == 1:
            paragraphs.append({** group[0], 'lines' : group_boxes})
            continue
        
        left = np.min(group_boxes[:, 0])
        
        text, infos, last_idx = '', {}, 0
        for j, (line, box) in enumerate(zip(group, group_boxes)):
            if (text) and (
                line.get('is_footnote', False)
                or (line['text'][0].isupper() and _is_indented(box, left))):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('[Parser] New paragraph detected : {}'.format(line['text']))
                
                paragraphs.append({
                    'text'  : text,
                    'box'   : compute_union(group_boxes[last_idx : j]),
                    'lines' : group_boxes[last_idx : j],
                    'font_size' : max(l['font_size'] for l in group[last_idx : j]),
                    ** infos
                })
                text, infos, last_idx = '', {}, j
            
            if text:
                if text[-1] != '\x02':
                    text += ' \n'
                else:
                    text = text[:-1]
            
            text += line['text']
            if line.get('is_footnote', False):
                infos.update({'is_footnote' : True, 'footnote_index' : line['footnote_index']})
            elif line.get('footnotes', []):
                infos.setdefault('footnotes', []).extend(line['footnotes'])
        
        if text:
            paragraphs.append({
                'text'  : text,
                'box'   : compute_union(group_boxes[last_idx:]),
                'lines' : group_boxes[last_idx :],
                'font_size' : max(l['font_size'] for l in group[last_idx :]),
                ** infos
            })
    
    return paragraphs

def compute_union(boxes):
    if len(boxes) == 1: return boxes[0]
    return np.concatenate([
        np.min(boxes[:, :2], axis = 0), np.max(boxes[:, 2:], axis = 0)
    ], axis = 0)


def _overlap_y(box1, box2):
    return min(box1[3], box2[3]) - max(box1[1], box2[1]) > 0

def _overlap_x(box1, box2):
    return min(box1[2], box2[2]) - max(box1[0], box2[0]) > 0

def _get_paragraph_order_weight(para):
    if para.get('is_header', False):
        return 0
    elif para.get('is_footnote', False):
        return 2
    elif para.get('is_page_number', False):
        return 3
    else:
        return 1
