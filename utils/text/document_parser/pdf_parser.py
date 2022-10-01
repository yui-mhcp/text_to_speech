
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

import os
import re
import shutil
import logging
import numpy as np

from PIL import Image

try:
    from loggers import DEV
except ImportError as e:
    DEV = 11
    logging.addLevelName(DEV, 'DEV')

logger = logging.getLogger(__name__)

_final_punctuation = re.compile(r"[\.\?\!,]\s*$")
_final_space = re.compile(r'\s+$')

def parse_pdf(filename,
              pagenos       = None,
              save_images   = True,
              img_folder    = None,
              
              add_eol_comma = False, # add comma at the end of a line
              skip_margin   = 0.05,
              max_diff_overlap  = 0.01,
              
              tqdm  = lambda x: x,
              ** kwargs
             ):
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTImage, LTChar
    
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    
    if not save_images:
        img_folder = None
    else:
        if img_folder is None: img_folder = os.path.join(os.path.dirname(filename), 'images')
        if os.path.exists(img_folder): shutil.rmtree(img_folder)
        os.makedirs(img_folder, exist_ok = True)
    paragraphes = []
    
    manager         = PDFResourceManager()
    layout_params   = LAParams(all_texts = False, line_margin = 0.001)
    device          = PDFPageAggregator(manager, laparams = layout_params)
    interpreter     = PDFPageInterpreter(manager, device)
    
    pages = {}
    with open(filename, 'rb') as file:
        pages_processor = PDFPage.get_pages(
            file, pagenos = pagenos, check_extractable = True
        )
        
        for i, page in enumerate(tqdm(pages_processor)):
            page_number = i if pagenos is None else pagenos[i]
            
            interpreter.process_page(page)
            page_layout = device.get_result()
            
            text = ''
            paragraphes = []
            last_layout = None
            
            logger.log(DEV, "Page #{} layout : {}".format(page_number, page_layout))
            
            # split page into 2 columns (if page is structured in 2-column format). 
            page_h, page_w = page_layout.bbox[3], page_layout.bbox[2]
            
            logger.log(DEV, "\nMargin filtering...")
            left_col = sorted([
                l for l in page_layout
                if l.bbox[0] < page_w / 2 and not is_in_margin(l, page_h, page_w, skip_margin)
            ], key = lambda l: l.bbox[1], reverse = True)
            right_col = sorted([
                l for l in page_layout
                if l.bbox[0] > page_w / 2 and not is_in_margin(l, page_h, page_w, skip_margin)
            ], key = lambda l: l.bbox[1], reverse = True)
            
            cols  = left_col + [None] + right_col
                
            logger.log(DEV, "\nParagraphs processing")
            # process each layout in page (beginning by left column)
            for layout_idx, l in enumerate(cols):
                # For column change (avoid overlap between last paragraph of left col and 1st paragraph of right col)
                if l is None:
                    last_layout = None
                    continue
                elif isinstance(l, LTTextBox) and len(l.get_text().strip()) == 0:
                    continue
                
                logger.log(DEV, 'Layout {} (box : {}) :\n  {}'.format(
                    layout_idx, [float('{:.2f}'.format(coord)) for coord in l.bbox], l
                ))
                
                overlap = False
                if last_layout is not None:
                    overlap = is_overlap(
                        last_layout, l, page_h = page_layout.bbox[-1],
                        max_diff_overlap = max_diff_overlap
                    )
                
                if (overlap == 0 or isinstance(l, (LTImage, LTFigure))) and len(text) > 0:
                    infos = {'text' : process_paragraph(text)}
                    
                    paragraphes.append(infos)
                    text = ""
                
                if isinstance(l, LTTextBox):
                    next_text = clean_text(l.get_text())
                    if len(text) > 0:
                        text = add_comma_if_needed(text, next_text, add_comma = add_eol_comma)
                    text += next_text
                
                elif isinstance(l, (LTImage, LTFigure)):
                    if img_folder is not None:
                        img_name = 'image_{}.jpg'.format(len(os.listdir(img_folder)))
                        img, (height, width) = extract_lt_image(
                            filename = filename, 
                            page_idx  = page_number, 
                            img_layout    = l, 
                            page_layout   = page_layout,
                            save_name = os.path.join(img_folder, img_name)
                        )
                        if img is not None:
                            paragraphes.append({
                                'type'   : 'image',
                                'name'   : img_name,
                                'height' : height,
                                'width'  : width
                            })
                                        
                last_layout = l
                
            if text:
                infos = {'text' : process_paragraph(text)}
                
                paragraphes.append(infos)
            
            pages[page_number] = paragraphes

    device.close()
    
    return pages

def add_comma_if_needed(txt, next_text, add_comma = True):
    if _final_punctuation.search(txt) is None:
        end = _final_space.search(txt)
        end = end.start() if end is not None else len(txt)
        txt = txt[:end]
        if add_comma or next_text[0].isupper(): txt += ','
    return txt + '\n'

def is_overlap(last, new, page_h, max_diff_overlap = 0.05):
    if isinstance(max_diff_overlap, float): max_diff_overlap = int(max_diff_overlap * page_h)
    _, y0, _, _ = last.bbox
    _, _, _, y1 = new.bbox
    overlap = 1 if abs(y1 - y0) < max_diff_overlap or y1 > y0 else 0
    
    logger.log(DEV, "Distance : {:.3f} (merging {})\n  Layout 1 : {}\n  Layout 2 : {}".format(
        abs(y1 - y0), overlap, last, new
    ))
    return overlap

def is_in_margin(layout, page_h, page_w, skip_margin):
    if not isinstance(skip_margin, (list, tuple)): skip_margin = [skip_margin, skip_margin]
    skip_h, skip_w = skip_margin
    if skip_h > 1.: skip_h = skip_h / page_h
    if skip_w > 1.: skip_w = skip_w / page_w
    
    x0, y0, x1, y1 = layout.bbox
    is_margin = False
    if y0 / page_h < skip_h or y0 / page_h > (1. - skip_h): is_margin = True
    if x0 / page_w < skip_w or x0 / page_w > (1. - skip_w): is_margin = True

    if is_margin:
        logger.log(DEV, "Layout at position ({:.2f}, {:.2f}) is in margin ! (page shape : {})\n  {}".format(
            y0 / page_h, x0 / page_w, [page_h, page_w], layout
        ))
    return is_margin

def clean_text(text):
    return text.replace('\n', '').strip()

def process_paragraph(text):
    if text.count('.') > 50:
        new_text = []
        for line in text.split('\n'):
            parts = [p for p in line.split('.') if len(p.strip()) > 0]
            if len(parts) > 1 or (len(parts) == 1 and not parts[0].isdigit()):
                new_text.append('.'.join(parts))
        text = '\n'.join(new_text)
        
    return text

def extract_lt_image(filename, page_idx, page_layout, img_layout, save_name = None):
    try:
        from pdf2image import convert_from_path

        image = convert_from_path(filename, first_page = page_idx+1, last_page = page_idx+1)
        image = np.asarray(image[0])

        _, _, lt_page_width, lt_page_height = page_layout.bbox
        image_height, image_width, _ = image.shape

        x0, y0, x1, y1 = img_layout.bbox
        x, y, w, h = x0, lt_page_height - y1, x1 - x0, y1 - y0

        x = int((x / lt_page_width) * image_width)
        y = int((y / lt_page_height) * image_height)
        w = int((w / lt_page_width) * image_width)
        h = int((h / lt_page_height) * image_height)

        image = image[y : y + h, x : x + w]

        image = Image.fromarray(image)
        image.save(save_name)

        return image, (image.height, image.width)
    except ImportError as e:
        logger.error("Cannot import pdf2image so cannot process LTImage !\n  Error : {}".format(e))
        return None, (-1, -1)
    except Exception as e:
        logger.error("Error while processing image : {}".format(e))
        return None, (-1, -1)


def save_first_page_as_image_pdf(filename, image_name = 'first_page.jpg'):
    from pdf2image import convert_from_path
    image = convert_from_path(filename, single_file = True, fmt = 'jpg')[0]
    image.save(image_name)
    return image_name
