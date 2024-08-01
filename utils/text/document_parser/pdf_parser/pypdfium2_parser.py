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

import os
import logging

from loggers import timer, time_logger

logger = logging.getLogger(__name__)

@timer
def parse_pypdfium2(filename, image_folder = None, pagenos = None, ** kwargs):
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

    with time_logger.timer('pdf processing'):
        pdf = pypdfium2.PdfDocument(filename)
    
    if pagenos is None: pagenos = range(len(pdf))
    
    filters = (pypdfium_c.FPDF_PAGEOBJ_TEXT, ) if not image_folder else ()
    
    document = {}
    for page_index in pagenos:
        with time_logger.timer('page processing'):
            page = pdf.get_page(page_index)
            text = page.get_textpage()
            page_w, page_h = int(page.get_width()), int(page.get_height())

            img_num = 0
            paragraphs = []
            for obj in page.get_objects(filters):
                with time_logger.timer('object extraction'):
                    box = obj.get_pos()
                    scaled_box = [int(c) for c in box]
                    scaled_box[1], scaled_box[3] = page_h - scaled_box[3], page_h - scaled_box[1]
                    if obj.type == pypdfium_c.FPDF_PAGEOBJ_TEXT:
                        paragraphs.append({
                            'text': text.get_text_bounded(* box),
                            'box' : scaled_box,
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
                            'image' : image_path,
                            'height': obj.height,
                            'width' : obj.width,
                            'box'   : scaled_box
                        })
                        img_num += 1
        
        document[page_index] = paragraphs  
    
    return document