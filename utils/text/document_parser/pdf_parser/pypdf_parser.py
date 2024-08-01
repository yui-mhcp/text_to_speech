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
def parse_pypdf(filename, image_folder = None, pagenos = None, ** kwargs):
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
    @timer
    def visitor(text, cm, tm, font_dict, font_size):
        text = text.strip()
        if text:
            
            box = [
                int(tm[4]),
                int(page_h - tm[5]),
                int(tm[4] + len(text) * font_size / 4.),
                int(page_h - tm[5] + font_size)
            ]
            paragraphs.append({
                'text'  : text,
                'size'  : font_size,
                'box'   : box,
                'page_w'    : page_w,
                'page_h'    : page_h
            })
        
    import pypdf

    pdf = pypdf.PdfReader(filename)
    
    if pagenos is None: pagenos = range(len(pdf.pages))
    
    document = {}
    for page_index in pagenos:
        with time_logger.timer('page processing'):
            page = pdf.pages[page_index]
            
            paragraphs = []
            page_w, page_h = page.mediabox[2:]
            page.extract_text(visitor_text = visitor)

            document[page_index] = paragraphs
    
    return document