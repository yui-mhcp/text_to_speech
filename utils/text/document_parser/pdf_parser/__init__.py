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
import numpy as np

from loggers import timer
from utils.wrapper_utils import dispatch_wrapper
from .post_processing import combine_blocks
from ..parser import parse_document

from .pypdf_parser import parse_pypdf
from .pypdfium2_parser import parse_pypdfium2
from .pdfminer_parser import parse_pdfminer

logger = logging.getLogger(__name__)

_pdf_parsing_methods = {
    'pypdf' : parse_pypdf,
    'pypdfium2' : parse_pypdfium2,
    'pdfminer'  : parse_pdfminer
}

@parse_document.dispatch
@dispatch_wrapper(_pdf_parsing_methods, 'method')
@timer
def parse_pdf(filename, image_folder = None, pagenos = None, method = 'pypdfium2', ** kwargs):
    if method not in _pdf_parsing_methods:
        raise ValueError('Unsupported processing method\n  Accepted : {}\n  Got : {}'.format(
            tuple(_pdf_parsing_methods.keys()), method
        ))
    
    pages = _pdf_parsing_methods[method](
        filename, image_folder, pagenos, ** kwargs
    )
    if all('box' in para for para in list(pages.values())[0]):
        pages = {
            idx : combine_blocks(blocks) for idx, blocks in pages.items()
        }
    return pages
