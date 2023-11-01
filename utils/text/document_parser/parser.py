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
import logging

from utils.wrapper_utils import dispatch_wrapper
from utils.text.document_parser.parser_utils import infer_pages, split_paragraphs, clean_paragraphs

logger = logging.getLogger(__name__)

_parsing_methods    = {}
_first_page_to_image_methods    = {}

@dispatch_wrapper(_parsing_methods, 'Filename extension')
def parse_document(filename, max_paragraph_length = None, clean = True, ** kwargs):
    """
        Extracts text from a document with supported extension (see below)
        
        Arguments : 
            - filename  : the file to parse
            - clean     : whether to clean paragraphs or not
            - max_paragraph_length  : the maximal length for a paragraph
            - kwargs    : additional arguments given to the parsing method
        Return :
            - document  : `dict` with `{page_number : paragraphs`
                - paragraphs    : `list` of `dict`, list of paragraphs information
                
        Note : all parsing methods have an unused kwargs arguments so that kwargs argument can be passed to all method called without errors
    """
    ext = filename.split('.')[-1]
    
    if ext not in _parsing_methods:
        raise ValueError("Unhandled extension !\n  Got : {} (filename = {})\n  Accepted : {}".format(
            ext, filename, tuple(_parsing_methods.keys())
        ))
    
    parsed = _parsing_methods[ext](filename, ** kwargs)
    
    if not isinstance(parsed, dict):
        parsed = infer_pages(parsed, ** kwargs)
    
    if clean: parsed = clean_paragraphs(parsed)
    
    if max_paragraph_length:
        parsed = split_paragraphs(parsed, max_paragraph_length)
    
    return parsed

@dispatch_wrapper(_first_page_to_image_methods, 'Filename extension')
def first_page_to_image(filename, output_dir = None, image_name = 'first_page.jpg'):
    ext = filename.split('.')[-1]
    if output_dir is None: output_dir = os.path.splitext(filename)[0] + '_images'
    image_filename = os.path.join(output_dir, image_name)
    
    if ext not in _parsing_methods: return None
    
    try:
        return _first_page_to_image_methods[ext](filename, image_name = image_filename)
    except Exception as e:
        logger.error("Error while converting 1st page to image : {}".format(e))
        return None

