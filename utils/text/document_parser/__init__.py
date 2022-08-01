
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

from utils.text.document_parser.parser_utils import infer_pages, split_paragraphs, clean_paragraphs
from utils.text.document_parser.pdf_parser import parse_pdf, save_first_page_as_image_pdf
from utils.text.document_parser.docx_parser import parse_docx, save_first_page_as_image_docx
from utils.text.document_parser.html_parser import parse_html, _wiki_cleaner
from utils.text.document_parser.txt_parser import parse_txt

def parse_document(filename, max_paragraph_length = None, clean = True, ** kwargs):
    """
        Extract text from a document with handled extension. 
        Arguments : 
            - filename  : the file to parse. 
            - kwargs    : additional arguments given to the parsing method. 
        Return : 
            a dict where keys are page number and values are a list of dict 
                The dict represents paragraphs from this page and have (at least) 'text' field (representing the text for this paragraph). 
                It can also contains additional information such as position, style, ...
        
        Note : all parsing methods have an unused kwargs arguments so that kwargs argument can be passed to all method called without errors.  
    """
    ext = filename.split('.')[-1]
    
    if ext not in _parsing_methods:
        raise ValueError("Unhandled extension !\n  Got : {} (filename = {})\n  Accepted : {}".format(ext, filename, list(_parsing_methods.keys())))
    
    parsed = _parsing_methods[ext](filename, ** kwargs)
    
    if not isinstance(parsed, dict):
        parsed = infer_pages(parsed, ** kwargs)
    
    if clean: parsed = clean_paragraphs(parsed)
    
    if max_paragraph_length is not None:
        parsed = split_paragraphs(parsed, max_paragraph_length)
    
    return parsed

def first_page_to_image(filename, output_dir = None, image_name = 'first_page.jpg'):
    ext = filename.split('.')[-1]
    if output_dir is None: output_dir = os.path.dirname(filename)
    image_filename = os.path.join(output_dir, image_name)
    
    if ext not in _parsing_methods: return None
    
    try:
        return _first_page_to_image_methods[ext](filename, image_name = image_filename)
    except Exception as e:
        print("Error : {}".format(e))
        return None

_parsing_methods = {
    'docx'  : parse_docx,
    'html'  : parse_html,
    'pdf'   : parse_pdf,
    'txt'   : parse_txt
}

_first_page_to_image_methods    = {
    'pdf'   : save_first_page_as_image_pdf,
    'docx'  : save_first_page_as_image_docx
}