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
import glob
import logging

from utils.file_utils import load_data, dump_data
from utils.wrapper_utils import dispatch_wrapper
from .parser_utils import *

logger = logging.getLogger(__name__)

_parsing_methods    = {}
_first_page_to_image_methods    = {}

@dispatch_wrapper(_parsing_methods, 'Filename extension')
def parse_document(filename,
                   *,
                   
                   recursive    = True,
                   
                   image_folder = None,
                   extract_images   = None,
                   
                   strip    = True,
                   remove_refs  = True,
                   
                   merge_by = None,
                   group_by = None,
                   max_paragraph_length = None,
                   
                   add_page = False,
                   
                   _cache   = None,
                   cache_file   = None,
                   
                   ** kwargs
                  ):
    """
        Extracts text from a document with supported extension (see below)
        
        Arguments : 
            - filename  : the file to parse
            
            - strip     : whether to strip texts or not
            - merge_by  : whether to merge paragraphs that have the same value for the given key
            - group_by  : whether to group paragraphs by the given key
            - max_paragraph_length  : split texts up to the given maximum length (if needed)
            
            - add_page  : whether to add the `page` key when it is not available (e.g., .txt files)
            
            - zip_filename  : save the output (processed paragraphs + images) in a .zip file

            - kwargs    : additional arguments given to the parsing method
        Return :
            - paragraphs    : the processed document
                If `merge_by` is None: `list` paragraphs
                Else    : `dict` in the following format `{value : list_of_paragraphs}`
                    where `values` are all unique values for the `merge_by` key
            
            A `paragraph` is a `dict` containing the following keys :
                Text paragraphs :
                - text  : the paragraph text
                - title_level   : the title level (if applicable)
                - section       : the section number (if applicable)
                - section_titles    : section titles hierarchy (if applicable)
                
                Image paragraphs :
                - image     : the image filename
                - height    : the image height
                - width     : the image width
    """
    if isinstance(filename, str):
        if '*' in filename:
            filename = glob.glob(filename)
        elif os.path.isdir(filename):
            filename = [os.path.join(filename, f) for f in os.listdir(filename)]
            if not recursive: filename = [f for f in filename if not os.path.isdir(filename)]
    
    if cache_file:
        _cache = load_data(cache_file, default = {})

    if isinstance(filename, list):
        filename = [
            f for f in filename if os.path.isdir(f) or f.endswith(tuple(_parsing_methods))
        ]
        documents = None
        for file in filename:
            paragraphs = parse_document(
                file,
                
                recursive   = recursive,
                image_folder    = image_folder,
                extract_images  = extract_images,
                
                strip   = strip,
                merge_by    = merge_by,
                group_by    = group_by,
                max_paragraph_length    = max_paragraph_length,

                add_page = add_page,
                
                _cache  = _cache,
                
                ** kwargs
            )
            if not documents: documents = paragraphs
            elif isinstance(documents, list): documents.extend(paragraphs)
            else: documents.update(paragraphs)
        
        if cache_file:
            dump_data(cache_file, _cache)

        return documents if documents is not None else []
    
    if _cache and filename in _cache:
        return _cache[filename]
        
    ext = filename.split('.')[-1]
    
    if ext not in _parsing_methods:
        raise ValueError("Unhandled extension !\n  Accepted : {}\n  Got : {} ({})".format(
            tuple(_parsing_methods.keys()), ext, filename
        ))
    
    if extract_images and image_folder is None:
        image_folder = os.path.splitext(filename)[0] + '-images'
    elif extract_images is False:
        image_folder = None
    
    try:
        paragraphs = _parsing_methods[ext](
            filename, image_folder = image_folder, ** kwargs
        )
    except Exception as e:
        logger.warning('An exception occured while loading {} : {}'.format(filename, e))
        raise e
        return []
    
    if isinstance(paragraphs, dict):
        _paragraphs = []
        for page_idx, page_paragraphs in paragraphs.items():
            for p in page_paragraphs: p['page'] = page_idx
            _paragraphs.extend(page_paragraphs)
        paragraphs = _paragraphs
    
    elif add_page or group_by == 'page' or merge_by == 'page':
        paragraphs = infer_pages(paragraphs, ** kwargs)
    
    if strip:
        for para in paragraphs:
            if 'text' in para: para['text'] = para['text'].strip()
        paragraphs = [p for p in paragraphs if p.get('text', None) != '']
    
    if remove_refs:
        paragraphs = remove_references(paragraphs)
    
    if max_paragraph_length:
        paragraphs = split_paragraphs(paragraphs, max_paragraph_length)
    
    for p in paragraphs: p.setdefault('filename', filename)

    if merge_by:
        if any(merge_by not in p for p in paragraphs):
            logger.warning('The `merge_by` key {} is missing in some paragraphs : {}'.format(
                merge_by, [p for p in paragraphs if merge_by not in p]
            ))
        else:
            paragraphs = merge_paragraphs(paragraphs, merge_by)
    
    if group_by:
        if any(group_by not in p for p in paragraphs):
            logger.warning('The `group_by` key {} is missing in some paragraphs'.format(group_by))
        else:
            paragraphs = group_paragraphs(paragraphs, group_by)
    
    if _cache is not None:
        _cache[filename] = paragraphs
    
    if cache_file:
        dump_data(cache_file, _cache)
    
    return paragraphs

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

