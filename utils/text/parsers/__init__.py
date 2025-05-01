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
import importlib

from loggers import timer
from .parser import Parser
from ...wrappers import dispatch_wrapper
from ...file_utils import load_data, dump_data

logger  = logging.getLogger(__name__)

_parsers = {}
for module in os.listdir(__package__.replace('.', os.path.sep)):
    if module.startswith(('.', '_')) or '_old' in module: continue
    module = importlib.import_module(__package__ + '.' + module.replace('.py', ''))
    
    _parsers.update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, Parser)
    })

globals().update(_parsers)

_parsers = {
    v.__extensions__ : v for v in _parsers.values() if hasattr(v, '__extensions__')
}

@dispatch_wrapper(_parsers, 'Filename extension')
@timer
def parse_document(filename,
                   *,
                   
                   recursive    = True,
                   
                   image_folder = None,
                   extract_images   = None,
                   
                   strip    = True,
                   merge_by = None,
                   
                   return_raw   = False,

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
    
    if cache_file and _cache is None:
        _cache = load_data(cache_file, default = {})

    if isinstance(filename, list):
        filename = [f for f in filename if os.path.isdir(f) or f.endswith(tuple(_parsing_methods))]
        
        documents = None
        for file in filename:
            paragraphs = parse_document(
                file,
                
                recursive   = recursive,
                image_folder    = image_folder,
                extract_images  = extract_images,
                
                strip   = strip,
                merge_by    = merge_by,
                return_raw  = return_raw,
                
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
    if ext not in _parsers:
        raise ValueError("Unhandled extension !\n  Accepted : {}\n  Got : {} ({})".format(
            tuple(_parsers.keys()), ext, filename
        ))
    
    if extract_images and image_folder is None:
        image_folder = os.path.splitext(filename)[0] + '-images'
    elif extract_images is False:
        image_folder = None
    
    try:
        parser = _parsers[ext](filename)
        if return_raw:
            paragraphs = parser.get_text(** kwargs)
        else:
            paragraphs = parser.get_paragraphs(image_folder = image_folder, ** kwargs)
    
    except Exception as e:
        logger.warning('An exception occured while loading {} : {}'.format(filename, e))
        raise e
    
    if strip:
        for para in paragraphs:
            if 'text' in para: para['text'] = para['text'].strip()
        paragraphs = [p for p in paragraphs if p.get('text', None) != '']
    
    for p in paragraphs: p.setdefault('filename', filename)

    if merge_by:
        if any(merge_by not in p for p in paragraphs):
            logger.warning('The `merge_by` key {} is missing in some paragraphs : {}'.format(
                merge_by, [p for p in paragraphs if merge_by not in p]
            ))
        else:
            paragraphs = merge_paragraphs(paragraphs, merge_by)
    
    if _cache is not None:
        _cache[filename] = paragraphs
    
    if cache_file:
        dump_data(cache_file, _cache)
    
    return paragraphs

def merge_paragraphs(paragraphs, key, sep = '\n\n', pop_keys = ('section', 'section_titles')):
    merged = paragraphs[0].copy()
    result = [merged]
    for para in paragraphs[1:]:
        if para[key] == merged[key]:
            for k, v in para.items():
                if k not in merged:
                    merged[k] = v
                elif k == 'text':
                    merged['text'] = merged['text'] + sep + v
                elif para[k] != merged[k]:
                    if isinstance(merged[k], list):
                        merged[k].append(v)
                    else:
                        merged[k] = [merged[k], v]
        else:
            merged = para.copy()
            result.append(merged)
    
    pop_keys = [k for k in pop_keys if key not in k]
    for res in result:
        for k in pop_keys: res.pop(k, None)
    
    return result
