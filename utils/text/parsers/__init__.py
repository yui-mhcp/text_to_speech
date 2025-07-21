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
import glob
import logging
import importlib

from functools import wraps

from loggers import timer
from .parser import Parser

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
    v.__extension__ : v for v in _parsers.values() if hasattr(v, '__extension__')
}
_extensions = tuple(_parsers.keys())

def _get_parse_method(parser):
    @wraps(parser.get_paragraphs)
    def inner(* args, ** kwargs):
        return parser(* args, ** kwargs).get_paragraphs(** kwargs)
    return inner

globals().update({
    'parse_{}'.format(ext) : _get_parse_method(parser)
    for ext, parser in _parsers.items()
})

_cache_dir  = os.path.expanduser('~/.cache/yui_mhcp/parsers')

@timer
def parse_document(filename,
                   *,
                   
                   recursive    = True,
                   
                   image_folder = None,
                   extract_images   = None,
                   
                   strip    = True,
                   return_raw   = False,

                   cache    = True,
                   overwrite    = False,
                   cache_dir    = _cache_dir,
                   
                   _cache   = None,
                   
                   ** kwargs
                  ):
    """
        Parse `filename` with appropriate parser and returns its paragraphes (or raw text as single paragraph)
        
        Arguments : 
            - filename  : the file(s) to parse
                          - list    : a list of file/directory
                          - directory   : iterates over all files (possibly recursively)
                          - unix-formatted  : iterates over all files/directories matching the format
            
            - recursive : whether to expand sub-directories
            
            - image_folder  : directory to save extracted images
            - extract_images    : whether to extract images from document (not all parsers support this option)
            
            - strip     : whether to strip texts or not
            - return_raw    : whether to return raw text or not
            
            - cache : whether to cache parsing result
                      This feature uses the `utils.databases.JSONDir` database to save
                      each parsed result in a different `.json` file within `cache_dir`
            - cache_dir : where to save the cache database
            
            - _cache    : reserved keyword to forward cache between nested calls
            
            - kwargs    : additional arguments given to the parsing method
        Return :
            - paragraphs    : a `list` of paragraphs (`dict`) extracted from the document
            
            A `paragraph` is a `dict` containing the following keys :
                Text paragraphs :
                - type  : the paragraph type (e.g., "text", "image", "list", "table", "code", ...)
                          Each type has some required entries (see below)
                - section       : the section title(s)
                
                "text" / "code" / "title" :
                - text  : the paragraph text
                
                "list" :
                - items : the list of text for each item in the list
                
                "table" :
                - rows  : a list of dict containing each row in the table
                
                "image" :
                - filename  : the image filename
                - height    : the image height
                - width     : the image width
    """
    if isinstance(filename, str):
        if '*' in filename:
            filename = glob.glob(filename)
        elif os.path.isdir(filename):
            filename = [os.path.join(filename, f) for f in os.listdir(filename)]
    
    if cache:
        from ...databases import init_database
        _cache = init_database('JSONDir', path = cache_dir, primary_key = 'filename')

    if isinstance(filename, list):
        filename = [
            f for f in filename if (f.endswith(_extensions)) or (recursive and os.path.isdir(f))
        ]

        _initial_cache_length = len(_cache) if cache else 0
        
        paragraphs = []
        for file in filename:
            paragraphs.extend(parse_document(
                file,
                
                recursive   = recursive,
                image_folder    = image_folder,
                extract_images  = extract_images,
                
                strip   = strip,
                return_raw  = return_raw,
                
                cache   = False,
                _cache  = _cache,
                overwrite   = overwrite,
                
                ** kwargs
            ))
        
        if (cache) and (overwrite or len(_cache) != _initial_cache_length):
            _cache.save()

        return paragraphs
    
    if _cache is not None and not overwrite and filename in _cache:
        return _cache[filename]['paragraphs']
        
    basename, _, ext = filename.rpartition('.')
    if ext not in _parsers:
        raise NotImplementedError("No parser found for {} !\n  Accepted : {}".format(
            filename, _extensions
        ))
    
    if extract_images and image_folder is None:
        image_folder = basename + '-images'
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
    
    if paragraphs and 'filename' not in paragraphs[0]:
        for p in paragraphs: p['filename'] = filename
    
    if _cache is not None:
        _cache[filename] = {'filename' : filename, 'paragraphs' : paragraphs}
        if cache: _cache.save()
    
    return paragraphs
