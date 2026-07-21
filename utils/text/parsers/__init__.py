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
import time
import logging

from threading import Lock

from loggers import timer
from utils.generic_utils import import_submodules
from .parser import (
    register_parser, get_paragraph_parser, get_text_parser, supported_extensions
)
from ...file_utils import load_json, dump_json

logger  = logging.getLogger(__name__)

# importing every submodule triggers the `@register_parser` decorators (auto-registration)
import_submodules(__package__)

SUPPORTED_EXTENSIONS    = supported_extensions()
_ext_suffixes           = tuple('.' + ext for ext in SUPPORTED_EXTENSIONS)

__all__ = [
    'parse_document', 'get_document_text', 'normalize_paragraphs',
    'register_parser', 'SUPPORTED_EXTENSIONS'
]

_lock   = Lock()
_cache_dir  = os.path.expanduser('~/.cache/yui_mhcp/parsers')
_dir_to_cache   = {}

@timer
def parse_document(filename,
                   *,

                   recursive    = True,

                   strip    = True,

                   reload   = None,
                   cache_dir    = _cache_dir,
                   extract_images   = True,

                   _cache   = None,

                   ** kwargs
                  ):
    """
        Parse `filename` with the appropriate parser and return its paragraphs

        Arguments :
            - filename  : the file(s) to parse
                          - list    : a list of file/directory
                          - directory   : iterates over all files (possibly recursively)
                          - unix-formatted  : iterates over all files/directories matching the format

            - recursive : whether to expand sub-directories

            - image_folder  : directory to save extracted images
            - extract_images    : whether to extract images from document (not all parsers support this option)

            - strip     : whether to strip texts or not

            - cache_dir : where to save parsed documents (see below for details)
            - reload    : whether to re-process or not (only relevant if `cache_dir is not None`)
                          if `None`, checks the last modification date of the file, and re-process
                          only if it is not equal to the last processing date

            - _cache    : reserved keyword to forward cache between nested calls

            - kwargs    : additional arguments given to the parsing method
        Return :
            - paragraphs    : a `list` of paragraphs (`dict`) extracted from the document
                              see `parser.py` for the canonical description of a `paragraph`

        For the raw-text (non-paragraph) variant, see `get_document_text`.

        if `cache_dir` is provided, the documents are stored in `json` files with a unique ID
            {cache_dir}/
                {doc_id_1}/         # images for file 1
                    image_0.png
                    ...
                {doc_id_1}.json     # parsed file 1
                {doc_id_2}.json     # parsed file 2
                ...
                map.json            # saves metadata + mapping filename->id
    """
    if isinstance(filename, str):
        if '*' in filename:
            filename = glob.glob(filename)
        elif os.path.isdir(filename):
            filename = [os.path.join(filename, f) for f in os.listdir(filename)]

    if _cache is None and cache_dir:
        with _lock:
            if cache_dir not in _dir_to_cache:
                _dir_to_cache[cache_dir] = load_json(os.path.join(cache_dir, 'map.json'), {})
        _cache = _dir_to_cache[cache_dir]

    if isinstance(filename, list):
        filename = [
            f for f in filename if f.lower().endswith(_ext_suffixes) or (recursive and os.path.isdir(f))
        ]

        paragraphs = []
        for file in filename:
            paragraphs.extend(parse_document(
                file,

                recursive   = recursive,

                strip   = strip,

                reload  = reload,
                _cache  = _cache,
                cache_dir   = cache_dir,
                extract_images  = extract_images,

                ** kwargs
            ))

        return paragraphs

    if _cache and reload is not True and filename in _cache:
        if reload is not None or os.path.getmtime(filename) == _cache[filename]['last_modified']:
            return normalize_paragraphs(
                load_json(_cache[filename]['parsed_file']), filename, strip = strip
            )

    ext = filename.rpartition('.')[2]

    parser = get_paragraph_parser(ext)
    if parser is None:
        raise NotImplementedError("No parser found for {} !\n  Accepted : {}".format(
            filename, SUPPORTED_EXTENSIONS
        ))

    if _cache is not None and filename in _cache:
        parsed_file = _cache[filename]['parsed_file']
    else:
        parsed_file = os.path.join(cache_dir, str(time.time()) + '.json')
    image_folder = parsed_file[:-5] if extract_images else None

    try:
        paragraphs = parser(filename, image_folder = image_folder, ** kwargs)

        if _cache is not None:
            os.makedirs(cache_dir, exist_ok = True)
            dump_json(parsed_file, paragraphs)

            _cache[filename]    = {
                'parsed_file'   : parsed_file,
                'last_modified' : os.path.getmtime(filename)
            }
            dump_json(os.path.join(cache_dir, 'map.json'), _cache)
    except Exception as e:
        logger.warning('An exception occured while loading {} : {}'.format(filename, e))
        raise e

    return normalize_paragraphs(paragraphs, filename, strip = strip)

@timer
def get_document_text(filename, *, sep = '\n\n', ** kwargs):
    """
        Return the raw text (`str`) of `filename`, using the fast text-parser when available
        (falls back to joining the paragraphs otherwise).

        Intended for text-based consumers (e.g. LLM), as opposed to `parse_document` which
        returns structured paragraphs (for RAG / TTS). `filename` may be a single file, a
        directory, a unix-formatted path or a list, in which case texts are joined with `sep`.
    """
    if isinstance(filename, str):
        if '*' in filename:
            filename = glob.glob(filename)
        elif os.path.isdir(filename):
            filename = [os.path.join(filename, f) for f in os.listdir(filename)]

    if isinstance(filename, list):
        return sep.join([
            get_document_text(f, sep = sep, ** kwargs)
            for f in filename if f.lower().endswith(_ext_suffixes)
        ])

    parser = get_text_parser(filename.rpartition('.')[2])
    if parser is None:
        raise NotImplementedError("No parser found for {} !\n  Accepted : {}".format(
            filename, SUPPORTED_EXTENSIONS
        ))

    return parser(filename, sep = sep, ** kwargs)

def normalize_paragraphs(paragraphs, filename, *, strip = True, ** kwargs):
    for para in paragraphs:
        if 'type' not in para:
            if 'text' not in para:
                raise RuntimeError('All paragraphs should either have a `type` or `text` entry, got {}'.format(para))
            para['type'] = 'text'

    if strip:
        for para in paragraphs:
            if 'text' in para: para['text'] = para['text'].strip()
        paragraphs = [p for p in paragraphs if p.get('text', True)]

    if paragraphs and 'filename' not in paragraphs[0]:
        for p in paragraphs: p['filename'] = filename

    return paragraphs
