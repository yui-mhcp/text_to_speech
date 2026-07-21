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

import logging

logger = logging.getLogger(__name__)

"""
    A `parser` is a plain function `(filename, ** kwargs) -> list[paragraph]` (or `-> str` for
    the `text` variant). Parsers are registered by extension with the `register_parser` decorator,
    and auto-discovered by importing every submodule of this package (see `__init__.py`).

    A `paragraph` is a `dict` with (at least) :
        - type  : the type of content (defaults to `text` if omitted, see `normalize_paragraphs`)
        - text  : the raw text (for `text` / `code` paragraphs)

    Each `type` has some specific keys :
        - `text`  : `text`
        - `code`  : `text` + optional `language`
        - `image` : `image` (path) + optional `height` / `width`
        - `table` : `rows` (`list` of `{column : value}`)
        - `list`  : `items` (`list` of `str`)

    Optional keys (format-dependent) : `page`, `box`, `title`, `section`, `filename`.
"""

_paragraph_parsers  = {}    # ext -> fn(filename, ** kwargs) -> list[paragraph]
_text_parsers       = {}    # ext -> fn(filename, ** kwargs) -> str

def _normalize_ext(ext):
    return ext.lower().lstrip('.')

def register_parser(* extensions, kind = 'paragraphs'):
    """
        Register `fn` as the parser for the given `extensions`

        Arguments :
            - extensions    : the file extension(s) handled by the decorated function
            - kind          : either `paragraphs` (`-> list[paragraph]`) or `text` (`-> str`)
    """
    assert kind in ('paragraphs', 'text'), 'Unknown parser kind : {}'.format(kind)

    target = _paragraph_parsers if kind == 'paragraphs' else _text_parsers
    def wrapper(fn):
        for ext in extensions: target[_normalize_ext(ext)] = fn
        return fn
    return wrapper

def supported_extensions():
    """ Return the tuple of extensions with a registered `paragraphs` parser """
    return tuple(_paragraph_parsers.keys())

def get_paragraph_parser(ext):
    """ Return the `paragraphs` parser for `ext`, or `None` if none is registered """
    return _paragraph_parsers.get(_normalize_ext(ext), None)

def get_text_parser(ext):
    """
        Return the `text` parser for `ext`.
        If no dedicated fast text-parser is registered, falls back to joining the paragraphs
        produced by the `paragraphs` parser. Returns `None` if the extension is unknown.
    """
    ext = _normalize_ext(ext)
    if ext in _text_parsers: return _text_parsers[ext]

    paragraph_parser = _paragraph_parsers.get(ext, None)
    if paragraph_parser is None: return None

    def fallback(filename, *, sep = '\n\n', ** kwargs):
        return sep.join(
            para['text'] for para in paragraph_parser(filename, ** kwargs) if 'text' in para
        )
    return fallback
