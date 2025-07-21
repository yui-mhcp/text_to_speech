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
import importlib

from .search_engine import *

_engines = {}
_default_engine = 'google'

for module in os.listdir(__package__.replace('.', os.path.sep)):
    if module.startswith(('.', '_')) or '_old' in module: continue
    module = importlib.import_module(__package__ + '.' + module.replace('.py', ''))
    
    _engines.update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, SearchEngine)
    })

globals().update(_engines)

_engines = {k.lower().replace('engine', '') : v for k, v in _engines.items()}

def get_default_engine():
    global _default_engine
    return _default_engine

def set_default_engine(engine):
    global _default_engine
    _default_engine = engine

def search_on_web(query, *, n = 5, engine = None, ** kwargs):
    """
        Returns the result of `query` search on the given `engine`
        
        Arguments :
            - query : the query to search
            - n     : the expected number of url to process (it may be less if the engine returns less url)
            - engine    : the search engine to use
        Return :
            - parsed    : `dict` containing the result of the engine's `search` method
                - urls      : list of urls returned
                - texts     : list of text, the paragraphs parsed from the urls
                - titles    : list of paragraphs' title
                - query     : the query
                - engine    : the search engine class
                - config    : the url parsing configuration
    """
    if engine is None: engine = get_default_engine()
    
    if engine not in _engines:
        raise ValueError('Unsupported search engine\n  Accepted : {}\n  Got : {}'.format(
            tuple(_engines.keys()), engine
        ))
    return _engines[engine](** kwargs).search(query, n = n, ** kwargs)

