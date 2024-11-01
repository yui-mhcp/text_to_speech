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

from utils.generic_utils import import_objects
from utils.wrapper_utils import dispatch_wrapper

_default_engine = 'google'
_search_engines = {}

def get_default_engine():
    global _default_engine
    return _default_engine

def set_default_engine(engine):
    global _default_engine
    _default_engine = engine

@dispatch_wrapper(_search_engines, 'engine')
def search(query, n = 5, engine = None, ** kwargs):
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
    
    if engine not in _search_engines:
        raise ValueError('Unsupported search engine\n  Accepted : {}\n  Got : {}'.format(
            tuple(_search_engines.keys()), engine
        ))
    return _search_engines[engine].search(query, n = n, ** kwargs)

globals().update(
    import_objects(__package__.replace('.', os.path.sep))
)
