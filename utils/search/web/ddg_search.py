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

from functools import wraps

from . import search
from .base_search import WebSearchEngine

def ddg_search(query, n = 10, timeout = 1, ** kwargs):
    import duckduckgo_search as ddgs
    
    with ddgs.DDGS(timeout = timeout) as engine:
        return [
            it['href'] for it in engine.text(query, max_results = n, region = 'fr')
        ]

@search.dispatch(('DDGSearchEngine', 'ddg', 'duckduckgo'), method = 'get_links', doc = ddg_search)
class DDGSearchEngine(WebSearchEngine):
    cache_dir   = 'ddg'

    @classmethod
    @wraps(ddg_search)
    def get_links(cls, query, n, ** kwargs):
        return ddg_search(query, n = int(n * 1.5), ** kwargs)
