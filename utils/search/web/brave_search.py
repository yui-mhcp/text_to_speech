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
import requests

from .base_search import WebSearchEngine

_base_api_url   = 'https://api.search.brave.com/res/v1/web/search'

class BraveSearchEngine(WebSearchEngine):
    cache_dir   = 'brave'
    api_key = os.environ.get('BRAVE_SEARCH_API_KEY', None)
    
    @classmethod
    def format_query(cls, query, ** _):
        return query
    
    @classmethod
    def get_links(cls, query, n, api_key = None, ** kwargs):
        if not api_key: api_key = cls.api_key
        if not api_key:
            raise RuntimeError(
                'You must set the BRAVE_SEARCH_API_KEY env variable or provide the `api_key` kwarg'
            )
        request = requests.PrepareRequest()
        request.prepare_url(
            _base_api_url, {** kwargs, 'q' : query}
        )
        if not request.url:
            raise ValueError('Unable to prepare the url')
        
        response = requests.get(request.url, headers = {
            'X-Subscription-Token' : api_key, 'Accept' : 'application/json'
        })
        return [
            item.get('url', None) for item in response.json().get('web', {}).get('results', [])
        ]
