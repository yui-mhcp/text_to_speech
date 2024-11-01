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

_base_api_url   = 'http://www.bing.com/search'
_user_agent     = 'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0)'

class BingSearchEngine(WebSearchEngine):
    cache_dir   = 'bing'

    @classmethod
    def get_links(cls, query, n, ** kwargs):
        return bing_search(query, ** kwargs)

def bing_search(query, user_agent = _user_agent, ** kwargs):
    from bs4 import BeautifulSoup
    
    url = '{}?q={}'.format(_base_api_url, query)
    res = BeautifulSoup(requests.get(url, headers = {'User-Agent' : user_agent}).text)

    print(res)
    links = []
    for tags in res.find_all('li', attrs = {'class' : 'b_algo'}):
        link = raw.find('a').get('href')
        if link: links.append(link)
    return links
