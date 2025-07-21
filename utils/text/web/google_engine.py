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
import time
import urllib
import logging
import requests

from functools import wraps
from http.cookiejar import LWPCookieJar
from urllib.request import Request, urlopen
from urllib.parse import quote_plus, urlparse, parse_qs

from loggers import Timer, timer
from .search_engine import WebSearchEngine, _cache_dir
from ..parsers.html_parser import _remove_tags

logger = logging.getLogger(__name__)

_user_agent = 'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0)'

_home_url   = 'https://www.google.{domain}/'
_search_url = 'https://www.google.{domain}/search?q={query}&safe={safe}'

_cookie_jar = None

@timer
def google_search(query,
                  n = 5,
                  safe  = 'on',
                  domain    = 'com',
                  
                  pause = 0.1,
                  
                  user_agent    = _user_agent,
                  verify_ssl    = True,
                  ** kwargs
                 ):
    if ' ' in query: query = urllib.parse.quote_plus(query)

    if not _get_cookie_jar():
        _get_page(_home_url.format(domain = domain), user_agent, verify_ssl)
        time.sleep(pause)

    page = _get_page(
        _search_url.format(domain = domain, query = query, safe = safe), user_agent, verify_ssl
    )

    links, visited = [], set()
    for link, link_root in _get_page_links(page):
        if link_root not in visited and not link.endswith('.pdf'):
            visited.add(link_root)
            links.append(link)

    return links

@timer
def _get_page(url, user_agent = _user_agent, verify_ssl = True):
    request = Request(url, headers = {'User-Agent' : user_agent})
    
    cookie_jar = _get_cookie_jar()
    cookie_jar.add_cookie_header(request)
    with urlopen(request) as response:
        cookie_jar.extract_cookies(response, request)
        page = response.read()

    with Timer('cookie saving'):
        try:
            cookie_jar.save()
        except Exception:
            pass
    
    return page

@timer
def _get_page_links(page):
    from bs4 import BeautifulSoup

    if isinstance(page, bytes): page = page.decode()
    page = _remove_tags(page, ('head', 'style', 'table'))
    with Timer('page parsing'):
        soup = BeautifulSoup(page, 'lxml')

        tags = soup.find_all('a')

    links = []
    for tag in tags:
        if not tag.has_attr('href'): continue

        link, infos = _filter_result(tag['href'])
        if link: links.append((link, infos))
    return links

def _filter_result(link):
    try:
        if link.startswith('/url?'):
            o       = urlparse(link, 'http')
            link    = parse_qs(o.query)['q'][0]

        infos = urlparse(link, 'http')
        if infos.netloc and ('google' not in infos.netloc or 'cloud.google' in infos.netloc):
            return link, (infos.netloc, infos.path, infos.params, infos.query)
    
    except Exception:
        pass
    return None, None

@timer
def _get_cookie_jar(root = None, overwrite = False):
    global _cookie_jar
    
    if _cookie_jar is None or overwrite:
        if root is None:
            root    = os.path.join(_cache_dir, 'google')
            os.makedirs(root, exist_ok = True)
        
        _cookie_jar = LWPCookieJar(os.path.join(root, '.google-cookie'))
        try:
            _cookie_jar.load()
        except Exception as e:
            logger.info('Error while loading cookie jar : {}'.format(e))
            pass

    return _cookie_jar

class GoogleEngine(WebSearchEngine):
    @wraps(google_search)
    def fetch_urls(self, query, n, ** kwargs):
        return google_search(query, n = n, ** kwargs)
