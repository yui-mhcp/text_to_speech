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
import queue
import urllib
import logging
import collections

from abc import ABC, abstractmethod
from threading import Thread

from loggers import Timer, timer
from ..parsers import parse_html
from ...generic_utils import time_to_string

logger = logging.getLogger(__name__)

_cache_dir  = os.path.expanduser('~/.cache/yui_mhcp/web')

class SearchEngine(ABC):
    cache_dir   = None
    
    @abstractmethod
    def format_query(self, query, ** kwargs):
        """ Formats a query before passing it to the `self.fetch_urls` method """

    @abstractmethod
    def fetch_urls(self, query, *, n, ** kwargs):
        """
            Returns an iterator of the `n` most relevant urls for the given `query`, returned by the search engine
        """
    
    def __init__(self, ** _):
        pass
    
    @timer
    def search(self,
               query    = None,
               *,
               
               n    = 5,
               urls     = None,
               parse    = True,
               
               save     = False,
               reload   = False,
               reparse  = False,
               
               ** kwargs
              ):
        """
            Generic method method that :
                1) Search `query` on the given search engine (if `urls` is not provided)
                    1.1) Format the query with `self.format_query`
                    1.2) Fetch the `n` most relevant urls with `self.fetch_urls` (if not cached)
                    1.3) Possibly save the mapping `{query : urls}` (if `save = True`)
                2) Process `urls` by fetching their content, and parsing it to extract paragraphs
                   See `process_urls` for more details
            
            Arguments :
                - query : the search query
                
                - n     : maximal number of urls to fetch
                - save  : whether to save best links
                - urls  : `list` of urls to use
                - reload    : whether to force fetching best urls or not
                - reparse   : whether to reparse already processed urls
                
                - kwargs    : forwarded to all downstream methods
            Return :
                - result : a `dict` with the search information
                    - query : the original query
                    - formatted_query   : the formatted query
                    - engine    : the search engine
                    - results   : `dict` of search results for each url `{url : paragraphs}`
                                  `paragraphs` is a `list` of `dict` returned by the parsing method
                                  
            **Important note** : the caching strategy is not always allowed by the engine/website. Set `save` to `True` **only** if you have the permissions to store the results.
        """
        assert query or urls
        
        if not urls:
            with Timer('fetch_urls'):
                formatted_query = self.format_query(query, n = n, ** kwargs)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Formatted query : `{}`'.format(formatted_query))
                
                _cache = {}
                if save:
                    from ...databases import init_database
                    
                    _cache = init_database(
                        'JSONDatabase',
                        path = self.get_cache_path('queries.json'),
                        primary_key = 'query'
                    )
                
                if reload or formatted_query not in _cache:
                    _cache[formatted_query] = {
                        'query' : formatted_query,
                        'urls'  : self.fetch_urls(formatted_query, n = n, ** kwargs)
                    }
                    if save: _cache.save()
                
                urls = _cache[formatted_query]['urls']
                if save: _cache.save()
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('# urls found : {}'.format(len(urls)))
        
        if not parse:
            return urls
        
        results = process_urls(
            urls, reparse = reparse or reload, save = save, ** kwargs
        )
        
        return {
            'query' : query,
            'engine'    : self.__class__.__name__,
            'config'    : kwargs,
            'results'   : results
        }
            
    @classmethod
    def get_cache_path(cls, file):
        return os.path.join(
            _cache_root, cls.cache_dir or cls.__name__.lower().replace('engine', ''), file
        )

class WebSearchEngine(SearchEngine):
    def format_query(self, query, *, exclude_site = 'youtube.com', ** kwargs):
        query = ''.join(c if c.isalnum() else ' ' for c in query).strip()
        if exclude_site: query += ' -site:' + exclude_site
        
        return query
    
@timer
def process_urls(urls,
                 *,

                 n  = None,
                 timeout    = None,
                 best_only  = False,

                 track_href = False,
                 
                 save   = False,
                 reparse    = False,

                 ** kwargs
                ):
    """
        Fetch and process a list of urls in a multi-threaded I/O way
        
        Arguments :
            - urls : the urls to process

            - save  : whether to save parsed urls
            - reparse   : whether to re-fetch and parse urls or not

            - n     : maximal number of links to fetch
            - timeout   : maximum request time
            - best_only : whether to only fetch the `n` best urls or get the `n` fastest to get

            - kwargs    : forwarded to all downstream methods
        Return :
            - parsed    : a mapping `{url : parsed_content}`
    """
    if isinstance(urls, str): urls = [urls]
    if n is None: n = len(urls)
    kwargs['timeout'] = timeout
    
    _cache = {}
    if save:
        from ...databases import init_database

        _cache = init_database(
            'JSONDir', path = os.path.join(_cache_root, 'parsed'), primary_key = 'url'
        )
    
    buffer  = queue.Queue()

    results, workers = {}, []
    for i, url in enumerate(urls):
        if save and not reparse and url in _cache:
            results[i] = _cache[url]
        else:
            workers.append(Thread(
                target = fetch_content, args = (url, buffer, i), kwargs = kwargs, daemon = True
            ))

    additional = 0 if best_only else n // 2
    worker_idx = min(len(workers), n - len(results) + additional)
    for i in range(worker_idx): workers[i].start()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('{} / {} workers started'.format(worker_idx, len(workers)))


    start_time = time.time()
    for _ in range(len(workers)):
        try:
            with Timer('waiting request'):
                idx, url, response = buffer.get(
                    timeout = max(0.01, timeout - ellapsed_time) if timeout and parsed else None
                )
        except queue.Empty:
            logger.info('Timeout exceeded, stopping the search...')
            break

        parsed = {}
        if response:
            with Timer('content parsing'):
                try:
                    parsed = parse_response(
                        response, origin = url if track_href else None, ** kwargs
                    )
                except NotImplementedError:
                    pass

        if parsed:
            for para in parsed: para['url'] = url

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('{} paragraphs parsed from {}'.format(len(parsed), url))

            results[idx] = {'url' : url, 'parsed' : parsed}

            if save: _cache[url] = {'url' : url, 'parsed' : parsed}

            if len(results) == n: break
        elif worker_idx < len(workers):
            workers[worker_idx].start()
            worker_idx += 1

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Start new worker ({} / {})'.format(worker_idx, len(workers)))

    if save: _cache.save()

    return collections.OrderedDict(
        (res[1]['url'], res[1]['parsed']) for res in sorted(results.items())
    )


def fetch_content(url, buffer = None, idx = None, allowed_contents = None, timeout = None, ** _):
    config = {'timeout' : timeout} if timeout else {}
    
    result = None
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, ** config) as response:
            content_type = response.getheader('Content-type')
            if not allowed_contents or content_type in allowed_contents:
                content = response.read()
                if content_type.startswith('text'): content = content.decode()
                
                result = {
                    'content'   : content,
                    'content_type'  : content_type,
                    'last_modified' : response.getheader('last-modified', default = None)
                }
        
    except Exception as e:
        logger.warning('An error occured with url {} : {}'.format(url, e))
    finally:
        if buffer is not None: buffer.put_nowait((idx, url, result))
        logger.info('Time for request #{} : {} - url : {}'.format(
            idx, time_to_string(time.time() - t0), url
        ))
    return result

def parse_response(response, ** kwargs):
    if response['content_type'].startswith('text/html'):
        return parse_html(html = response['content'], ** kwargs)
    else:
        raise NotImplementedError('The content-type {} is not supported yet'.format(
            response['content_type']
        ))