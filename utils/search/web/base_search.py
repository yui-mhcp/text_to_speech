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
import time
import queue
import urllib
import logging
import requests
import collections

from threading import Thread
from abc import ABCMeta, abstractmethod

from loggers import timer, time_logger
from utils.generic_utils import time_to_string
from utils.file_utils import load_json, dump_json
from utils.text.document_parser import _wiki_cleaner
from utils.text.document_parser.html_parser import parse_html

logger = logging.getLogger(__name__)

_cache_root = '.cache/web_search'
_map_files  = {
    'query' : 'queries.json',
    'urls'  : 'urls.json',
    'parsed'    : 'parsed.json'
}
_query_map_file = 'queries.json'
_url_map_file   = 'urls.json'

class SearchEngine:
    cache_dir   = None
    parser_config   = {}
    
    caches  = {
        'query' : None,
        'urls'  : None,
        'parsed'    : None
    }
    
    @classmethod
    @abstractmethod
    def format_query(cls, query, ** kwargs):
        """ Formats a query before passing it to the `cls.make_request` method """

    @classmethod
    @abstractmethod
    def get_links(cls, query, n, ** kwargs):
        """ Returns an iterator of at most `n` links based on the given `query` """
    
    @classmethod
    @abstractmethod
    def parse(cls, data, ** kwargs):
        """ Parses the result of `cls.make_request`, typically with a document parsing method """
    
    @classmethod
    @timer
    def search(cls,
               query    = None,
               *,
               
               save     = False,
               links    = None,
               overwrite    = False,
               
               n    = 5,
               reload   = True,
               timeout  = None,
               best_only    = False,
               save_parsed  = False,
               
               ** kwargs
              ):
        """
            Generic method that searches `query` on the given search engine, gets best urls, and processes them. 
                1) Process the query with `cls.format_query`
                If `links is None` and (`overwrite` or formatted query is not cached) :
                    2) Get the (best) links with `cls.get_links
                For each link in `links` :
                    If `reload` or url is not cached :
                        3) Get the html page with `cls.make_request`
                        4) Parses the result with `cls.parse`
                5) Aggregates all results
            
            Arguments :
                - query : the search query
                
                - save  : whether to save best links
                - links : `list` of urls to use
                - overwrite : whether to re-fetch the links for the given query
                
                - n     : maximal number of links to fetch
                - reload    : whether to reload cached urls
                - timeout   : maximum request time
                - best_only : whether to only fetch the `n` best urls or get the `n` fastest to get
                - save_parsed   : whether to store the parsing result for a given url
                
                - kwargs    : forwarded to all downstream methods
            Return :
                - infos : a `dict` with the search information
                    - query : the original query
                    - formatted_query   : the formatted query
                    - engine    : the search engine
                    - config    : kwargs + default search engine config (`cls.parser_config`)
                    - results   : `dict` of search results for each url `{url : paragraphs}`
                                  `paragraphs` is a `list` of `dict` returned by the parsing method
                                  
            **Important note** : the `save` and `save_parsed` are not always allowed by the engine/website. Set them to `True` **only** if you have the permissions to store the results.
            
        """
        assert query or links
        
        if links is None:
            with time_logger.timer('get_links'):
                formatted_query = cls.format_query(query, n = n, ** kwargs)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Formatted query : `{}`'.format(formatted_query))
                
                links = cls.maybe_update_cache(
                    'query',
                    formatted_query,
                    loader  = lambda: cls.get_links(formatted_query, n = n, ** kwargs),
                    save    = save,
                    overwrite   = overwrite,
                )
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('# links found : {}'.format(len(links)))
            
        elif not isinstance(links, (list, tuple)):
            links   = [links]
        
        if save_parsed: cls.load_cache('parsed')
            
        config  = {** cls.parser_config, ** kwargs}
        parsed  = {}
        buffer  = queue.Queue()
        workers = []
        for i, url in enumerate(links):
            if save_parsed and not overwrite and url in cls.caches['parsed']:
                parsed[i] = {'url' : url, 'parsed' : cls.caches['parsed'][url]}
            else:
                workers.append(Thread(
                    target = cls.make_request, args = (url, buffer, i), kwargs = config, daemon = True
                ))
        
        additional = 0 if best_only else n // 2
        worker_idx = min(len(workers), n - len(parsed) + additional)
        for i in range(worker_idx): workers[i].start()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('{} / {} workers started'.format(worker_idx, len(workers)))
        
        start_time = time.time()
        with time_logger.timer('results parsing'):
            for _ in range(len(workers)):
                try:
                    idx, data = buffer.get(
                        timeout = max(0.01, timeout - ellapsed_time) if timeout and parsed else None
                    )
                except queue.Empty:
                    logger.info('Timeout exceeded, stopping the search...')
                    break
                
                if data: data = cls.parse(data, ** config)

                if data:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('{} paragraphs parsed from {}'.format(len(data), links[idx]))
                    
                    parsed[idx] = {'url' : links[idx], 'parsed' : data}
                    
                    if save_parsed:
                        cls.caches['parsed'] = data
                        cls.save_cache('parsed')
                    
                    if len(parsed) == n: break
                elif worker_idx < len(workers):
                    workers[worker_idx].start()
                    worker_idx += 1
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('Start new worker ({} / {})'.format(worker_idx, len(workers)))
        
        results = collections.OrderedDict()
        for idx, res in list(sorted(parsed.items()))[:n]:
            results[res['url']] = res['parsed']
        
        return {
            'query' : query,
            'engine'    : cls.__name__,
            'config'    : config,
            'results'   : results
        }
    
    @classmethod
    def make_request(cls, url, buffer = None, idx = None, ** _):
        result = None
        t0 = time.time()
        try:
            result = requests.get(url)
        except Exception as e:
            logger.warning('An error occured with url {} : {}'.format(url, e))
        finally:
            if buffer is not None: buffer.put_nowait((idx, result))
            logger.info('Time for request #{} : {} - url : {}'.format(
                idx, time_to_string(time.time() - t0), url
            ))
        return result

    @classmethod
    def maybe_update_cache(cls, cache_key, key, loader, save = True, overwrite = False):
        if not save: return loader()
        
        cls.load_cache(cache_key)
        
        if key not in cls.caches[cache_key] or overwrite:
            cls.caches[cache_key][key] = loader()
            cls.save_cache(cache_key)
        
        return cls.caches[cache_key][key]
        
    @classmethod
    def load_cache(cls, key, cache_dir = None):
        if cls.caches.get(key, '') is not None: return
        
        if cache_dir is None: cache_dir = cls.cache_dir or cls.__name__.lower().replace('engine', '')
        path    = os.path.join(_cache_root, cache_dir)
        os.makedirs(path, exist_ok = True)
        cls.caches[key] = load_json(os.path.join(path, _map_files[key]), default = {})
    
    @classmethod
    def save_cache(cls, key, cache_dir = None):
        if cls.caches.get(key, '') is not None: return
        
        if cache_dir is None: cache_dir = cls.cache_dir or cls.__name__.lower().replace('engine', '')
        path    = os.path.join(_cache_root, cache_dir)
        dump_json(os.path.join(path, _map_files[key]), cls.caches[key], indent = 4)


class WebSearchEngine(SearchEngine):
    _raw_query_options  = ('lang', 'site')
    parser_config       = {
        'skip_header'   : True,
        'skip_footer'   : True,
        'skip_aside'    : True,
        'skip_hrefs'    : True,
        'remove_pattern'    : _wiki_cleaner
    }
    
    @classmethod
    def format_query(cls, query, exclude_site = 'youtube.com', ** kwargs):
        query = ''.join([c if c.isalnum() else ' ' for c in query]).strip()
        for k in cls._raw_query_options:
            if kwargs.get(k, None):
                query += ' {}:{}'.format(k, kwargs[k])
        
        if exclude_site: query += ' -site:' + exclude_site
        
        return query
    
    @classmethod
    def parse(cls, response, ** kwargs):
        if 'html' not in response.headers.get('Content-Type', ''): return None
        return parse_html(response.text, ** kwargs)
