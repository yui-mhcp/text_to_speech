# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import queue
import logging
import multiprocessing
import pandas as pd

KEEP_ALIVE  = '__keep_alive__'

def create_stream(fn,
                  stream,
                  timeout   = None,
                  
                  callback  = None,
                  stop_callback = None,
                  
                  logger    = None,
                  return_results    = False,
                  dict_as_kwargs    = True,
                  ** kwargs
                 ):
    """
        Creates a streaming iteration of `fn` based on `stream`
        
        Example :
        ```python
            for item in create_iterator(stream):
                res = fn(item)
                if callback: callback(res)
        ```
        
        Arguments :
            - fn    : the function to call on each `stream` item
            - stream    : regular stream accepted by `create_iterator` (e.g., `Queue` instance, generator function, ...)
            - timeout   : waiting timeout if `stream` is a `Queue` instance
            
            - callback  : function called on each result
            - stop_callback : function called at the end of the stream
            
            - logger    : custom logger to use
            - return_results    : whether to return a list of results or not
            - dict_as_kwargs    : if `True` and `item` is a `dict`, it is interpreted as kwargs when calling `fn` (i.e. `fn(** item)`)
        Return :
            - None if `return_results == False` else `list` of results
    """
    if logger is None: logger = logging.getLogger(__name__)
    
    if callback is not None and not callable(callback):
        if hasattr(callback, 'put'): callback = callback.put
        else: raise ValueError('Callback {} must be callable or have a `put` method'.format(callback))
    
    logger.debug('[STREAM] Start...')
    
    results = [] if return_results else None
    for data in create_iterator(stream, timeout = timeout):
        if isinstance(data, str) and data == KEEP_ALIVE: continue
        if dict_as_kwargs and isinstance(data, dict):
            res = fn(** {** kwargs, ** data})
        else:
            res = fn(data, ** kwargs)
        
        if return_results: results.append(res)
        if callback is not None: callback(res)
    
    if stop_callback is not None: stop_callback()
    
    logger.debug('[STREAM] End')
    
    return results

def create_iterator(generator, ** kwargs):
    """
        Creates a regular iterator (usable in a `for` loop) based on multiple types
            - `pd.DataFrame`    : iterates on the rows
            - `{queue / multiprocessing.queues}.Queue`  : iterates on the queue items (blocking)
            - `callable`    : generator function
            - else  : returns `generator`
        
        Note : `kwargs` are forwarded to `queue.get` (if `Queue` instance) or to the function call (if `callable`)
    """
    if isinstance(generator, pd.DataFrame):
        def _df_iterator():
            for idx, row in generator.iterrows():
                yield row
        return _df_iterator()
    elif isinstance(generator, (queue.Queue, multiprocessing.queues.Queue)):
        def _queue_iterator():
            try:
                while True:
                    item = generator.get(** kwargs)
                    if item is not None:
                        yield item
            except queue.Empty:
                pass
            
        return _queue_iterator()
    elif callable(generator):
        return generator(** kwargs)
    return generator

def text_input_stream(msg = 'Enter a text :', quit = 'q', ** kwargs):
    """ Creates a generator function asking user input until `quit` is entered """
    txt = input(msg)
    while txt and txt != quit:
        yield txt
        txt = input(msg)
