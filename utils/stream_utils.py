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

import time
import queue
import logging
import multiprocessing

from threading import Event

from .pandas_utils import is_dataframe
from .generic_utils import time_to_string

logger  = logging.getLogger(__name__)

KEEP_ALIVE  = '__keep_alive__'

WARMUP_DELAY_MS = 1

class Iterator:
    def __init__(self,
                 generator,
                 
                 max_items  = None,
                 max_time   = None,
                 item_rate  = None,
                 
                 ** kwargs
                ):
        self.generator  = create_iterable(generator, ** kwargs)
        
        self.item_rate  = item_rate
        self.max_items  = max_items if max_items else -1
        self.max_time   = max_time if max_time else -1
        
        self.n_items    = -1
        self.prev_time  = -1
        self.start_time = -1
        
        self.timer  = Event()
        self.wait_time  = (1. / item_rate - WARMUP_DELAY_MS / 1000) if item_rate else -1
        self.end_times  = []
        self.start_times    = []

    def __repr__(self):
        des = '<Iterator'
        if self.item_rate: des += ' rate={}'.format(self.item_rate)
        return des + '>'
    
    def __iter__(self):
        self.start_time = time.time()
        start_time = self.start_time
        i = -1
        for i, item in enumerate(self.generator()):
            yield item
            
            end_time = time.time()
            self.start_times.append(start_time)
            self.end_times.append(end_time)
            
            if self.max_items != -1 and i + 1 >= self.max_items:
                break
            elif self.max_time != -1 and end_time - self.start_time >= self.max_time:
                break
            elif self.item_rate:
                self.wait(start_time, end_time)
                start_time = time.time()
            else:
                start_time = end_time
        
        if logger.isEnabledFor(logging.DEBUG):
            i += 1
            now = time.time()
            logger.debug('Iterator finished after {} - {} items produced ({:.3f} item(s) / sec)'.format(
                time_to_string(now - self.start_time), i, i / (now - self.start_time)
            ))
            if self.item_rate and i:
                avg_item_time = sum(
                    end - start for start, end in zip(self.start_times, self.end_times)
                ) / len(self.start_times)
                avg_wait_time = sum(
                    start_i_plus_1 - end for end, start_i_plus_1 in zip(
                        self.end_times, self.start_times[1:]
                    )
                ) / (len(self.start_times) - 1)
                logger.debug('  Average wait time : {}\n  Effective rate : {}'.format(
                    time_to_string(avg_wait_time), 1. / avg_item_time
                ))
    
    def __call__(self):
        return iter(self)
    
    def wait(self, prev_time, now = None):
        if now is None: now = time.time()
        wait_time = self.wait_time - (now - prev_time)
        self.timer.clear()
        while wait_time > 0 and self.timer.wait(wait_time):
            self.timer.clear()
            wait_time = self.wait_time - (time.time() - prev_time)
        
    def set_item_rate(self, rate):
        self.item_rate  = rate
        self.wait_time  = 1. / rate - WARMUP_DELAY_MS / 1000
        self.timer.set()
    
        
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
        
        inp = data if not hasattr(data, 'data') else data.data
        if dict_as_kwargs and isinstance(inp, dict):
            res = fn(** {** kwargs, ** inp})
        else:
            res = fn(inp, ** kwargs)
        
        if return_results: results.append(res)
        if callback is not None:
            if not hasattr(data, 'data'):
                callback(res)
            else:
                data.result = res
                callback(data)
    
    if stop_callback is not None: stop_callback()
    
    logger.debug('[STREAM] End')
    
    return results

def create_iterator(generator, ** kwargs):
    if isinstance(generator, Iterator): return generator
    return Iterator(generator, ** kwargs)

def create_iterable(generator, ** kwargs):
    """
        Creates a regular iterator (usable in a `for` loop) based on multiple types
            - `pd.DataFrame`    : iterates on the rows
            - `{queue / multiprocessing.queues}.Queue`  : iterates on the queue items (blocking)
            - `callable`    : generator function
            - else  : returns `generator`
        
        Note : `kwargs` are forwarded to `queue.get` (if `Queue` instance) or to the function call (if `callable`)
    """
    if is_dataframe(generator):
        def _df_iterator():
            for idx, row in generator.iterrows():
                yield row
        return _df_iterator
    elif isinstance(generator, (queue.Queue, multiprocessing.queues.Queue)):
        def _queue_iterator():
            try:
                while True:
                    item = generator.get(** kwargs)
                    if item is not None:
                        yield item
            except queue.Empty:
                pass
            
        return _queue_iterator
    elif callable(generator):
        return lambda: generator(** kwargs)
    elif hasattr(generator, '__len__'):
        def _iterator():
            for i in range(len(generator)):
                yield generator[i]
        return _iterator
    return lambda: generator

def text_input_stream(msg = 'Enter a text :', quit = 'q', ** kwargs):
    """ Creates a generator function asking user input until `quit` is entered """
    txt = input(msg)
    while txt and txt != quit:
        yield txt
        txt = input(msg)
