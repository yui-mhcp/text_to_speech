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

import queue
import logging
import threading
import multiprocessing
import multiprocessing.queues

from utils.stream_utils import create_iterator
from .producer import _get_thread_name, _item_to_str, Producer, Event, StoppedException, locked_property
from .priority_queue import PriorityQueue

logger = logging.getLogger(__name__)

STOP    = '__stop__'

_buffers    = {
    'queue' : (queue.Queue, multiprocessing.Queue),
    'fifo'  : (queue.Queue, multiprocessing.Queue),
    'stack' : (queue.LifoQueue, ),
    'lifo'  : (queue.LifoQueue, ),
    'priority'  : (PriorityQueue, multiprocessing.PriorityQueue),
    'min_priority'  : (PriorityQueue, multiprocessing.PriorityQueue),
    'max_priority'  : (PriorityQueue, multiprocessing.PriorityQueue),
}

def get_buffer(buffer, maxsize = 0, use_multiprocessing = False):
    if buffer is None: buffer = 'queue'
    if buffer is not None:
        if isinstance(buffer, str):
            buffer = buffer.lower()
            if buffer not in _buffers:
                raise ValueError('`buffer` is an unknown queue type :\n  Accepted : {}\n  Got : {}\n'.format(tuple(_buffers.keys()), buffer))
            
            idx = 0 if not use_multiprocessing else 1
            buffer = _buffers[buffer][idx](maxsize)
        
        elif not isinstance(buffer, (queues.Queue, multiprocessing.queues.Queue)):
            raise ValueError('`buffer` must be a Queue instance or subclass')

    return buffer

class Consumer(Producer):
    def __init__(self,
                 consumer,
                 *,
                 
                 buffer     = None,
                 buffer_size    = 0,
                 timeout    = None,
                 
                 stateful   = False,
                 init_state = None,
                 
                 keep_result    = False,
                 
                 name = None,
                 
                 ** kwargs
                ):
        """
            Constructor for the class
            
            Arguments :
                - consumer  : the consumer function called on each item
                - buffer    : a `queue.*` class representing the buffer (or its name)
                - stateful  : whether the `consumer` requires a state or not
                    if yes, the function must return a tuple (result, next_state)
                - init_state    : the initial state to pass (for the 1st call)
                - batch_size    : number of items given at once. Each time `consumer` is called, it takes the maximum between the currently available items and `batch_size`
                - keep_result   : whether to keep results or not
                - name  : the `Thread`'s name
            
            /!\ WARNING : if the buffer is a `PriorityQueue`, the order in `results` can differ from the order of `append` due to the priority re-ordering
            In case of `FIFO` buffer, the order will be the same as the `append` order
            In case of `LIFO` buffer, make sure to add items before starting the Consumer if you ant the exact reverse order, otherwise the 1st item will be consumed directly when appened and then it will be in the 1st position of result (and not the last one)
        """
        if hasattr(consumer, '__doc__'): kwargs.setdefault('doc', consumer.__doc__)
        
        super().__init__(generator = self, name = _get_thread_name(consumer, name), ** kwargs)
        
        self.consumer   = consumer
        self.buffer     = get_buffer(buffer, buffer_size)
        self.timeout    = timeout
        self.buffer_type    = buffer if buffer is not None else 'queue'
        
        self.stateful   = stateful
        self._state     = () if init_state is None else init_state
        
        self.keep_result    = keep_result
        self._results       = []
        
        self._in_index  = 0
        self._out_index = 0
        self._stop_empty    = False
    
    stop_empty  = locked_property('stop_empty')
    in_index    = locked_property('in_index')
    out_index   = locked_property('out_index')
    
    @property
    def results(self):
        if not self.keep_result:
            raise RuntimeError("You must set `keep_result` to True to get results")
        return self._results.copy()
    
    @property
    def empty(self):
        with self.mutex: return self._in_index == self._out_index
    
    @property
    def node_text(self):
        des = super().node_text
        if self.buffer_type:     des += "Buffer type : {}\n".format(self.buffer_type)
        if self.batch_size != 1: des += "Batch size : {}\n".format(self.batch_size)
        return des
    
    def _apply_async(self, * args, kwds = {}, callback = None, priority = None, ** kwargs):
        if kwargs and not kwds: kwds = kwargs
        
        with self.mutex:
            if self._stopped: raise StoppedException()
            self._in_index += 1

        self.on_append(* args)
        
        if self.run_main_thread:
            result = self.consume((args, kwds, callback))
            self.on_item_produced(result)
            return result
        
        kwargs = {}
        if 'priority' in self.buffer_type:
            if 'max' in self.buffer_type and isinstance(priority, (int, float)):
                priority = -priority
            kwargs['priority'] = priority
        
        out = AsyncResult(callback = callback)
        self.buffer.put((args, kwds, out), ** kwargs)
        return out
    
    def _map_async(self, items, callback = None, ** kwargs):
        results = []
        for it in items:
            if not isinstance(it, tuple): it = (it, )
            results.append(self._apply_async(* it, kwds = kwargs, callback = callback))
        return results
    
    def consume(self, item = None):
        logger.debug('[Consume {}] Waiting item'.format(self.name))
        if item is None:
            item = self.buffer.get()
            if item is None: raise StopIteration()
        
        args, kwargs, callback = item
        
        logger.debug('[CONSUME {}] {}'.format(self.name, _item_to_str(args)))
        if not self.stateful:
            result = self.consumer(* args, ** kwargs)
        else:
            result, next_state = self.consumer(
                * args, * self._state, ** kwargs
            )
            self._state = next_state
        
        if callback is not None: callback(result)
        return result
    
    __iter__    = lambda self: self
    
    __next__    = consume
    __call__    = _apply_async
    append  = _apply_async
    extend  = _map_async
    
    def run(self):
        if self.run_main_thread:
            self.on_start()
        else:
            super().run()

    def extend_and_wait(self, items, tqdm = lambda x: x, ** kwargs):
        return [res.get() for res in tqdm(self.extend(items, ** kwargs))]
    
    def append_and_wait(self, * args, ** kwargs):
        return self.append(* args, ** kwargs).get()
    
    def on_item_produced(self, item):
        if self.keep_result: self._results.append(item)
        
        super().on_item_produced(item)
        
        with self.mutex:
            self._out_index += 1
            if self._stop_empty and self._in_index == self._out_index:
                logger.debug('[STOP {}] Stop when empty (in : {} - out : {})'.format(
                    self.name, self._in_index, self._out_index
                ))
                self.stop()

    def stop_when_empty(self):
        if self._stop_empty or self._stopped or self._finished: return
        self.stop_empty = True
        if self.empty: self.stop()
    
    def stop(self):
        if self._stopped or self._finished: return
        if self.empty: self.buffer.put(None)
        super().stop()
        if self.run_main_thread: self.on_stop()
    
    def join(self, * args, ** kwargs):
        self.stop_when_empty()
        super().join(* args, ** kwargs)
        
class AsyncResult:
    def __init__(self, callback = None):
        self._callback  = callback
        self._event = threading.Event()
        self._result    = None
    
    @property
    def ready(self):
        return self._event.is_set()
    
    def __call__(self, result):
        self._result    = result
        self._success   = not isinstance(result, Exception)
        if self._callback is not None:
            self._callback(self._result)
        self._event.set()
    
    def wait(self, timeout = None):
        self._event.wait(timeout)
    
    def get(self, timeout = None):
        self.wait(timeout)
        if not self.ready:
            raise TimeoutError
        if self._success:
            return self._result
        else:
            raise self._result
    
    
    