# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading

from queue import Empty, Queue, LifoQueue, PriorityQueue

from utils.stream_utils import create_iterator
from utils.thread_utils.producer import (
    STOP_ITEM, _get_thread_name, Producer, Item, Event, StoppedException, update_item
)

logger = logging.getLogger(__name__)

_queues = {
    'queue' : Queue,
    'fifo'  : Queue,
    'stack' : LifoQueue,
    'lifo'  : LifoQueue,
    'priority'  : PriorityQueue,
    'max_priority' : PriorityQueue,
    'min_priority' : PriorityQueue
}

def get_buffer(buffer, * args, ** kwargs):
    if buffer is not None:
        if isinstance(buffer, str):
            if buffer not in _queues:
                raise ValueError('`buffer` is an unknown queue type :\n  Accepted : {}\n  Got : {}\n'.format(tuple(_queues.keys()), buffer))
            buffer = _queues[buffer](* args, ** kwargs)
        elif not isinstance(buffer, Queue):
            raise ValueError('`buffer` must be a queue.Queue instance or subclass')
    else:
        buffer = Queue(* args, ** kwargs)
    return buffer

class Consumer(Producer):
    def __init__(self,
                 consumer,
                 * args,
                 buffer     = None,
                 buffer_size    = 0,
                 timeout    = None,
                 
                 stateful   = False,
                 init_state = None,
                 
                 batch_size = 1,
                 
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
        if stateful and batch_size != 1:
            raise ValueError('`batch_size = {} and stateful = True` are incompatible !\nWhen using a`stateful  consumer, the `batch_size` must be 1'.format(batch_size))
        
        if hasattr(consumer, '__doc__'): kwargs.setdefault('description', consumer.__doc__)
        
        super().__init__(generator = self, name = _get_thread_name(consumer, name), ** kwargs)
        
        self.consumer   = consumer
        self.buffer     = get_buffer(buffer, buffer_size)
        self.timeout    = timeout
        self.buffer_type    = buffer
        
        self.stateful   = stateful
        self._state     = () if init_state is None else init_state
        
        self.batch_size = batch_size
        
        self.keep_result    = keep_result
        self._results       = []
        
        self._index     = 0
        self._stop_empty    = False
    
    @property
    def is_max_priority(self):
        return self.buffer_type and 'min' not in self.buffer_type
    
    @property
    def append_listeners(self):
        return self._listeners.get(Event.APPEND, [])
    
    @property
    def stop_empty(self):
        with self.mutex_infos: return self._stop_empty

    @property
    def results(self):
        if not self.keep_result:
            raise RuntimeError("You must set `keep_result` to True to get results")
        return self._results.copy()
    
    @property
    def node_text(self):
        des = super().node_text
        if self.buffer_type:     des += "Buffer type : {}\n".format(self.buffer_type)
        if self.batch_size != 1: des += "Batch size : {}\n".format(self.batch_size)
        return des
    
    def __iter__(self):
        return self
    
    def __next__(self):
        logger.debug('[NEXT {}] Waiting for data'.format(self.name))
        item = self.get()

        if item.stop or self.stopped: raise StopIteration()

        logger.debug('[NEXT {}] Consuming data'.format(self.name))
        if not self.stateful:
            result = self.consumer(item.data, * item.args, ** item.kwargs)
        else:
            result, next_state = self.consumer(
                item.data, * item.args, * self._state, ** item.kwargs
            )
            self._state = next_state
        
        self.buffer.task_done()
        if self.keep_result: self._results.append(result)
        return result
    
    def __call__(self, item, * args, ** kwargs):
        """ Equivalent to `self.append` """
        return self.append(item, * args, ** kwargs)

    def empty(self):
        with self.mutex_infos:
            return self._size == self._index
    
    def get(self, raise_empty = False, ** kwargs):
        with self.mutex_infos:
            stop_empty = self._stop_empty
            if stop_empty and self._size == self._index: return STOP_ITEM
        
        try:
            if stop_empty:
                kwargs['block'] = False
                raise_empty     = False
            elif self.timeout:
                kwargs['timeout'] = self.timeout
                raise_empty     = False

            item = self.buffer.get(** kwargs)
        except Empty as e:
            logger.debug('[GET {}] Empty !'.format(self.name))
            if raise_empty: raise e
            return STOP_ITEM
        
        logger.debug('[GET {}] {}'.format(self.name, item))
        return item
    
    def run(self, * args, ** kwargs):
        if not self.run_main_thread: super().run(* args, ** kwargs)
    
    def extend_and_wait(self, items, * args, stop = False, tqdm = None, ** kwargs):
        def append_result(item):
            results.append(item)
            if tqdm is not None: tqdm.update()
        
        def append_and_wake_up(item):
            append_result(item)
            event.set()
        
        if tqdm is not None: tqdm = tqdm(total = len(items), unit = 'item')
        
        results = []
        event   = threading.Event()
        
        for i, item in enumerate(create_iterator(items)):
            callback = append_result if i < len(items) - 1 else append_and_wake_up
            self.append(item, * args, callback = callback, ** kwargs)
        if not self.run_main_thread: event.wait()
        
        if stop: self.stop()
        
        return results
        
    def append_and_wait(self, item, * args, stop = False, ** kwargs):
        def append_and_wake_up(item):
            result.append(item)
            event.set()
        
        result = []
        event  = threading.Event()
        
        self.append(item, * args, callback = append_and_wake_up, ** kwargs)
        if not self.run_main_thread: event.wait()
        if stop: self.stop()
        
        return result[0]
    
    def extend(self, items, * args, ** kwargs):
        return [self.append(item, * args, ** kwargs) for item in create_iterator(items)]
    
    def append(self, item, * args, priority = -1, callback = None, ** kwargs):
        """ Add the item to the buffer (raise ValueError if `stop` has been called) """
        if self.batch_size > 1 and len(args) + len(kwargs) != 0:
            raise ValueError('When using `batch_size` > 1, args / kwargs in `append` must be empty ! Found :\n  Args : {}\n  Kwargs : {}'.format(args, kwargs))
        
        with self.mutex_infos:
            if self.stopped:
                raise StoppedException('Consumer stopped, you cannot add new items !')
            idx = self._index
            self._index += 1
        
        if not isinstance(item, Item):
            item = Item(
                data = item, args = args, kwargs = kwargs, index = idx, callback = callback,
                priority = priority if not self.is_max_priority else -priority
            )
        else:
            item = update_item(item, index = idx, clone = True)
        
        self.on_append(item)
        self.buffer.put(item)
        
        if self.run_main_thread:
            try:
                self.on_item_produced(self.__next__())
            except StopIteration:
                self.stop()
                raise StoppedException('Consumer stopped, you cannot add new items !')
        
        return item

    def stop(self, force = False, ** kwargs):
        if self.stopped: return
        if force: super().stop(** kwargs)
        if self.run_main_thread: self.on_stop()
        return self.stop_when_empty()
    
    def stop_when_empty(self):
        with self.mutex_infos:
            if self._stopped: return
            self._stop_empty = True
            logger.debug('[STATUS {}] Stop when empty !'.format(self.name))
            if self.empty(): self.buffer.put(STOP_ITEM)

    def join(self, * args, ** kwargs):
        """ Stop the thread then wait its end (that all items have been consumed) """
        self.stop_when_empty()
        super().join(* args, ** kwargs)
    
    def wait(self, * args, ** kwargs):
        """ Waits the Thread is finished (equivalent to `join`) """
        return self.join(* args, ** kwargs)

    def on_append(self, item):
        logger.debug('[APPEND {}] {}'.format(self.name, item))
        for l, infos in self.append_listeners:
            l(item) if infos.get('pass_item', False) else l(item.data)
