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

import heapq
import logging

from threading import Thread, RLock, Semaphore
from queue import Empty, Queue, LifoQueue, PriorityQueue

from utils.thread_utils.producer import STOP_ITEM, _get_thread_name, _create_generator, Producer, Event, Item, StoppedException, update_item
from utils.thread_utils.threaded_dict import ThreadedDict

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

def get_buffer(buffer):
    if buffer is not None:
        if isinstance(buffer, str):
            if buffer not in _queues:
                raise ValueError('`buffer` is an unknown queue type :\n  Accepted : {}\n  Got : {}\n'.format(tuple(_queues.keys()), buffer))
            buffer = _queues[buffer]()
        elif not isinstance(buffer, Queue):
            raise ValueError('`buffer` must be a queue.Queue instance or subclass')
    else:
        buffer = Queue()
    return buffer

class Consumer(Producer):
    def __init__(self,
                 consumer,
                 * args,
                 buffer     = None,
                 
                 stateful   = False,
                 init_state = None,
                 
                 batch_size = 1,
                 
                 max_workers    = 0,
                 keep_result    = False,
                 allow_multithread  = True,
                 
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
                - max_workers   : number of parallel threads calling `consumer`
                    if -2   : call `consumer` directly in `append()` and `start()` does nothing
                    if -1   : call `__next__` directly in `append()` and `start()` does nothing
                    if 0    : call `consumer` when required by the iteration in the Consumer's thread
                    0 <     : call `consumer` in at most `n` separate threads
                - keep_result   : whether to keep results or not
                - name  : the `Thread`'s name
            
            Note that `max_workers == -2` skips all features of the `Consumer` class to execute everything sequentially without even using the `buffer`
            
            /!\ WARNING : if the buffer is a `PriorityQueue`, the order in `results` can differ from the order of `append` due to the priority re-ordering
            In case of `FIFO` buffer, the order will be the same as the `append` order
            In case of `LIFO` buffer, make sure to add items before starting the Consumer if you ant the exact reverse order, otherwise the 1st item will be consumed directly when appened and then it will be in the 1st position of result (and not the last one)
        """
        if stateful:
            if batch_size != 1:
                raise ValueError('`batch_size = {} and stateful = True` are incompatible !\nWhen using a`stateful  consumer, the `batch_size` must be 1'.format(batch_size))
            elif max_workers > 1:
                raise ValueError('`max_workers = {} and stateful = True` are incompatible !\nWhen using a`stateful  consumer, the `max_workers` must be <= 1'.format(max_workers))
        
        if batch_size == 0: batch_size = 1
        elif batch_size < -1:
            raise ValueError('Only `batch_size = -1 or batch_size > 0` are allowed')
        
        if hasattr(consumer, '__doc__'): kwargs.setdefault('description', consumer.__doc__)
        
        Thread.__init__(self, name = _get_thread_name(consumer, name))
        super(Consumer, self).__init__(
            generator = self, run_main_thread = max_workers < 0, ** kwargs
        )
        
        self.consumer   = consumer
        self.buffer     = get_buffer(buffer)
        self.buffer_type    = buffer

        self.stateful   = stateful
        self.__state    = () if init_state is None else init_state
        
        self.batch_size = batch_size if batch_size != 0 else 1
        
        self.keep_result    = keep_result
        self._results       = ThreadedDict()
        
        self.max_workers    = max_workers if allow_multithread else min(max_workers, 0)
        
        self._current_index     = 0
        self._next_index    = 0
        self._last_index    = 0
        self._stop_index    = -1
        
        self.__stop_empty   = False
        
        self.mutex_get  = RLock()
        self.workers    = []
    
    @property
    def append_listeners(self):
        return self._listeners.get(Event.APPEND, [])
    
    @property
    def is_max_priority(self):
        return isinstance(self.buffer_type, str) and 'max' in self.buffer_type
    
    @property
    def listener(self):
        return self
    
    @property
    def current_index(self):
        with self.mutex_infos: return self._current_index
    
    @property
    def next_index(self):
        with self.mutex_infos: return self._next_index
    
    @property
    def last_index(self):
        with self.mutex_infos: return self._last_index

    @property
    def stop_index(self):
        with self.mutex_infos: return self._stop_index
    
    @property
    def stop_empty(self):
        with self.mutex_infos: return self.__stop_empty

    @property
    def results(self):
        if not self.keep_result:
            raise ValueError("You must set `keep_result` to True to get results")
        return [self._results[idx].data for idx in range(self.last_index)]
    
    @property
    def is_stopped(self):
        return self.stop_index != -1 or self.stop_empty
    
    @property
    def multi_threaded(self):
        return self.max_workers > 0
    
    @property
    def node_text(self):
        des = super().node_text
        if self.buffer_type:     des += "Buffer type : {}\n".format(self.buffer_type)
        if self.batch_size != 1: des += "Batch size : {}\n".format(self.batch_size)
        if self.max_workers > 0: des += "Max workers : {}\n".format(self.max_workers)
        return des
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.multi_threaded:
            if self.batch_size in (0, 1) or self.current_index not in self._results:
                self.consume_next()
        
        with self.mutex_infos:
            if self._stop_index != -1 and self._current_index >= self._stop_index:
                raise StopIteration()

            idx = self._current_index
            self._current_index += 1

        logger.debug('[NEXT {}] Waiting for index {}'.format(self.name, idx))
        item = self._results[idx]
        if not self.keep_result: self._results.pop(idx)
        if item.stop: raise StopIteration()
        return item
    
    def __call__(self, item, * args, ** kwargs):
        """ Equivalent to `self.append` """
        return self.append(item, * args, ** kwargs)

    def empty(self):
        with self.mutex_infos:
            return self._last_index == self._next_index
        
    def get_stop_item(self, index):
        p = -1 if not isinstance(self.buffer, PriorityQueue) else float('inf')
        return Item(data = None, priority = p, stop = True)
        
    def get(self, raise_empty = False, ** kwargs):
        with self.mutex_get:
            try:
                if self.stop_empty:
                    kwargs['block'] = False
                    raise_empty     = False
                item = self.buffer.get(** kwargs)
            except Empty as e:
                logger.debug('[GET {}] Empty !'.format(self.name))
                if raise_empty: raise e
                return STOP_ITEM, -1
            
            if item.stop:
                logger.debug('[GET {}] Stop item !'.format(self.name))
                return item, -1
            
            logger.debug('[GET {}] {}'.format(self.name, item))
            
            with self.mutex_infos:
                if self._stop_index != -1 and self._next_index >= self._stop_index:
                    return STOP_ITEM, -1

                idx = self._next_index
                self._next_index += 1
        
        return item, idx
    
    def get_batch(self, ** kwargs):
        items, indexes  = Item(data = [], items = []), []
        with self.mutex_get:
            is_empty = False
            while len(indexes) < self.batch_size and not is_empty:
                try:
                    item, idx = self.get(
                        raise_empty = True, block = True if len(indexes) == 0 else False
                    )
                except Empty:
                    is_empty = True
                    continue
                
                if item.stop:
                    if len(indexes) == 0: return item, idx
                    self.buffer.put(item)
                    break
                
                items.data.append(item.data)
                items.items.append(item)
                indexes.append(idx)
        
        return items, indexes
    
    def consume_next(self):
        item, idx = self.get() if self.batch_size == 1 else self.get_batch()
        
        if item.stop:
            with self.mutex_infos:
                if self.__stop_empty:
                    self.__stop_empty = False
                    self.stop()
            return False

        self.consume(idx, item)
        return True
    
    def consume(self, idx, item):
        """ Consume an item and return the result """
        if not self.stateful:
            res = self.consumer(item.data, * item.args, ** item.kwargs)
        else:
            res, next_state = self.consumer(item.data, * item.args, * self.__state, ** item.kwargs)
            self.__state = next_state
        
        return self.set_result(idx, item, res)

    def set_result(self, idx, item, res):
        if isinstance(idx, (list, tuple)):
            return [
                self.set_result(idx_i, item_i, res_i)
                for idx_i, item_i, res_i in zip(idx, item.items, res)
            ]

        res = self.update_res_item(item, res)
        
        if self.max_workers != -2:
            self._results[idx] = res
            self.buffer.task_done()
        
        return res
    
    def update_res_item(self, item, res):
        return update_item(item, data = res, clone = False)
    
    def run_thread(self):
        run = True
        while run:
            run = self.consume_next()
    
    def run(self, * args, ** kwargs):
        if self.multi_threaded:
            self.workers = [
                Thread(target = self.run_thread, name = '{}_{}'.format(self.name, i))
                for i in range(self.max_workers)
            ]
            for w in self.workers: w.start()
        if not self.run_main_thread:
            super().run(* args, ** kwargs)
    
    def extend_and_wait(self, items, * args, stop = False, tqdm = None, ** kwargs):
        def append_and_wake_up(item, idx):
            result[idx] = item
            clock.release()
            if tqdm is not None: tqdm.update()
        
        def get_callback(idx):
            return lambda item: append_and_wake_up(item, idx)
        
        if tqdm is not None: tqdm = tqdm(total = len(items), unit = 'item')
        
        result = [None] * len(items)
        clock  = Semaphore(0)
        
        for i, item in enumerate(_create_generator(items)()):
            self.append(item, * args, callback = get_callback(i), ** kwargs)
        if not self.run_main_thread:
            for _ in range(len(items)): clock.acquire()
        
        if stop: self.stop()
        
        return result
        
    def append_and_wait(self, item, * args, stop = False, ** kwargs):
        def append_and_wake_up(item):
            result.append(item)
            clock.release()
        
        result = []
        clock  = Semaphore(0)
        
        self.append(item, * args, callback = append_and_wake_up, ** kwargs)
        if not self.run_main_thread: clock.acquire()
        if stop: self.stop()
        
        return result[0]
    
    def extend(self, items, * args, ** kwargs):
        return [self.append(item, * args, ** kwargs) for item in _create_generator(items)()]
    
    def append(self, item, * args, priority = -1, callback = None, ** kwargs):
        """ Add the item to the buffer (raise ValueError if `stop` has been called) """
        if self.is_stopped:
            raise StoppedException('Consumer stopped, you cannot add new items !')
        elif self.batch_size > 1 and len(args) + len(kwargs) != 0:
            raise ValueError('When using `batch_size` > 1, args / kwargs in `append` must be empty ! Found :\n  Args : {}\n  Kwargs : {}'.format(args, kwargs))
        
        with self.mutex_infos:
            idx = self._last_index
            self._last_index += 1
        
        if callback: callback = (self, callback)
        if not isinstance(item, Item):
            item = Item(
                data = item, args = args, kwargs = kwargs, index = idx, callback = callback,
                priority = priority if not self.is_max_priority else -priority
            )
        else:
            item = update_item(item, index = idx, clone = True)
        
        self._append_item(item)
        
        return item
    
    def _append_item(self, item):
        self.on_append(item)
        if self.max_workers == -2:
            self.on_item_produced(self.consume(item.index, item))
        else:
            self.buffer.put(item)
            if self.max_workers == -1:
                self.on_item_produced(self.__next__())

    
    def update_priority(self, item, new_priority, keep_best = True):
        if not isinstance(self.buffer, PriorityQueue): return

        if keep_best and item.priority <= new_priority: return
        with self.buffer.mutex:
            logger.debug('[PRIORITY UPDATE {}] Update priority to {} for item {}'.format(
                self.name, new_priority, item
            ))
            item.priority = new_priority
            heapq.heapify(self.buffer.queue)

    def stop(self, force = False, ** kwargs):
        """
            Set the `stop_index` to either `next_index` (if not force) else `current_index`
            The difference is in case of multi-threaded version, the `current` and `next` indexes can differ : the `current` represent the current iteration's index while `next` is the index for the next appened item. 
            If there is 3 items currently in the `buffer`, `next = current + 3`. Therefore, setting `force` to True will skip the 3 remaining items while `force = False` will first consume the 3 items then stop.
        """
        with self.mutex_infos:
            if self.finished: return
            logger.debug('[STATUS {}] Call to stop at index {}'.format(
                self.name, self._last_index
            ))
            if self._stop_index == -1 and not self.__stop_empty:
                self._stop_index   = self._last_index
                if not isinstance(self.buffer, LifoQueue) or self.empty():
                    for _ in range(max(1, self.max_workers)):
                        self.buffer.put(self.get_stop_item(self._last_index))
            
            if force: self._stop_index = self._current_index
            self._results.setdefault(self._stop_index, STOP_ITEM)
        
        if self.run_main_thread:
            self._finished = True
            self.on_stop()
    
    def stop_when_empty(self):
        with self.mutex_infos:
            self.__stop_empty = True
            if self.empty():
                self.buffer.put(self.get_stop_item(self._last_index))

    def join(self, * args, ** kwargs):
        """ Stop the thread then wait its end (that all items have been consumed) """
        self.stop_when_empty()
        for w in self.workers: w.join()
        super().join(* args, ** kwargs)
    
    def wait(self, * args, ** kwargs):
        """ Waits the Thread is finished (equivalent to `join`) """
        return super().join(* args, ** kwargs)

    def on_append(self, item):
        logger.debug('[APPEND {}] {}'.format(self.name, item))
        for l, infos in self.append_listeners:
            l(item) if infos.get('pass_item', False) else l(item.data)
    