import logging

from typing import Any
from dataclasses import dataclass, field
from threading import Thread, RLock, Semaphore
from queue import Empty, Queue, LifoQueue, PriorityQueue

from utils.thread_utils.producer import _get_thread_name, Producer, Result, StoppedException
from utils.thread_utils.threaded_dict import ThreadedDict

_queues = {
    'queue' : Queue,
    'fifo'  : Queue,
    'stack' : LifoQueue,
    'lifo'  : LifoQueue,
    'priority'  : PriorityQueue,
    'max_priority' : PriorityQueue,
    'min_priority' : PriorityQueue
}

@dataclass(order = True)
class Item:
    priority    : Any
    index       : int
    item        : Any   = field(compare = False)
    args        : Any   = field(compare = False)
    kwargs      : dict  = field(compare = False)

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
    STOP_ITEM = Item(priority = -1, index = -1, item = None, args = (), kwargs = {})
    
    def __init__(self,
                 consumer,
                 * args,
                 buffer     = None,
                 
                 stateful   = False,
                 init_state = None,
                 
                 batch_size = 1,
                 
                 max_workers    = 0,
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
            if batch_size > 1:
                raise ValueError('`batch_size = {} and stateful = True` are incompatible !\nWhen using a`stateful  consumer, the `batch_size` must be 1'.format(batch_size))
            elif max_workers > 1:
                raise ValueError('`max_workers = {} and stateful = True` are incompatible !\nWhen using a`stateful  consumer, the `max_workers` must be <= 1'.format(max_workers))
        
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
        
        self.batch_size = max(batch_size, 1)
        
        self.keep_result    = keep_result
        self.__results      = ThreadedDict()
        
        self.max_workers    = max_workers
        
        self.__current_index    = 0
        self.__next_index   = 0
        self.__last_index   = 0
        self.__stop_index   = -1
        
        self.__stop_empty   = False
        
        self.mutex_get  = RLock()
        self.workers    = []
    
    @property
    def is_max_priority(self):
        return isinstance(self.buffer_type, str) and 'max' in self.buffer_type
    
    @property
    def listener(self):
        return self
    
    @property
    def current_index(self):
        with self.mutex_infos: return self.__current_index
    
    @property
    def next_index(self):
        with self.mutex_infos: return self.__next_index
    
    @property
    def last_index(self):
        with self.mutex_infos: return self.__last_index

    @property
    def stop_index(self):
        with self.mutex_infos: return self.__stop_index
    
    @property
    def stop_empty(self):
        with self.mutex_infos: return self.__stop_empty

    @property
    def results(self):
        if not self.keep_result:
            raise ValueError("You must set `keep_result` to True to get results")
        return [self.__results[idx].result for idx in range(self.last_index)]
    
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
            if self.batch_size in (0, 1) or self.current_index not in self.__results:
                self.consume_next()
        with self.mutex_infos:
            if self.__stop_index != -1 and self.__current_index >= self.__stop_index:
                raise StopIteration()

            idx = self.__current_index
            self.__current_index += 1

        logging.debug('[NEXT {}] Waiting for index {}'.format(self.name, idx))
        item = self.__results[idx]
        if not self.keep_result: self.__results.pop(idx)
        if isinstance(item, Item): raise StopIteration()
        return item
    
    def __call__(self, item, * args, ** kwargs):
        """ Equivalent to `self.append` """
        self.append(item, * args, ** kwargs)

    def empty(self):
        with self.mutex_infos:
            return self.__last_index == self.__next_index
        
    def get_stop_item(self, index):
        p = -1 if not isinstance(self.buffer, PriorityQueue) else float('inf')
        return Item(item = None, args = (), kwargs = {}, index = -1, priority = p)
        
    def get(self, raise_empty = False, ** kwargs):
        with self.mutex_get:
            try:
                if self.stop_empty:
                    kwargs['block'] = False
                    raise_empty     = False
                item = self.buffer.get(** kwargs)
            except Empty as e:
                logging.debug('[GET {}] Empty !'.format(self.name))
                if raise_empty: raise e
                return Consumer.STOP_ITEM, -1
            
            if item.index == -1:
                logging.debug('[GET {}] Stop item !'.format(self.name))
                return item, -1
            
            logging.debug('[GET {}] item with index {} and priority {}'.format(
                self.name, item.index, item.priority
            ))
            
            with self.mutex_infos:
                if self.__stop_index != -1 and self.__next_index >= self.__stop_index:
                    return Consumer.STOP_ITEM, -1

                idx = self.__next_index
                self.__next_index += 1
        
        return item, idx
    
    def get_batch(self, ** kwargs):
        items, indexes  = Item(item = [], priority = [], index = [], args = (), kwargs = {}), []
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
                
                if idx == -1:
                    if len(indexes) == 0: return item, idx
                    self.buffer.put(item)
                    break
                
                for attr in ['item', 'priority', 'index']:
                    getattr(items, attr).append(getattr(item, attr))
                indexes.append(idx)
        return items, indexes
    
    def consume_next(self):
        item, idx = self.get() if self.batch_size == 1 else self.get_batch()
        
        if idx == -1:
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
            res = self.consumer(item.item, * item.args, ** item.kwargs)
        else:
            res, next_state = self.consumer(item.item, * item.args, * self.__state, ** item.kwargs)
            self.__state = next_state

        if self.max_workers != -2:
            if isinstance(idx, (list, tuple)):
                for i, (idx_i, res_i) in enumerate(zip(idx, res)):
                    self.__results[idx_i] = Result(
                        result = res_i, index = item.index[i], priority = item.priority[i]
                    )
                    self.buffer.task_done()
            else:
                self.__results[idx] = Result(
                    result = res, index = item.index, priority = item.priority
                )
                self.buffer.task_done()
        
        return res
    
    def run_threads(self):
        run = True
        while run:
            run = self.consume_next()
    
    def run(self, * args, ** kwargs):
        if self.multi_threaded:
            self.workers = [
                Thread(target = self.run_threads, name = '{}_{}'.format(self.name, i))
                for i in range(self.max_workers)
            ]
            for w in self.workers: w.start()
        if not self.run_main_thread:
            super().run(* args, ** kwargs)
    
    def append(self, item, * args, priority = -1, ** kwargs):
        """ Add the item to the buffer (raise ValueError if `stop` has been called) """
        if self.is_stopped:
            raise StoppedException('Consumer stopped, you cannot add new items !')
        elif self.batch_size > 1 and len(args) + len(kwargs) != 0:
            raise ValueError('When using `batch_size` > 1, args / kwargs in `append` must be empty ! Found :\n  Args : {}\n  Kwargs : {}'.format(args, kwargs))
        
        with self.mutex_infos:
            item = Item(
                priority = priority if not self.is_max_priority else -priority,
                index = self.__last_index, item = item, args = args, kwargs = kwargs
            )
            self.__last_index += 1
        logging.debug('[APPEND {}] index {} and priority {}'.format(
            self.name, item.index, item.priority
        ))
        if self.max_workers == -2:
            self.on_item_produced(self.consume(item.index, item))
        else:
            self.buffer.put(item)
            if self.max_workers == -1:
                self.on_item_produced(self.__next__())

    
    def stop(self, force = False, ** kwargs):
        """
            Set the `stop_index` to either `next_index` (if not force) else `current_index`
            The difference is in case of multi-threaded version, the `current` and `next` indexes can differ : the `current` represent the current iteration's index while `next` is the index for the next appened item. 
            If there is 3 items currently in the `buffer`, `next = current + 3`. Therefore, setting `force` to True will skip the 3 remaining items while `force = False` will first consume the 3 items then stop.
        """
        with self.mutex_infos:
            if self.finished: return
            logging.debug('[STATUS {}] Call to stop at index {}'.format(
                self.name, self.__last_index
            ))
            if self.__stop_index == -1 and not self.__stop_empty:
                self.__stop_index   = self.__last_index
                if not isinstance(self.buffer, LifoQueue) or self.empty():
                    for _ in range(max(1, self.max_workers)):
                        self.buffer.put(self.get_stop_item(self.__last_index))
            
            if force: self.__stop_index = self.__current_index
            self.__results.setdefault(self.__stop_index, Consumer.STOP_ITEM)
        
        if self.max_workers == -1:
            self._finished = True
            self.on_stop()
    
    def stop_when_empty(self):
        with self.mutex_infos:
            self.__stop_empty = True
            if self.empty():
                self.buffer.put(self.get_stop_item(self.__last_index))

    def join(self, * args, ** kwargs):
        """ Stop the thread then wait its end (that all items have been consumed) """
        self.stop_when_empty()
        if self.multi_threaded:
            for w in self.workers: w.join()
        super().join(* args, ** kwargs)
    
    def wait(self, * args, ** kwargs):
        return super().join(* args, ** kwargs)