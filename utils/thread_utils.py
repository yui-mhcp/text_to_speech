import time
import logging

from typing import Any
from multiprocessing import cpu_count
from dataclasses import dataclass, field
from threading import Thread, RLock, Semaphore
from queue import Queue, LifoQueue, PriorityQueue

_queues = {
    'queue' : Queue,
    'stack' : LifoQueue,
    'lifo'  : LifoQueue,
    'priority'  : PriorityQueue,
    'max_priority' : PriorityQueue,
    'min_priority' : PriorityQueue
}

@dataclass(order = True)
class Data:
    priority : Any
    index    : int
    data     : Any = field(compare = False)

class ThreadedQueue(Thread):
    def __init__(self, target, mode = 'queue', keep_result = False, max_size = 0,
                 max_workers = cpu_count(), ** kwargs):
        Thread.__init__(self)
        
        self.max_workers = max_workers
        self.is_max_priority    = 'max' in mode

        self.__tasks    = _queues[mode](max_size)
        self.__runnings = 0
        self.__run_mutex    = RLock()
        
        self.keep_result = keep_result
        self.__results   = {} if keep_result else None
        self.__mutex     = RLock() if keep_result else None
        
        self.__target   = target
        self.__default_kwargs   = kwargs
        
        self.__stop = False
        self.__stop_empty   = False
        self.__semaphore    = Semaphore(max_workers) if self.multi_producer else None
        self.__semaphore_running = Semaphore(0)
    
    @property
    def queue(self):
        return self.__tasks
    
    @property
    def results(self):
        if not self.keep_result: return []
        with self.__mutex:
            return [self.__results[i] for i in range(len(self.__results))]
    
    @property
    def closed(self):
        return self.__stop or self.__stop_empty
    
    @property
    def multi_producer(self):
        return self.max_workers > 0
    
    def __len__(self):
        if not self.keep_result: return 0
        with self.__mutex:
            return len(self.__results)
    
    def __setitem__(self, idx, value):
        if not self.keep_result: return
        with self.__mutex:
            self.__results[idx] = value
    
    def __getitem__(self, idx):
        if not self.keep_result: return None
        with self.__mutex:
            return self.__results.get(idx, None)
    
    def __call__(self, data):
        logging.debug('start task {} with priority {}'.format(data.index, data.priority))
        result = None
        try:
            self.start_task()
            result = self.__target(
                * data.data.get('args', []),
                ** {** self.__default_kwargs, ** data.data.get('kwargs', {})}
            )
        except Exception as e:
            logging.error('Error occured : {}'.format(e))
        finally:
            self.finish_task(data.index, result)
        
        logging.debug('finished task {}'.format(data.index))
        return result
    
    def add_index(self):
        if not self.keep_result: return -1
        
        with self.__mutex:
            index = len(self)
            self[index] = None
        return index
    
    def start_task(self):
        self.__semaphore_running.acquire(blocking = False)
        with self.__run_mutex:
            self.__runnings += 1
        
    def finish_task(self, index, result):
        self[index] = result
        if self.multi_producer: self.__semaphore.release()
        
        self.__tasks.task_done()
        with self.__run_mutex:
            self.__runnings -= 1
            if self.__tasks.empty() and self.__runnings == 0:
                self.__semaphore_running.release()
        
    def run_task(self, data):
        logging.info('Get a new task : {}'.format(data))
        if data.data is None:
            self.__semaphore.release()
            return
        
        if not self.multi_producer: return self(data)

        Thread(target = self, args = (data, )).start()
    
    def run(self):
        logging.debug('Start queue')
        
        while not self.__stop and not (self.__stop_empty and self.__tasks.empty()):
            if self.multi_producer:
                logging.debug('Try to get semaphore...')
                self.__semaphore.acquire()
                
            self.run_task(self.pop())

        self.__semaphore_running.acquire(blocking = not self.__stop)
        logging.debug("Queue stopped")
    
    def stop(self, wait_empty = True):
        if wait_empty:
            self.__stop_empty = True
        else:
            self.__stop = True
        self.__tasks.put(Data(priority = 1, index = -1, data = None))
    
    def append(self, * args, priority = 0, ** kwargs):
        if self.closed: raise ValueError("You cannot add more data !")
        
        logging.debug('Adding new data on the queue !')
        
        if self.is_max_priority: priority = -priority
        
        self.__tasks.put(Data(
            priority = priority, index = self.add_index(), data = {'args' : args, 'kwargs' : kwargs}
        ))
    
    def pop(self, * args):
        data = self.__tasks.get(* args)
        
        logging.debug('Pop new data : {}'.format(data))
        
        return data

    def wait_result(self):
        if not self.closed:
            self.stop()
            self.join()
        return self.results

    def save(self, filename):
        from utils.file_utils import dump_pickle
        dump_pickle(filename, self.__tasks)
    
    def load(self, filename):
        from utils.file_utils import load_pickle
        tasks   = load_pickle(filename)
        while not tasks.empty():
            data = tasks.get()
            self.append(* data.data['args'], priority = data.priority, ** data.data['kwargs'])
        
class ThreadWithReturn(Thread):
    def __init__(self, * args, ** kwargs):
        Thread.__init__(self, * args, ** kwargs)
        self._result = None
        self._done = False
        
    def run(self):
        self._result = self._target(* self._args, ** self._kwargs)
        self._done = True
    
    def result(self):
        if not self._done:
            self.join()
        return self._result
    