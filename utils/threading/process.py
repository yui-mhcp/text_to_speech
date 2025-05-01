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
import logging
import multiprocessing.queues

from datetime import datetime
from functools import wraps
from threading import Thread, RLock

from .async_result import AsyncResult
from .stream import STOP, KEEP_ALIVE, IS_RUNNING, DataWithResult, _locked_property, _run_callbacks

RESULTS_HANDLER_WAKEUP_TIME = 1.

logger = logging.getLogger(__name__)

_processes  = {}
_global_mutex   = RLock()

_buffers    = {
    'queue' : multiprocessing.Queue,
    'fifo'  : multiprocessing.Queue,
    'priority'  : multiprocessing.PriorityQueue,
    'min_priority'  : multiprocessing.PriorityQueue,
    'max_priority'  : multiprocessing.PriorityQueue
}

def run_in_thread(fn = None, name = None, callback = None, ** thread_kwargs):
    def wrapper(fn):
        @wraps(fn)
        def inner(* args, ** kwargs):
            thread = Thread(
                target = fn, args = args, kwargs = kwargs, name = name or fn.__name__, ** thread_kwargs
            )
            thread.start()
            
            return thread
        return inner
    return wrapper if fn is None else wrapper(fn)

class MetaProcess(type):
    def __call__(self, fn, * args, add_stream = True, name = None, ** kwargs):
        if not name:
            if isinstance(fn, str):     name = fn
            elif hasattr(fn, 'name'):   name = fn.name
            elif hasattr(fn, '__name__'):   name = fn.__name__
            else:   fn = fn.__class__.__name__
        
        with _global_mutex:
            if name not in _processes or _processes[name].stopped:
                if add_stream and 'stream' not in kwargs:
                    kwargs['input_stream'] = 'queue'
                
                _processes[name] = super().__call__(fn, * args, name = name, ** kwargs)
        
            return _processes[name]
        
class Process(metaclass = MetaProcess):
    def __init__(self,
                 fn,
                 args   = (),
                 kwargs = {},
                 
                 *,
                 
                 callbacks  = [],
                 input_stream   = None,
                 skip_outputs   = False,
                 only_process_last  = False,
                 
                 restart    = False,
                 
                 result_key = None,
                 keep_results   = False,
                 
                 name   = None,
                 
                 ** kw
                ):
        self.fn = fn
        self.name   = name
        self.args   = args
        self.kwargs = kwargs or kw
        
        self.callbacks  = callbacks
        
        self.restart    = restart

        self.input_stream   = _get_buffer(input_stream) if input_stream is not None else None
        self.output_stream  = _get_buffer('queue')
        self.skip_outputs   = skip_outputs
        self.only_process_last  = only_process_last
        
        self.result_key = result_key
        self.keep_results   = keep_results
        
        self.mutex  = RLock()
        self._process   = None
        self._finalizer = None
        
        self._results   = {}
        self._waiting_results   = {}
        self._results_handler   = None
        
        self._index = 0
        self._stopped   = False
        self._exitcode  = None
    
    def _get_index(self, data):
        if (self.result_key is None) or (isinstance(data, str) and data in (KEEP_ALIVE, IS_RUNNING)):
            return self.index
        elif isinstance(self.result_key, str):
            return data[self.result_key] if isinstance(data, dict) else data
        elif isinstance(self.result_key, (list, tuple)):
            return tuple(data[k] for k in self.result_key) if isinstance(data, dict) else data
    
    def _apply_async(self, data, *, priority = 0, callback = None):
        result = AsyncResult(callback = callback)
        with self.mutex:
            if self._stopped:
                raise RuntimeError('Cannot add new data to a stopped process')
            
            index = self._get_index(data)
            if (self.keep_results and index in self._results) and (not isinstance(data, dict) or not data.get('overwrite', False)):
                result(self._results[index])
                return result
            elif index in self._waiting_results and self.buffer_type == 'Queue':
                self._waiting_results[index].append(result)
                return result
            else:
                self._index += 1
                self._waiting_results.setdefault(index, []).append(result)
        
        if self.only_process_last: self.clear()
        
        if isinstance(data, dict):
            _args, _kwargs = (), data
        elif (data is STOP) or (isinstance(data, str) and data in (IS_RUNNING, KEEP_ALIVE)):
            _args, _kwargs = data, {}
        else:
            _args, _kwargs = (data, ), {}
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('[{}] Add new item to queue'.format(datetime.now()))
        
        self.input_stream.put(DataWithResult(
            args = _args, kwargs = _kwargs, index = index, priority = priority
        ))
        return result

    def _map_async(self, items, *, priority = 0, callback = None):
        with self.mutex:
            if isinstance(priority, (int, float)): priority = [priority] * len(items)
            return [
                self._apply_async(it, priority = p, callback = callback)
                for it, p in zip(items, priority)
            ]
    
    __call__    = _apply_async
    append      = _apply_async
    send    = _apply_async
    put     = _apply_async
    
    extend  = _map_async
    
    
    index   = _locked_property('index')
    stopped = _locked_property('stopped')
    exitcode    = _locked_property('exitcode')
    
    @property
    def process(self):
        return self._process
    
    @property
    def buffer_type(self):
        return self.input_stream.__class__.__name__
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, * args):
        self.terminate()
    
    def __repr__(self):
        des = '<Process name={}'.format(self.name)
        if self.exitcode is not None:
            des += ' exitcode={}'.format(self.exitcode)
        elif self.is_alive():
            des += ' running'
        
        return des + '>'
    
    def __str__(self):
        return self.name or repr(self)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if hasattr(other, 'name'): other = other.name
        return self.name == other
    
    def start(self):
        with self.mutex:
            if self.is_alive(): return self
            elif self.stopped:  raise RuntimeError('The process has been stopped')

            kwargs = self.kwargs.copy()
            if self.input_stream is not None:
                kwargs['stream'] = self.input_stream

            if not self.skip_outputs:
                kwargs['callback'] = self.output_stream
            else:
                kwargs['control_callback'] = self.output_stream

            self._process   = multiprocessing.Process(
                target = self.fn, kwargs = kwargs, name = self.name
            )
            self._process.start()
            if self._finalizer is None:         self._finalizer = self.start_finalizer()
            if self._results_handler is None:   self._results_handler = self.start_results_handler()
        return self
    
    def stop(self):
        if not self._stopped:
            self.stopped = True
            if self.input_stream is not None:
                self.input_stream.put(STOP)
            else:
                self.terminate()
    
    def clear(self):
        try:
            while True:
                it = self.input_stream.get_nowait()
                for res in self._waiting_results.pop(it.index): res(None)
        except queue.Empty:
            pass

    def is_running(self):
        self._apply_async(IS_RUNNING).get()
    
    def keep_alive(self):
        self.input_stream.put(KEEP_ALIVE)
    
    def is_alive(self):
        with self.mutex:
            return self.process is not None and self.process.is_alive()

    def join(self, ** kwargs):
        self._finalizer.join(** kwargs)

    def terminate(self):
        with _global_mutex:
            if _processes.get(self.name, None) is self:
                _processes.pop(self.name)

        with self.mutex:
            if self.process is None: return
            self._stopped   = True
            
            self.process.terminate()
            self.process.join()
            self.output_stream.put(STOP)

            self._exitcode = self.process.exitcode
        
        logger.info('Process `{}` is closed (status {}) !'.format(
            self.name, self.exitcode
        ))

    

    @run_in_thread(daemon = True)
    def start_results_handler(self):
        while not self.stopped:
            data = self.output_stream.get()
            if data is STOP: return

            if isinstance(data, DataWithResult):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('[{}] New result received (index {}) : {}'.format(
                        datetime.now(), data.index, data.result
                    ))

                with self.mutex:
                    for res in self._waiting_results.pop(data.index, []):
                        res(data.result)

                    if self.keep_results: self._results[data.index] = data.result
                result = data.result
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('[{}] New result received : {}'.format(datetime.now(), data))
                result = data
            
            _run_callbacks(self.callbacks, None, result)
    
    @run_in_thread(daemon = True)
    def start_finalizer(self):
        finalize, run = False, 0
        while not finalize:
            self.process.join()
            if self.stopped or self.process.exitcode != 0:
                finalize = True
            elif (self.restart) and (self.restart is True or run < self.restart):
                run += 1
                self.start()
            else:
                finalize = True
        
        self.terminate()

def _get_buffer(buffer = 'fifo', maxsize = 0):
    if buffer is None: buffer = 'queue'

    if isinstance(buffer, str):
        buffer = buffer.lower()
        if buffer not in _buffers:
            raise ValueError('`buffer` is an unknown queue type :\n  Accepted : {}\n  Got : {}\n'.format(tuple(_buffers.keys()), buffer))
        
        buffer = _buffers[buffer](maxsize)
    
    elif not isinstance(buffer, (queue.Queue, multiprocessing.queues.Queue)):
        raise ValueError('`buffer` must be a Queue instance or subclass')

    return buffer
