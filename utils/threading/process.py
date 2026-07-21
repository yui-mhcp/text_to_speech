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

from functools import wraps
from threading import Thread, RLock

from .async_result import AsyncResult
from .stream import STOP, KEEP_ALIVE, IS_RUNNING, DataWithResult, _locked_property, _run_callbacks

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
            else:   name = fn.__class__.__name__

        with _global_mutex:
            if name not in _processes or _processes[name].stopped:
                # `stream` in `kwargs` means the target `fn` already receives its own
                # stream (generator-like worker) : no `input_stream` is created then
                if add_stream and 'stream' not in kwargs and 'input_stream' not in kwargs:
                    kwargs['input_stream'] = 'queue'
                
                _processes[name] = super().__call__(fn, * args, name = name, ** kwargs)
        
            return _processes[name]
        
class Process(metaclass = MetaProcess):
    def __init__(self,
                 fn,
                 args   = (),
                 kwargs = None,

                 *,

                 callbacks  = None,
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

        # copy : `_run_callbacks` mutates the list in-place (removes failing callbacks)
        if callbacks is None:                   callbacks = []
        elif not isinstance(callbacks, list):   callbacks = [callbacks]
        else:                                   callbacks = list(callbacks)
        self.callbacks  = callbacks
        
        self.restart    = restart

        # `max_priority` is implemented parent-side (negation at `put` time) : custom
        # attributes on a `multiprocessing.Queue` would not survive pickling to the child
        self._negate_priority   = isinstance(input_stream, str) and input_stream.lower() == 'max_priority'
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
    
    def _apply_async(self, data, *, priority = 0, callback = None, loop = None):
        result = AsyncResult(callback = callback, loop = loop)
        with self.mutex:
            if self._stopped:
                raise RuntimeError('Cannot add new data to a stopped process')

            index = self._get_index(data)
            _is_control = (data is STOP) or (
                isinstance(data, str) and data in (IS_RUNNING, KEEP_ALIVE)
            )
            if (isinstance(data, str) and data == KEEP_ALIVE) or (self.skip_outputs and not _is_control):
                # `KEEP_ALIVE` is never acknowledged by the worker and, with
                # `skip_outputs`, regular outputs never come back either : the
                # `AsyncResult` is resolved right away rather than registered
                # (registering it would leak memory and hang any `get()`)
                self._index += 1
                result(None)
            elif (self.keep_results and index in self._results) and (not isinstance(data, dict) or not data.get('overwrite', False)):
                result(self._results[index])
                return result
            elif index in self._waiting_results and self.buffer_type == 'Queue':
                # NB : the deduplication is restricted to FIFO buffers on purpose — on a
                # priority queue, re-submitting the same index with a higher priority
                # must re-enqueue the item so it can jump ahead
                self._waiting_results[index].append(result)
                return result
            else:
                self._index += 1
                self._waiting_results.setdefault(index, []).append(result)
        
        if self.only_process_last: self.clear()

        if _is_control:
            # keep the same semantic as raw tokens (cf `PriorityQueue._build_item`) :
            # the liveness ping jumps ahead, `STOP` / `KEEP_ALIVE` drain pending items first
            priority = float('-inf') if data == IS_RUNNING else float('inf')
        elif self._negate_priority:
            priority = -priority

        if isinstance(data, dict):
            _args, _kwargs = (), data
        elif (data is STOP) or (isinstance(data, str) and data in (IS_RUNNING, KEEP_ALIVE)):
            _args, _kwargs = data, {}
        else:
            _args, _kwargs = (data, ), {}
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('[{}] Add new item to queue'.format(self.name))
        
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
                target = self.fn, args = self.args, kwargs = kwargs, name = self.name
            )
            self._process.start()
            if self._finalizer is None:         self._finalizer = self.start_finalizer()
            if self._results_handler is None:   self._results_handler = self.start_results_handler()
        return self
    
    def stop(self):
        with self.mutex:
            if self._stopped: return
            self._stopped = True
            # the `STOP` token is put while holding the mutex, so that no item can be
            # enqueued *after* it (`_apply_async` would register a result that the
            # worker will never produce)
            if self.input_stream is not None:
                self.input_stream.put(STOP)

        # called outside the mutex : `terminate` takes `_global_mutex` first, and the
        # lock ordering must stay `_global_mutex` -> `self.mutex` (cf `MetaProcess`)
        if self.input_stream is None:
            self.terminate()

    def clear(self):
        if self.input_stream is None: return

        try:
            while True:
                it = self.input_stream.get_nowait()
                if not isinstance(it, DataWithResult): continue    # control token
                with self.mutex:
                    for res in self._waiting_results.pop(it.index, []): res(None)
        except queue.Empty:
            pass

    def is_running(self, timeout = 5):
        """ Round-trip check : returns `False` (instead of hanging or raising) when the worker is stopped, dead, or unresponsive within `timeout` """
        if self.stopped or not self.is_alive():
            return False

        try:
            self._apply_async(IS_RUNNING).get(timeout = timeout)
            return True
        except Exception:
            return False
    
    def keep_alive(self):
        if self.input_stream is not None: self.input_stream.put(KEEP_ALIVE)
    
    def is_alive(self):
        with self.mutex:
            return self.process is not None and self.process.is_alive()

    def join(self, ** kwargs):
        if self._finalizer is not None: self._finalizer.join(** kwargs)

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

            # wake up any caller still blocked on a pending `get()` : the worker will
            # never produce these results
            if self._waiting_results:
                error = RuntimeError('The process `{}` has been terminated'.format(self.name))
                for waiting in self._waiting_results.values():
                    for res in waiting: res.set_exception(error)
                self._waiting_results   = {}

        logger.info('Process `{}` is closed (status {}) !'.format(
            self.name, self.exitcode
        ))

    

    @run_in_thread(daemon = True)
    def start_results_handler(self):
        # loops until the `STOP` token (always put by `terminate`) rather than checking
        # `self.stopped` : results already produced by the worker when `stop` is called
        # must still be dispatched to their waiting `AsyncResult`
        while True:
            data = self.output_stream.get()
            if data is STOP: return

            if isinstance(data, DataWithResult):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('[{}] New result received (index {}) : {}'.format(
                        self.name, data.index, data.result
                    ))

                with self.mutex:
                    for res in self._waiting_results.pop(data.index, []):
                        if isinstance(data.result, Exception):
                            res.set_exception(data.result)
                        else:
                            res(data.result)

                    # errors are not cached : a retry on the same index must re-run `fn`
                    if self.keep_results and not isinstance(data.result, Exception):
                        self._results[data.index] = data.result
                result = data.result
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('[{}] New result received : {}'.format(self.name, data))
                result = data
            
            _run_callbacks(self.callbacks, None, result)
    
    @run_in_thread(daemon = True)
    def start_finalizer(self):
        """
            Waits for the sub-process end, then finalizes (`terminate`) or restarts it

            `restart` only revives workers that exited *cleanly* (exitcode 0, e.g., an
            idle-timeout on their input stream) : a crashed worker (exitcode != 0) is
            always finalized, to avoid crash-loops. A fresh `Process` is transparently
            re-created by `MetaProcess` on the next request anyway
        """
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