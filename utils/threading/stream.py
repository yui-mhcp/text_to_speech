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

import queue
import logging
import inspect
import multiprocessing.queues

from typing import Any, Dict
from datetime import datetime
from functools import partial
from threading import Thread, RLock, Event
from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool

from .async_result import AsyncResult
from ..generic_utils import time_to_string, create_iterable, get_fn_name

logger = logging.getLogger(__name__)

STOP    = None
CONTROL = inspect._empty
IS_RUNNING  = '__is_running__'
KEEP_ALIVE  = '__keep_alive__'

WARMUP_DELAY = 0.25

@dataclass(order = True)
class DataWithResult:
    args    : Any   = field(compare = False, repr = False, default_factory = tuple)
    kwargs  : Dict  = field(compare = False, repr = False, default_factory = dict)
    
    result  : Any   = field(default = None, compare = False, repr = False)
    
    priority    : Any   = field(default = 0, compare = True)
    index       : int   = field(default = -1, compare = True)

class FakeLock:
    def __enter__(self):        pass
    def __exit__(self, * args): pass

def _locked_property(name):
    def getter(self):
        with self.mutex: return getattr(self, '_' + name)
    
    def setter(self, value):
        with self.mutex: setattr(self, '_' + name, value)
    
    return property(fget = getter, fset = setter)

class Stream(Thread):
    def __init__(self,
                 fn,
                 stream = None,
                 *,
                 
                 callback   = None,
                 start_callback = None,
                 stop_callback  = None,
                 control_callback   = None,
                 
                 dict_as_kwargs = None,
                 
                 prefetch_size  = 0,
                 max_workers    = 0,
                 daemon = True,
                 name   = None,
                 
                 ** kwargs
                ):
        Thread.__init__(self, name = name or get_fn_name(fn), daemon = daemon)
        
        if dict_as_kwargs is None:
            dict_as_kwargs = isinstance(stream, (queue.Queue, multiprocessing.queues.Queue))
        
        self.fn = fn
        self.stream = stream
        self.kwargs = kwargs
        self.dict_as_kwargs = dict_as_kwargs
        
        self.max_workers    = max_workers
        self.prefetch_size  = prefetch_size

        self._callbacks = {
            'start' : start_callback or [],
            'stop'  : stop_callback or [],
            'item'  : callback or [],
            'control'   : control_callback or []
        }
        for k, v in self._callbacks.items():
            if not isinstance(v, list): self._callbacks[k] = [v]

        self.mutex  = RLock() if max_workers else FakeLock()
        self.__started  = False
        self.__finished = False
        self._stopped   = False
        
        self._pool  = None
        self._empty = False
        self._results_buffer    = None
        self._generator_finished    = None
    
    empty   = _locked_property('empty')
    stopped = _locked_property('stopped')
    
    def _apply_async(self, * args, return_input = False, ** kwargs):
        """
            Call `self.fn` with the given `args` and `kwargs`
            
            Note that if `len(args) == 1 and len(kwargs) == 0`, `args[0]` is used instead. This enables passing special control data such as `STOP, KEEP_ALIVE` and `IS_RUNNING` or an already instanciated `DataWithResult`
            
            If `self.max_workers == 0`, this function calls `self.fn` in the main thread
            If `self.max_workers == 1`, this function calls `self.fn` in a separate thread
            If `self.max_workers > 1`, this function does not directly call `self.fn`, but calls `self.pool.apply_async` instead, such that `self.fn` will be called in other threads
        """
        if len(args) != 1 or kwargs:
            data = DataWithResult(args = args, kwargs = kwargs)
        else:
            args = data = args[0]
            if data is STOP:
                self.stopped = True
                self.on_item_produced(DataWithResult(), CONTROL)
                return (args, CONTROL) if return_input else CONTROL
            
            elif isinstance(data, DataWithResult):
                if data.args is STOP:
                    self.stopped = True
                    self.on_item_produced(data, CONTROL)
                    return (args, CONTROL) if return_input else CONTROL
                elif isinstance(data.args, str) and data.args == IS_RUNNING:
                    self.on_item_produced(data, CONTROL)
                    return (args, CONTROL) if return_input else CONTROL
                
            elif isinstance(data, str) and data == KEEP_ALIVE:
                return (args, CONTROL) if return_input else CONTROL
            elif isinstance(data, str) and data == IS_RUNNING:
                self.on_item_produced(DataWithResult(args = (IS_RUNNING, )), CONTROL)
                return (args, CONTROL) if return_input else CONTROL
            
            elif isinstance(data, dict) and self.dict_as_kwargs:
                data = DataWithResult(kwargs = data)
            else:
                data = DataWithResult(args = (data, ))
        
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('[{}] Processing new item'.format(datetime.now()))
        
        _kwargs = {** self.kwargs, ** data.kwargs} if data.kwargs else self.kwargs
        if self.max_workers == 0:
            result = self.fn(* data.args, ** _kwargs)
            self.on_item_produced(data, result)
            return (args, result) if return_input else result
        elif self._pool is not None:
            self._pool.apply_async(
                self.fn, data.args, _kwargs, callback = partial(self.on_item_produced, data)
            )
        else:
            result = self.fn(* data.args, ** _kwargs)
            self.on_item_produced(data, result)
    
    def __iter__(self):
        for _, res in self.items():
            yield res
    
    def __call__(self, * args, ** kwargs):
        if not self.__started: self.start()
        
        if self.max_workers == 0:
            return self._apply_async(* args, ** kwargs)
        else:
            res = AsyncResult()
            self.stream.put(DataWithResult(args = args, kwargs = kwargs, result = res))
            return res
    
    def run(self):
        if self.max_workers > 1: self._pool = ThreadPool(self.max_workers).__enter__()
        
        if self.stream is None:
            assert self._results_buffer is None, 'You must provide the `stream` argument'
            self.stream = queue.Queue()

        self.on_start()
        
        try:
            for item in create_iterable(self.stream):
                self._apply_async(item)
                if self.stopped: break
        except Exception as e:
            if self.max_workers > 1: self._pool.terminate()
            if not isinstance(e, StopIteration): raise e
        finally:
            if self.max_workers > 1:
                self._pool.close()
                self._pool.join()

            if self._results_buffer is not None:
                self._empty = True
                self._generator_finished.wait()
            
            self.on_stop()

    def start(self):
        if self.max_workers == 0:
            self.on_start()
        else:
            super().start()
        return self

    def stop(self):
        return self.join(force = True)
        
    def clear(self):
        while True:
            try:
                self.stream.get_nowait()
            except queue.Empty:
                break
        
    def items(self):
        """
            Iterates over tuples `(input, output)`, where `output` is equivalent to `self(input)`
            
            If `self.max_workers == 0`, this function sequentially calls items and yields the result
            If `self.max_workers > 0`, the stream-thread is started (`self.start()`), and all results are added in a queue, then yielded by this function
            This means that results are prefetched in a separate thread
        """
        if self.max_workers == 0:
            try:
                self.start()
                for inp in create_iterable(self.stream):
                    inp, out = self._apply_async(inp, return_input = True)
                    if self.stopped:        break
                    elif out is CONTROL:    continue
                    else:                   yield inp, out
            
            except StopIteration:
                pass
            finally:
                self.join()
            
        else:
            self._generator_finished    = Event()
            self._results_buffer = queue.Queue(self.prefetch_size)
            self._callbacks['item'].insert(0, self._results_buffer)
            
            self.start()
            
            try:
                while self._results_buffer.qsize() or not self.empty:
                    try:
                        data = self._results_buffer.get(timeout = WARMUP_DELAY)
                    except queue.Empty:
                        continue

                    if data.result is CONTROL:
                        continue
                    else:
                        yield data.args[0] if data.args else data.kwargs, data.result
            
            except StopIteration:
                pass
            finally:
                self._generator_finished.set()
                self.join()
    
    def join(self, *, wakeup_timeout = 0.25, force = False, ** kwargs):
        if self.max_workers:
            if force:
                self.stopped = True

            if hasattr(self.stream, 'put'):
                self.stream.put(STOP)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('[STATUS {}] join...'.format(self.name))
            
            if kwargs.get('timeout', 1) is None: kwargs.pop('timeout')
            try:
                while super().is_alive():
                    super().join(timeout = kwargs.get('timeout', wakeup_timeout))
                    if 'timeout' in kwargs: break
            except KeyboardInterrupt:
                logger.info('Thread stopped while being joined !')
                self.on_stop()
        elif not self.__finished:
            self.on_stop()
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('[STATUS {}] Joined !'.format(self.name))

    def on_start(self):
        """ Function called when starting the thread """
        self.__started = True
        if logger.isEnabledFor(logging.DEBUG): logger.debug('[STATUS {}] Start'.format(self.name))
        _run_callbacks(self._callbacks['start'])

    def on_stop(self):
        """ Function called when stopping the thread """
        self._stopped   = True
        self.__finished = True
        if self._pool is not None: self._pool.terminate()
        if logger.isEnabledFor(logging.DEBUG): logger.debug('[STATUS {}] Stop'.format(self.name))
        _run_callbacks(self._callbacks['stop'])

    def on_item_produced(self, item, result):
        """ Function called when a new item is generated """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('[ITEM PRODUCED {}]'.format(self.name))

        _run_callbacks(self._callbacks['item'], item, result)
        if result is CONTROL:
            _run_callbacks(self._callbacks['control'], item, result)

def _run_callbacks(callbacks, data = None, res = None):
    assert data is None or isinstance(data, DataWithResult), str(data)
    
    if data is not None: data.result = res

    if not callbacks: return
    elif not isinstance(callbacks, list): callbacks = [callbacks]

    _remove = []
    for i, callback in enumerate(callbacks):
        try:
            if getattr(callback, 'stopped', False): _remove.append(i)
            elif callable(callback):
                if res is not CONTROL: callback(res) if res is not None else callback()
            elif hasattr(callback, 'put'):  callback.put(data if data is not None else res)
            else:   raise ValueError('Unsupported callback : {}'.format(callback))
        except Exception as e:
            if not isinstance(e, StopIteration):
                logger.error('An exception occured while calling callback {} : {}'.format(
                    callback, e
                ))
            _remove.append(i)
    
    for i in reversed(_remove): callbacks.pop(i)
    return callbacks

