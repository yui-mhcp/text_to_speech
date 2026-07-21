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
import traceback
import multiprocessing.queues

from typing import Any, Dict
from functools import partial
from threading import Thread, Lock, Event, Semaphore
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
        with self.mutex: return getattr(self, attr_name)
    
    def setter(self, value):
        with self.mutex: setattr(self, attr_name, value)
    
    attr_name = '_' + name
    return property(fget = getter, fset = setter)

class Stream(Thread):
    def __init__(self,
                 fn,
                 stream = None,
                 *,
                 
                 timeout    = None,
                 
                 callback   = None,
                 start_callback = None,
                 stop_callback  = None,
                 control_callback   = None,
                 
                 dict_as_kwargs = None,
                 
                 name   = None,
                 daemon = True,
                 max_workers    = 0,
                 prefetch_size  = 0,
                 stop_on_error  = True,
                 
                 ** kwargs
                ):
        """
            This class represents a (possibly multi-threaded) consumer
            
            The `Stream` object can be used in 2 different ways :
                1) Iterator-like : the `Stream` object can be used as an iterator, and yields the application of `fn` to each item of `stream`. In this case, the `stream` argument is required and acts as the producer (i.e., item generator).
                
                Example :
                ```python
                # only yields the result of fn(item)
                for item in Stream(fn, stream, ...):
                    ...
                # yields both input and output, where `out` is roughly equivalent to `fn(inp)`
                for inp, out in Stream(fn, stream, ...).items():
                    ...
                ```
                
                2) Function-like  : in this case, the items are manually provided by calling the `Stream` object, returning an `AsyncResult` object or the result directly. In this setup, the `stream` object must be omitted, and the `join` method should be called at the end.
                
                Example :
                ```python
                stream = Stream(fn)
                
                # when `max_workers == 0` : the output is the effective result
                out = stream(inp)
                # when `max_workers > 0` : the output is an `AsyncResult` object
                out = stream(inp).get()
                # out = await stream(inp).aget() # asyncio-compatible version

                # This is required to ensure that all items have been finalized, then call `on_stop`
                stream.join()
                ```

                Error handling : if `fn` raises, `get / aget` re-raise the exception in
                the caller (`max_workers > 0`), while the sequential mode
                (`max_workers == 0`) returns the exception object as the result
            
            The `max_workers` argument defines whether it `fn` executed in the main thread or not :
                - `max_workers = 0` : `fn` is called in the current thread
                - `max_workers = 1` : `fn` is called in the `Stream` thread
                - `max_workers > 1` : `fn` is called in multiple separate threads (`ThreadPool`)
        """
        Thread.__init__(self, name = name or get_fn_name(fn), daemon = daemon)
        
        if dict_as_kwargs is None:
            dict_as_kwargs = isinstance(stream, (queue.Queue, multiprocessing.queues.Queue))
        
        self.fn = fn
        self.stream = stream
        self.kwargs = kwargs
        self.timeout    = timeout
        self.dict_as_kwargs = dict_as_kwargs
        
        self.max_workers    = max_workers
        self.prefetch_size  = prefetch_size
        self.stop_on_error  = stop_on_error

        self._callbacks = {
            'start' : start_callback or [],
            'stop'  : stop_callback or [],
            'item'  : callback or [],
            'control'   : control_callback or []
        }
        for k, v in self._callbacks.items():
            if not isinstance(v, list): self._callbacks[k] = [v]

        self.mutex  = Lock() if max_workers else FakeLock()
        self.__started  = False
        self.__finished = False
        self._stopped   = False
        self._thread_started    = False
        
        self._pool  = None
        self._sema  = None

        self._results_buffer    = None
        self._stop_generator    = False
        self._generator_finished    = None
    
    stopped = _locked_property('stopped')
    stop_generator  = _locked_property('stop_generator')
    
    def _safe_fn(self, * args, ** kwargs):
        try:
            return self.fn(* args, ** kwargs)
        except Exception as e:
            if self.stop_on_error: self.stop()
            logger.error('An exception occured :\n{}'.format(traceback.format_exc()))
            return e
    
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
                self.stop()
                self.on_item_produced(DataWithResult(), CONTROL)
                return (args, CONTROL) if return_input else CONTROL
            
            elif isinstance(data, DataWithResult):
                if data.args is STOP:
                    self.stop()
                    self.on_item_produced(data, CONTROL)
                    return (args, CONTROL) if return_input else CONTROL
                elif isinstance(data.args, str) and data.args == IS_RUNNING:
                    self.on_item_produced(data, CONTROL)
                    return (args, CONTROL) if return_input else CONTROL
                elif isinstance(data.args, str) and data.args == KEEP_ALIVE:
                    if self.max_workers > 1: self._sema.release()
                    return (args, CONTROL) if return_input else CONTROL
                
            elif isinstance(data, str) and data == KEEP_ALIVE:
                if self.max_workers > 1: self._sema.release()
                return (args, CONTROL) if return_input else CONTROL
            elif isinstance(data, str) and data == IS_RUNNING:
                self.on_item_produced(DataWithResult(args = (IS_RUNNING, )), CONTROL)
                return (args, CONTROL) if return_input else CONTROL
            
            elif isinstance(data, dict) and self.dict_as_kwargs:
                data = DataWithResult(kwargs = data)
            else:
                data = DataWithResult(args = (data, ))
        
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Processing new item')
        
        _kwargs = {** self.kwargs, ** data.kwargs} if data.kwargs else self.kwargs
        if self.max_workers <= 1:
            # if `max_workers == 0`, this is called in the current thread
            # if `max_workers == 1`, this is called in the `Stream` thread
            result = self._safe_fn(* data.args, ** _kwargs)
            self.on_item_produced(data, result)
            return (args, result) if return_input else result
        else:
            self._pool.apply_async(
                self._safe_fn, data.args, _kwargs, callback = partial(self.on_item_produced, data)
            )
    
    def __iter__(self):
        """ Yields the result of `self.fn` applied on each element from `self.stream` """
        for _, res in self.items():
            yield res
    
    def __call__(self, * args, ** kwargs):
        # `_thread_started` is set synchronously in `start` (under `self.mutex` here),
        # while `__started` is only set by `on_start` (i.e., asynchronously in the
        # `Stream` thread when `max_workers > 0`) : checking the latter could start
        # the thread twice ("threads can only be started once")
        if not self._thread_started:
            with self.mutex:
                if not self._thread_started: self.start()

        if self.stopped:
            raise RuntimeError('The `Stream` is stopped !')

        if self.max_workers == 0:
            return self._apply_async(* args, ** kwargs)
        else:
            res = AsyncResult()
            self.stream.put(DataWithResult(args = args, kwargs = kwargs, result = res))
            return res

    def run(self):
        if self.max_workers == 0:
            raise RuntimeError('The `run` method should not be called in sequential mode')
        elif self.max_workers > 1:
            self._pool  = ThreadPool(self.max_workers).__enter__()
            self._sema  = Semaphore(self.max_workers)

        try:
            self.on_start()
            for item in create_iterable(self.stream, timeout = self.timeout):
                self._apply_async(item)
                if self.max_workers > 1: self._sema.acquire()
                if self.stopped: break
        except StopIteration:
            pass
        except Exception as e:
            # only call `terminate` on error, otherwise wait the end of all threads
            if self.max_workers > 1: self._pool.terminate()
            raise e from None
        finally:
            if self.max_workers > 1:
                self._pool.close()
                self._pool.join()

            if self._results_buffer is not None:
                self._results_buffer.put(STOP)
                self._generator_finished.wait()
            
            self.on_stop()

    def start(self):
        if self.max_workers == 0:
            self._thread_started    = True
            self.on_start()
        else:
            # the queue must exist *before* the thread starts : `__call__` may `put`
            # into it as soon as `start` returns
            if self.stream is None:
                assert self._results_buffer is None, 'You must provide the `stream` argument'
                self.stream = queue.Queue()

            self._thread_started    = True
            super().start()
        return self

    def stop(self):
        self.stopped = True
        
    def clear(self):
        if not hasattr(self.stream, 'get_nowait'): return

        try:
            while True:
                it = self.stream.get_nowait()
                # function-like mode : a caller may be blocked on `get()` for a
                # discarded item — wake it up with an explicit error
                if isinstance(it, DataWithResult) and isinstance(it.result, AsyncResult):
                    it.result.set_exception(
                        RuntimeError('The `Stream` was stopped before processing this item')
                    )
        except queue.Empty:
            pass
        
    def items(self):
        """
            Iterates over tuples `(input, output)`, where `output` is equivalent to `self(input)`
            
            If `self.max_workers == 0`:
                The results are sequentially created in the main thread, then yielded with its input
                
                It is roughly equivalent to :
                ```python
                self.on_start()
                for item in create_iterable(self.stream):
                    yield item, self._apply_async(item)
                self.on_stop()
                ```
            
            If `self.max_workers > 0`:
                The results are generated in separate thread(s), then added to a buffer, then yielded
                This implies that, if `max_workers > 1`, the results may be yielded in a different order than the calls order, depending on the result generation time
        """
        if self.max_workers == 0:
            try:
                self.on_start()
                for inp in create_iterable(self.stream, timeout = self.timeout):
                    inp, out = self._apply_async(inp, return_input = True)
                    # yield *before* checking `stopped` : on error (`stop_on_error`),
                    # the caller receives the exception instead of a silently
                    # truncated stream (mirrors the `max_workers > 0` behavior)
                    if out is not CONTROL:  yield inp, out
                    if self.stopped:        break
            
            except StopIteration:
                pass
            finally:
                self.on_stop()
        else:
            self._results_buffer = queue.Queue(self.prefetch_size)
            self._generator_finished    = Event()

            self._callbacks['item'].insert(0, self._results_buffer)
            
            self.start()
            
            try:
                stop_generator = False
                while not stop_generator:
                    data = self._results_buffer.get()
                    if data is STOP:
                        stop_generator = True
                    elif data.result is CONTROL:
                        continue
                    else:
                        yield data.args[0] if data.args else data.kwargs, data.result
            finally:
                self._generator_finished.set()
                if not stop_generator:
                    # early exit (`break` / exception in the consumer) : keep draining
                    # the buffer while the thread runs, otherwise workers blocked on a
                    # bounded `prefetch_size` buffer would deadlock the `join`
                    self.stop()
                    while super().is_alive():
                        try:
                            self._results_buffer.get(timeout = WARMUP_DELAY)
                        except queue.Empty:
                            pass
                self.join(force = True)
    
    def join(self, *, wakeup_timeout = 0.25, force = False, ** kwargs):
        if self.max_workers:
            if force:
                self.stop()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('[STATUS {}] join...'.format(self.name))

            if kwargs.get('timeout', 1) is None: kwargs.pop('timeout')
            try:
                while super().is_alive():
                    # the `STOP` token is only put when the queue looks empty, so that
                    # pending items are processed first (and priority queues never have
                    # to order `STOP` against real items). Re-attempted at every wakeup:
                    # a single upfront `put` could be missed when items are still pending
                    if hasattr(self.stream, 'put') and self.stream.qsize() == 0:
                        self.stream.put(STOP)

                    super().join(timeout = kwargs.get('timeout', wakeup_timeout))
                    if 'timeout' in kwargs: break
            except KeyboardInterrupt:
                # simply mark the stream as stopped : `run` will call `on_stop` itself
                # (calling it here too raised "called multiple times" in the thread)
                logger.info('Thread stopped while being joined !')
                self.stop()

            self.clear()
        elif not self.__finished:
            with self.mutex:
                if self.__started and not self.__finished: self.on_stop()
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('[STATUS {}] Joined !'.format(self.name))

    def on_start(self):
        """ Function called when starting the thread """
        if self.__started:
            raise RuntimeError('The `on_start` method is called multiple times')
        
        self.__started = True
        if logger.isEnabledFor(logging.DEBUG): logger.debug('[STATUS {}] Start'.format(self.name))
        _run_callbacks(self._callbacks['start'])

    def on_stop(self):
        """ Function called when stopping the thread """
        if not self.__started:
            raise RuntimeError('The stream was not started')
        elif self.__finished:
            # multiple termination paths (`run` finally, `join`, ...) may reach this
            # point : only the first one runs the callbacks
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('[STATUS {}] `on_stop` skipped (already stopped)'.format(self.name))
            return

        self.__finished = True
        if logger.isEnabledFor(logging.DEBUG): logger.debug('[STATUS {}] Stop'.format(self.name))
        _run_callbacks(self._callbacks['stop'])

    def on_item_produced(self, item, result):
        """ Function called when a new item is generated """
        if self.max_workers > 1: self._sema.release()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('[ITEM PRODUCED {}]'.format(self.name))

        if isinstance(item.result, AsyncResult):
            # function-like mode : `__call__` stored the caller's `AsyncResult` in
            # `item.result`. Resolve it now, before `_run_callbacks` overwrites the
            # field with the raw result. Exceptions caught by `_safe_fn` are re-raised
            # in the caller (`get / aget`) instead of being returned as values
            promise = item.result
            if isinstance(result, Exception):
                promise.set_exception(result)
            else:
                promise(result)

        _run_callbacks(self._callbacks['item'], item, result)
        if result is CONTROL:
            _run_callbacks(self._callbacks['control'], item, result)

_NO_RESULT  = object()

def _run_callbacks(callbacks, data = None, res = _NO_RESULT):
    """ Runs `callbacks`, either with no argument (`res` omitted, e.g., start / stop
        callbacks), or with `res` (item callbacks) — a legitimate `None` result is
        therefore forwarded as `callback(None)`, and not confused with "no argument"
    """
    assert data is None or isinstance(data, DataWithResult), str(data)

    if data is not None and res is not _NO_RESULT: data.result = res

    if not callbacks: return
    elif not isinstance(callbacks, list): callbacks = [callbacks]

    _remove = []
    for i, callback in enumerate(callbacks):
        if getattr(callback, 'stopped', False):
            _remove.append(i)
        elif hasattr(callback, 'put'):
            if data is not None or res is not _NO_RESULT:
                callback.put(data if data is not None else res)
        elif not callable(callback):
            logger.error('Unsupported callback : {}'.format(callback))
            _remove.append(i)
        elif res is not CONTROL:
            try:
                callback() if res is _NO_RESULT else callback(res)
            except Exception as e:
                if not isinstance(e, StopIteration):
                    logger.error('An exception occured while calling callback {} : {}'.format(
                        callback, e
                    ))
                _remove.append(i)
    
    for i in reversed(_remove): callbacks.pop(i)
    return callbacks