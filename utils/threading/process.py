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

import os
import time
import queue
import logging
import multiprocessing

from typing import Any, Dict
from threading import RLock
from dataclasses import dataclass

from .producer import _item_to_str
from .async_result import AsyncResult
from . import run_in_thread, locked_property, get_buffer

KEEP_ALIVE  = '__keep_alive__'
RESULTS_HANDLER_WAKEUP_TIME = 1.

logger = logging.getLogger(__name__)

_processes  = {}
_global_mutex   = RLock()

@dataclass
class DataWithResults:
    data    : Dict
    index   : Any
    result  : Any   = None

class MetaProcess(type):
    def __call__(self, fn, * args, add_stream = False, add_callback = False, ** kwargs):
        name = kwargs.get('name', getattr(fn, '__name__', None))
        
        with _global_mutex:
            if name not in _processes or _processes[name].stopped:
                if add_stream:      kwargs.setdefault('input_stream', 'queue')
                if add_callback:    kwargs.setdefault('output_stream', 'queue')
                if 'stream' in kwargs: kwargs['input_stream'] = kwargs.pop('stream')
                
                _processes[name] = super().__call__(fn, * args, ** kwargs)
        
        return _processes[name]
        
class Process(metaclass = MetaProcess):
    def __init__(self,
                 fn,
                 args   = (),
                 kwargs = {},
                 *,
                 
                 input_stream   = None,
                 output_stream  = None,
                 
                 restart    = False,
                 
                 result_key = None,
                 keep_results   = False,
                 
                 name   = None,
                 
                 ** kw
                ):
        if output_stream is None: output_stream = 'queue'
        
        self.fn = fn
        self.name   = name
        self.args   = args
        self.kwargs = kwargs or kw
        
        self.input_stream   = get_buffer(input_stream, use_multiprocessing = True) if input_stream is not None else None
        self.output_stream  = get_buffer(output_stream, use_multiprocessing = True)
        
        self.result_key = result_key
        self.keep_results   = keep_results

        self.restart    = restart
        
        self.mutex  = RLock()
        self.process    = None
        self.finalizer  = None
        self.results_handler    = None
        self._waiting_results   = {}
        self._results   = {}

        self.pipes  = {'input' : set(), 'output' : set()}
        self.synchronizer   = None  # thread keeping `self` alive until `len(self.pipes['input']) > 0`
        
        self._index = 0
        self._stopped   = False
        self._exitcode  = None
    
    def _get_index(self, data):
        if self.result_key is None:
            return self.index
        elif isinstance(self.result_key, str):
            return data[self.result_key] if isinstance(data, dict) else data
        elif isinstance(self.result_key, (list, tuple)):
            return tuple(data[k] for k in self.result_key) if isinstance(data, dict) else data
    
    def _apply_async(self, data, callback = None, ** kwargs):
        result = AsyncResult(callback = callback)
        with self.mutex:
            index = self._get_index(data)
            if self.keep_results and not kwargs.get('overwrite', False) and index in self._results:
                result(self._results[index])
                return result
            elif index in self._waiting_results and self.buffer_type == 'Queue':
                self._waiting_results[index].append(result)
                return result
            else:
                self._index += 1
                self._waiting_results.setdefault(index, []).append(result)
        
        if self.stopped:
            raise RuntimeError('Cannot add new data to a stopped process')
        
        self.input_stream.put(DataWithResults(data = data, index = index), ** kwargs)
        return result

    def _map_async(self, items, *, callback = None, ** kwargs):
        with self.mutex:
            return [self._apply_async(it, callback = callback, ** kwargs) for it in items]
    
    __call__    = _apply_async
    append      = _apply_async
    send    = _apply_async
    put     = _apply_async
    
    extend  = _map_async
    
    
    index   = locked_property('index')
    stopped = locked_property('stopped')
    
    @property
    def target(self):
        return self.fn
    
    @property
    def buffer_type(self):
        return self.input_stream.__class__.__name__
    
    @property
    def exitcode(self):
        return self._exitcode
    
    @property
    def input_pipes(self):
        return self.pipes['input']
    
    @property
    def output_pipes(self):
        return self.pipes['output']
    
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
        
        if self.input_pipes:
            des += ' in_pipes={}'.format(self.input_pipes)
        
        if self.output_pipes:
            des += ' out_pipes={}'.format(self.output_pipes)
        
        return des + '>'
    
    def __str__(self):
        return self.name or repr(self)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if hasattr(other, 'name'): other = other.name
        return self.name == other
    
    def start(self):
        if self.is_alive(): return self
        elif self.stopped: raise RuntimeError('The process has been stopped')

        kwargs = self.kwargs.copy()
        if self.input_stream is not None:
            assert 'stream' not in kwargs
            kwargs['stream'] = self.input_stream

        if self.output_stream is not None:
            assert 'callback' not in kwargs
            kwargs['callback'] = self.output_stream

        self.process = multiprocessing.Process(
            target = self.target, args = self.args, kwargs = kwargs, name = self.name
        )
        self.process.start()
        if self.finalizer is None:          self.finalizer = self.start_finalizer()
        if self.results_handler is None:    self.results_handler = self.start_results_handler()
        return self
    
    def stop(self):
        if not self._stopped: self.stopped = True
    
    def join(self, ** kwargs):
        self.finalizer.join(** kwargs)

    def terminate(self):
        with _global_mutex:
            if _processes.get(self.name, None) is self:
                _processes.pop(self.name, None)

        with self.mutex:
            if self.process is None: return
            process = self.process
            self.process    = None
            self._stopped   = True
            
        process.terminate()
        process.join()

        self._exitcode = process.exitcode
        
        logger.info('Process `{}` is closed (status {}) !'.format(
            self.name, self.exitcode
        ))

    
    def is_alive(self):
        with self.mutex:
            return self.process is not None and self.process.is_alive()

    def add_observer(self, process):
        if isinstance(process, str): process = get_process(process)
        self.output_pipes.add(process)
    
    def remove_observer(self, process):
        if isinstance(process, str): process = get_process(process)
        self.output_pipes.remove(process)
    
    def add_input(self, process):
        if isinstance(process, str): process = get_process(process)
        self.input_pipes.add(process)
        
        with self.mutex:
            if self.synchronizer is None:
                self.synchronizer = self.start_keep_alive()

    def remove_input(self, process):
        if isinstance(process, str): process = get_process(process)
        self.input_pipes.remove(process)

    @run_in_thread(daemon = True)
    def start_keep_alive(self):
        sleep_time  = self.kwargs.get('timeout', 10) / 2
        
        while self.input_pipes:
            time.sleep(sleep_time)
            if self.stopped: break
            
            for inp in self.input_pipes.copy():
                if get_process(inp.name) is not None:
                    self.put(KEEP_ALIVE)
                    break
                else:
                    self.remove_input(inp)

        logger.info('Synchronizer finished !')

    @run_in_thread(daemon = True)
    def start_results_handler(self):
        while not self.stopped:
            try:
                data = self.output_stream.get(timeout = RESULTS_HANDLER_WAKEUP_TIME)
            except queue.Empty:
                continue

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('New result received (index {}) : {}'.format(
                    data.index, _item_to_str(data.result)
                ))
            
            with self.mutex:
                for res in self._waiting_results.pop(data.index, []):
                    res(data.result)
                
                if self.keep_results: self._results[data.index] = data.result
            
            data = data.result
            for obs in self.output_pipes.copy():
                if obs.is_alive():
                    logger.debug('Transferring data from {} to {}'.format(self.name, out.name))
                    obs.put(data)
                elif get_process(obs.name) is None:
                    logger.info('The output of the pipe ({}) does not exist anymore !'.format(out))
                    self.remove_observer(obs)
        
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

def get_process(name):
    with _global_mutex: return _processes.get(name, None)

def terminate_process(name):
    with _global_mutex:
        process = _processes.pop(name, None)
    if process is not None: process.terminate()
    return process
