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
import subprocess
import multiprocessing

from threading import Thread
from typing import Tuple, Dict
from dataclasses import dataclass
from multiprocessing import Queue

from . import run_in_thread
from ..stream_utils import KEEP_ALIVE

logger = logging.getLogger(__name__)

_processes  = {}

class MetaProcess(type):
    def __call__(self, fn, * args, add_stream = False, add_callback = False, ** kwargs):
        name = kwargs.get('name', fn.__name__)
        if name not in _processes:
            if add_stream:      kwargs['input_stream'] = Queue()
            if add_callback:    kwargs['output_stream'] = Queue()
                
            _processes[name] = super().__call__(fn, * args, ** kwargs)
        
        return _processes[name]
        
class Process(metaclass = MetaProcess):
    def __init__(self,
                 fn,
                 args   = (),
                 kwargs = {},
                 *,
                 
                 name   = None,
                 
                 input_stream   = None,
                 output_stream  = None,
                 
                 callback   = None,
                 restart    = False,
                 
                 ** kw
                ):
        self.fn = fn
        self.name   = name or fn.__name__
        self.args   = args
        self.kwargs = kwargs or kw
        
        self.input_stream   = input_stream
        self.output_stream  = output_stream
        
        self.callback   = callback
        self.restart    = restart
        
        self.process    = None
        self.finalizer  = None
        
        self.pipes  = {'input' : set(), 'output' : set()}
        self.forwarder  = None  # thread forwarding `self.output_stream` to `pipes['output']`
        self.synchronizer   = None  # thread keeping `self` alive until `len(self.pipes['input']) > 0`
        
        _processes[self.name] = self
    
    @property
    def target(self):
        return self.fn
    
    @property
    def links(self):
        return self.pipes['input']
    
    @property
    def observers(self):
        return self.pipes['output']
    
    def __repr__(self):
        des = '<Process name={}'.format(self.name)
        if self.process is not None:
            if self.process.is_alive():
                des += ' running'
            else:
                des += ' exitcode={}'.format(self.process.exitcode)
        
        if self.links:
            des += ' links={}'.format(self.links)
        
        if self.observers:
            des += ' pipes={}'.format(self.observers)
        
        return des + '>'
    
    def __str__(self):
        return self.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if hasattr(other, 'name'): other = other.name
        return self.name == other
    
    def send(self, data, ** kwargs):
        self.input_stream.put(data, ** kwargs)
    
    def get(self, ** kwargs):
        return self.output_stream.get(** kwargs)
    
    def is_alive(self):
        return self.process is not None and self.process.is_alive()
    
    def start(self):
        if self.is_alive(): return self
        
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
        self.finalizer = self.start_finalizer()
        return self
    
    def join(self, ** kwargs):
        self.finalizer.join(** kwargs)

    def add_pipe(self, name):
        self.observers.add(name)
        get_process(name).add_link(self)
        
        if self.forwarder is None:
            self.forwarder = self.start_forwarder()
    
    def add_link(self, name):
        self.links.add(name)
        if self.synchronizer is None:
            self.synchronizer = self.start_keep_alive()

    def remove_link(self, name):
        self.links.remove(name)
    
    @run_in_thread(daemon = True)
    def start_keep_alive(self):
        sleep_time  = self.kwargs.get('timeout', 10) / 2
        
        while self.links:
            time.sleep(sleep_time)
            
            for link in self.links.copy():
                if is_alive(link):
                    self.send(KEEP_ALIVE)
                    break
                else:
                    self.links.remove(link)

        logger.info('Synchronizer finished !')

    @run_in_thread(daemon = True)
    def start_forwarder(self):
        while self in _processes or not self.output_stream.empty():
            try:
                data = self.get(timeout = 1)
            except queue.Empty:
                continue

            for out in self.observers.copy():
                if out in _processes:
                    logger.debug('Transferring data from {} to {}'.format(self, out))
                    start_process(out).send(data)
                else:
                    logger.info('The output of the pipe ({}) does not exist anymore !'.format(out))
                    self.observers.remove(out)
                    get_process(out).remove_link(self)
            
            if len(self.observers) == 0:
                break
        
        logger.info('The process {} has no pipe anymore'.format(self))
        
    @run_in_thread(daemon = True)
    def start_finalizer(self):
        finalize, run, t = False, 0, 0
        while not finalize:
            if self.is_alive():
                self.process.join()
                if self.process.exitcode != 0: finalize = True
                elif self.restart:
                    if self.restart is True or run < self.restart:
                        self.start()
                        run += 1
                    else:
                        finalize = True
            else:
                time.sleep(1)

                if self.pipes['input']:
                    for proc in self.pipes['input']:
                        if proc in _processes and proc.is_alive():
                            logger.info('Waiting `{}` before finalizing `{}`'.format(
                                proc, self.name
                            ))
                            _processes[proc].join()
                else:
                    finalize = True

        logger.info('Finalizing process `{}` (status {}) !'.format(
            self.name, self.process.exitcode if self.process is not None else None
        ))

        if self.callback is not None: self.callback(name = self.name)
        _processes.pop(self)
        logger.info('Process `{}` is closed !'.format(self.name))

def is_alive(name):
    p = get_process(name)
    return p is not None and p.is_alive()

def get_process(name):
    return _processes.get(name, None)

def start_process(name):
    _processes[name].start()
    return _processes[name]

