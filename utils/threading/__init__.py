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

import queue
import multiprocessing
import multiprocessing.queues

from functools import wraps
from threading import Thread

from .priority_queue import PriorityQueue

_buffers    = {
    'queue' : (queue.Queue, multiprocessing.Queue),
    'fifo'  : (queue.Queue, multiprocessing.Queue),
    'stack' : (queue.LifoQueue, ),
    'lifo'  : (queue.LifoQueue, ),
    'priority'  : (PriorityQueue, multiprocessing.PriorityQueue),
    'min_priority'  : (PriorityQueue, multiprocessing.PriorityQueue),
    'max_priority'  : (PriorityQueue, multiprocessing.PriorityQueue),
}

def get_buffer(buffer = 'fifo', maxsize = 0, use_multiprocessing = False):
    if buffer is None: buffer = 'queue'

    if isinstance(buffer, str):
        buffer = buffer.lower()
        if buffer not in _buffers:
            raise ValueError('`buffer` is an unknown queue type :\n  Accepted : {}\n  Got : {}\n'.format(tuple(_buffers.keys()), buffer))
        
        idx = 0 if not use_multiprocessing else 1
        buffer = _buffers[buffer][idx](maxsize)
    
    elif not isinstance(buffer, (queues.Queue, multiprocessing.queues.Queue)):
        raise ValueError('`buffer` must be a Queue instance or subclass')

    return buffer

def locked_property(name):
    def getter(self):
        with self.mutex: return getattr(self, '_' + name)
    
    def setter(self, value):
        with self.mutex: setattr(self, '_' + name, value)
    
    return property(fget = getter, fset = setter)

def run_in_thread(fn = None, name = None, callback = None, ** thread_kwargs):
    def wrapper(fn):
        @wraps(fn)
        def inner(* args, ** kwargs):
            thread = Thread(
                target = fn, args = args, kwargs = kwargs, name = name or fn.__name__, ** thread_kwargs
            )
            thread.start()
            
            if callback is not None: callback(thread, * args, ** kwargs)
            
            return thread
        return inner
    return wrapper if fn is None else wrapper(fn)

def get_name(item, name = None, error = True):
    if name: return name
    elif hasattr(item, 'name'):         return item.name
    elif hasattr(item, '__name__'):     return item.__name__
    elif hasattr(item, '__class__'):    return item.__class__.__name__
    elif error: raise RuntimeError('Cannot infer name for {}'.format(item))
    else:       return None

from .process import Process
from .producer import StoppedException, Producer, Event
from .consumer import Consumer
from .threaded_dict import ThreadedDict
from .priority_queue import PriorityQueue, MultiprocessingPriorityQueue, PriorityItem
