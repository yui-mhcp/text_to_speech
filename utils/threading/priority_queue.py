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

from typing import Any
from threading import Lock
from dataclasses import dataclass, field

@dataclass(order = True)
class PriorityItem:
    priority    : Any
    index       : int
    data        : Any = field(compare = False)

class PriorityQueue(queue.PriorityQueue):
    def __init__(self, maxsize = 0):
        super().__init__(maxsize)
        self.__mutex    = Lock()
        self.__index    = 0
    
    def _get_index(self):
        with self.__mutex:
            idx = self.__index
            self.__index += 1
        return idx
    
    def _build_item(self, item, priority = None):
        if isinstance(item, PriorityItem) or all(hasattr(item, attr) for attr in ('priority', 'index', 'data')):
            return item
        if priority is None:
            if isinstance(item, tuple) and len(item) == 2:
                priority = item[0]
            elif isinstance(item, dict) and 'priority' in item:
                priority = item['priority']
            elif hasattr(item, 'priority'):
                priority = item.priority
            else:
                raise ValueError('You must provide the `priority` for the item : {}'.format(item))
        return PriorityItem(priority, self._get_index(), item)
    
    def put(self, item, priority = None, block = True, timeout = None):
        return super().put(self._build_item(item, priority), block, timeout)
    
    def put_nowait(self, item, priority = None):
        return super().put_nowait(self._build_item(item, priority))
    
    def get(self, block = True, timeout = None, return_full_item = False):
        item = super().get(block = block, timeout = timeout)
        return item if return_full_item else item.data

    def get_nowait(self, block = True, timeout = None, return_full_item = False):
        item = super().get_nowait(block = block, timeout = timeout)
        return item if return_full_item else item.data


class _MultiprocessingPriorityQueue(multiprocessing.queues.Queue):
    def __init__(self, * args, ** kwargs):
        super().__init__(* args, ** kwargs)
        self._priority_buffer = None
        
    @property
    def priority_buffer(self):
        if getattr(self, '_priority_buffer', None) is None:
            self._priority_buffer = PriorityQueue()
        return self._priority_buffer
    
    def fill_buffer(self, ** kwargs):
        if self.priority_buffer.qsize() == 0:
            self.priority_buffer.put(super().get(** kwargs))
        try:
            while True:
                self.priority_buffer.put(super().get(block = False))
        except queue.Empty:
            pass

    def get(self, ** kwargs):
        self.fill_buffer(** kwargs)
        item = self.priority_buffer.get(** kwargs)
        return item if not isinstance(item, PriorityItem) else item.data
    
    def get_nowait(self):
        self.fill_buffer(block = False)
        return self.priority_buffer.get_nowait()

multiprocessing.context.BaseContext.PriorityQueue = lambda self, * args, ** kwargs: _MultiprocessingPriorityQueue(* args, ctx = self.get_context(), ** kwargs)

multiprocessing.PriorityQueue = multiprocessing.context._default_context.PriorityQueue
MultiprocessingPriorityQueue = multiprocessing.PriorityQueue
