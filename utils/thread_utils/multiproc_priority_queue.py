# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
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
import multiprocessing
import multiprocessing.queues

from typing import Any
from dataclasses import dataclass, field

@dataclass(order = True)
class PriorityItem:
    priority : Any
    data     : Any = field(compare = False)

class _MultiprocessingPriorityQueue(multiprocessing.queues.Queue):
    def __init__(self, * args, ** kwargs):
        super().__init__(* args, ** kwargs)
        self._priority_buffer = None
        
    @property
    def priority_buffer(self):
        if not hasattr(self, '_priority_buffer') or self._priority_buffer is None:
            self._priority_buffer = queue.PriorityQueue()
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
        self.fill_buffer()
        return self.priority_buffer.get_nowait()
        
multiprocessing.context.BaseContext.PriorityQueue = lambda self, * args, ** kwargs: _MultiprocessingPriorityQueue(* args, ctx = self.get_context(), ** kwargs)

multiprocessing.PriorityQueue = multiprocessing.context._default_context.PriorityQueue
MultiprocessingPriorityQueue = multiprocessing.PriorityQueue
