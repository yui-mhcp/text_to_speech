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

import logging

from .callback import Callback

logger = logging.getLogger(__name__)

class FunctionCallback(Callback):
    def __init__(self, fn, name = None, include_outputs = True, ** kwargs):
        if name is None: name = getattr(fn, '__name__', fn.__class__.__name__)
        
        self.fn = fn
        self.include_outputs    = include_outputs
        
        super().__init__(name = name, ** kwargs)
    
    def apply(self, infos, output, ** kwargs):
        kwargs.update(infos)
        if self.include_outputs: kwargs.update(output)
        return self.fn(** kwargs)

class QueueCallback(Callback):
    def __init__(self, queue, name = 'queue', ** kwargs):
        super().__init__(name = name, ** kwargs)
        
        self.queue  = queue
    
    
    def apply(self, infos, output, ** kwargs):
        kwargs.update(infos)
        return self.queue.put(kwargs)
