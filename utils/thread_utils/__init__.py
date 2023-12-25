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

from functools import wraps
from threading import Thread

from utils.thread_utils.producer import StoppedException, Producer, Event, Item
from utils.thread_utils.consumer import Consumer
from utils.thread_utils.threaded_dict import ThreadedDict
from utils.thread_utils.multiproc_priority_queue import MultiprocessingPriorityQueue, PriorityItem

def run_in_thread(fn = None, name = None, callback = None, ** thread_kwargs):
    def wrapper(fn):
        @wraps(fn)
        def inner(* args, ** kwargs):
            thread = Thread(
                target = fn, args = args, kwargs = kwargs, name = name, ** thread_kwargs
            )
            thread.start()
            
            if callback is not None: callback(thread, * args, ** kwargs)
            
            return thread
        
        thread_name = name if name else fn.__name__
        
        return inner
    return wrapper if fn is None else wrapper(fn)