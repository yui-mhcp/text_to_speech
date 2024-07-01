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

import time

from functools import wraps
from threading import Condition, RLock, Lock

def locked_method(fn):
    @wraps(getattr(dict, fn))
    def wrapper(self, * args, ** kwargs):
        with self: return getattr(self._data, fn)(* args, ** kwargs)
    return wrapper

class ThreadedDict(dict):
    """ `dict`-like class with blocking get and thread-safe methods """
    def __init__(self, * args, ** kwargs):
        self._data  = dict(* args, ** kwargs)
        self.mutex  = RLock()
        self.__timestamps = {}
        
        self._observers = {}
        self._mutex_observers   = Lock()
    
    def _get_observers(self, key, create = True):
        # conditions are never popped or modified
        # It is therefore not required to get them without mutex
        if key in self._observers: return self._observers.get(key, None)
        with self._mutex_observers:
            if key not in self._observers: self._observers[key] = Condition()
            return self._observers[key]

    def __enter__(self):
        self.mutex.__enter__()
    
    def __exit__(self, * args):
        self.mutex.__exit__(* args)

    __str__ = locked_method('__str__')
    __len__ = locked_method('__len__')
    __contains__    = locked_method('__contains__')
    __delitem__ = lambda self, key: self.pop(key, None)
    
    copy    = locked_method('copy')
    
    items   = lambda self: self.copy().items()
    keys    = lambda self: self.copy().keys()
    values  = lambda self: self.copy().values()

    def __setitem__(self, key, value):
        with self:
            self._data[key] = value
            self.__timestamps[key]  = time.time()
            self.notify_all(key)
    
    def __getitem__(self, key):
        return self.get(key)

    def wait_for(self, key, cond = None, ** kwargs):
        if cond is None: cond = lambda: key in self
        c = self._get_observers(key, create = True)
        if c is not None:
            with c: c.wait_for(cond)

    def wait_for_update(self, key, ** kwargs):
        now = time.time()
        is_updated  = lambda: now <= self.__timestamps.get(key, -1)
        self.wait_for(key, cond = is_updated, ** kwargs)
        return self[key]
    
    def notify_all(self, key):
        cond = self._get_observers(key, create = False)
        if cond is not None:
            with cond: cond.notify_all()

    def get(self, key, default = None, /, *, blocking = True, timeout = None):
        with self:
            if default is not None or not blocking or key in self:
                return self._data.get(key, default)
        
        self.wait_for(key, timeout = timeout)
        with self: return self._data[key]
    
    def pop(self, key, default = None, /):
        with self:
            it = self._data.pop(key, default)
            self.notify_all(key)
        return it
    
    def update(self, data):
        now = time.time()
        with self:
            self._data.update(data)
            self._timestamps.update({k : now for k in data})
            for k in data: self.notify_all(k)
    
