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

import time

from threading import Condition, RLock, Lock

class ThreadedDict(dict):
    """ `dict`-like class with blocking get and thread-safe methods """
    def __init__(self, init = None, ** kwargs):
        self.__data = kwargs if init is None else {** init, ** kwargs}
        self.__cond = {}
        self.__timestamps   = {}
        
        self.__mutex_data   = RLock()
        self.__mutex_cond   = Lock()
    
    @property
    def mutex(self):
        return self.__mutex_data
    
    def __enter__(self):
        self.__mutex_data.__enter__()
    
    def __exit__(self):
        self.__mutex_data.__exit__()

    def __str__(self):
        with self.__mutex_data: return str(self.__data)
    
    def __len__(self):
        with self.__mutex_data: return len(self.__data)
    
    def __contains__(self, key):
        with self.__mutex_data: return key in self.__data
    
    def __getitem__(self, key):
        with self.__mutex_data: return self.get(key)

    def __setitem__(self, key, value):
        with self.__mutex_data:
            self.__data[key] = value
            self.__timestamps[key]  = time.time()
            self.notify_all(key)
    
    def __delitem__(self, key):
        self.pop(key, None)
    
    def get_condition(self, key, create = True):
        # conditions are never popped or modified
        # It is therefore not required to get them without mutex
        if key in self.__cond or not create: return self.__cond.get(key, None)
        with self.__mutex_cond:
            if key not in self.__cond: self.__cond[key] = Condition()
            return self.__cond[key]
    
    def wait_for(self, key, cond = None, ** kwargs):
        if cond is None: cond = lambda: key in self
        condition = self.get_condition(key)
        with condition: condition.wait_for(cond, ** kwargs)

    def wait_for_update(self, key, ** kwargs):
        now = time.time()
        is_updated  = lambda: now <= self.__timestamps.get(key, -1)
        self.wait_for(key, cond = is_updated, ** kwargs)
    
    def notify_all(self, key):
        condition = self.get_condition(key, create = False)
        if condition is not None:
            with condition: condition.notify_all()

    def get(self, key, * args, blocking = True, timeout = None, ** kwargs):
        with self.__mutex_data:
            if key in self.__data or not blocking or len(args) > 0:
                return self.__data.get(key, * args)
        
        self.wait_for(key, timeout = timeout)
        with self.__mutex_data:
            return self.__data.get(key, * args)
    
    def setdefault(self, key, default):
        with self.__mutex_data:
            if key not in self.__data: self[key] = default
            return self[key]
    
    def pop(self, key, * args):
        with self.__mutex_data: return self.__data.pop(key, * args)
    
    def copy(self):
        with self.__mutex_data: return self.__data.copy()
    
    def items(self):
        return self.copy().items()
    
    def keys(self):
        return self.copy().keys()
    
    def values(self):
        return self.copy().values()
    
    def update(self, data):
        now = time.time()
        with self.__mutex_data:
            self.__data.update(data)
            self.__timestamps.update({k : now for k in data})
            for k in data.keys(): self.notify_all(k)
    
