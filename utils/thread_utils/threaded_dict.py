from collections import defaultdict
from threading import Condition, RLock, Lock

class ThreadedDict(dict):
    """ `dict`-like class with blocking get and thread-safe methods """
    def __init__(self, init = None, ** kwargs):
        self.__data = kwargs if init is None else {** init, ** kwargs}
        self.__cond = {}
        
        self.__mutex_data   = RLock()
        self.__mutex_cond   = Lock()
    
    @property
    def mutex(self):
        return self.__mutex_data
    
    def __str__(self):
        return str(self.__data)
    
    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __delitem__(self, key):
        self.pop(key, None)
    
    def __setitem__(self, key, value):
        with self.__mutex_data:
            self.__data[key] = value
        self.notify_all(key)
    
    def __contains__(self, key):
        return key in self.__data
    
    def get_condition(self, key, create = True):
        if key in self.__cond: return self.__cond[key]
        with self.__mutex_cond:
            if create and key not in self.__cond: self.__cond[key] = Condition()
            return self.__cond.get(key, None)
    
    def wait_for(self, key, cond = None, ** kwargs):
        if cond is None: cond = lambda: key in self
        cv = self.get_condition(key)
        with cv: cv.wait_for(cond, ** kwargs)

    def notify_all(self, key):
        cv = self.get_condition(key, create = False)
        if cv is not None:
            with cv: cv.notify_all()

    def get(self, key, * args, blocking = True, timeout = None, ** kwargs):
        with self.__mutex_data:
            if key in self or not blocking or len(args) > 0:
                return self.__data.get(key, * args)
        
        self.wait_for(key, timeout = timeout)
        with self.__mutex_data:
            return self.__data.get(key, * args)
    
    def setdefault(self, key, default):
        with self.__mutex_data:
            if key not in self: self[key] = default
            return self[key]
    
    def pop(self, key, * args):
        with self.__mutex_data: return self.__data.pop(key, * args)
    
    def items(self):
        return self.__data.copy().items()
    
    def keys(self):
        return self.__data.copy().keys()
    
    def values(self):
        return self.__data.copy().values()
    
    def update(self, data):
        with self.__mutex_data:
            self.__data.update(data)
        for k, v in data.items():
            self.notify_all(k)
    
