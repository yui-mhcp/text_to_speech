from tqdm import tqdm
from threading import Thread
from multiprocessing import cpu_count

class ThreadPool(object):
    def __init__(self, max_workers = cpu_count(), with_result = True, ** kwargs):
        self.max_workers    = max_workers
        self.with_result    = with_result
        self.__threads      = []
        self.__thread   = None
        self.tqdm       = lambda x: x
        self.__thread_type = Thread if not with_result else ThreadWithReturn
        self.__done = 0
        self.__default_kwargs = kwargs
    
    def __len__(self):
        return len(self.__threads)
    
    def __str__(self):
        return 'ThreadPool with {} threads and status : {} (done {:.2f} %)'.format(self.__len__(), self.status(), self.progress() * 100)
    
    def status(self):
        if self.__thread is None: return 'not started'
        elif self.__thread.is_alive(): return 'running'
        else: return 'finished'
    
    def append(self, *args, ** kwargs):
        for k, v in self.__default_kwargs.items(): kwargs.setdefault(k, v)
        t = self.__thread_type(* args, ** kwargs)
        self.__threads.append(t)
    
    def run(self):
        threads_actifs = []
        for t in self.tqdm(self.__threads):
            if len(threads_actifs) >= self.max_workers:
                threads_actifs[0].join()
                threads_actifs.pop(0)
                self.__done += 1
            
            threads_actifs.append(t)
            t.start()
        
        for t in threads_actifs:
            t.join()
            self.__done += 1
    
    def start(self, *args, tqdm = tqdm, ** kwargs):
        if self.__thread is None:
            self.tqdm = tqdm
            self.__thread = Thread(* args, target = self.run, ** kwargs)
            self.__thread.start()
    
    def join(self):
        if self.status() == 'running':
            self.__thread.join()
    
    def progress(self):
        return self.__done / len(self.__threads) if len(self.__threads) > 0 else 1.
        
    def result(self):
        status = self.status()
        if status == 'running': self.join()
        elif status == 'not started': 
            raise ValueError("You must start the pool to get result")
        if not self.with_result: return None
        return [t.result() for t in self.__threads]

class ThreadWithReturn(Thread):
    def __init__(self, *args, **kwargs):
        Thread.__init__(self, *args, **kwargs)
        self._result = None
        self._done = False
        
    def run(self):
        self._result = self._target(* self._args, ** self._kwargs)
        self._done = True
    
    def result(self):
        if not self._done:
            self.join()
        return self._result