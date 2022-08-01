from threading import Thread

from utils.thread_utils.consumer import Consumer

def _init_consumer(task, ** kwargs):
    if isinstance(task, Consumer): return task
    if not isinstance(task, dict): task = {'consumer' : task}
    return Consumer(** {** kwargs, ** task})

class Pipeline(Consumer):
    def __init__(self, tasks, * args, name = None, ** kwargs):
        Thread.__init__(self, name = name)
        
        self.consumers  = []
        cons = _init_consumer(tasks[0], ** kwargs)
        self.consumers.append(cons)
        for task in tasks[1:]:
            if not isinstance(task, dict): task = {'consumer' : task}
            task.update({'link_stop' : True, 'start' : False})
            cons = cons.add_consumer(** {** kwargs, ** task})
            self.consumers.append(cons)
        
        super(Consumer, self).__init__(generator = self, ** kwargs)
    
    @property
    def listener(self):
        return self

    @property
    def first(self):
        return self.consumers[0]
    
    @property
    def last(self):
        return self.consumers[-1]
    
    def __iter__(self):
        return iter(self.last)
    
    def __call__(self, item, * args, ** kwargs):
        self.append(item, * args, ** kwargs)
    
    def add_listener(self, * args, ** kwargs):
        self.last.add_listener(* args, ** kwargs)

    def append(self, item, * args, ** kwargs):
        self.first.append(item, * args, ** kwargs)
    
    def start(self):
        for cons in self.consumers:
            if not cons.is_alive(): cons.start()
    
    def stop(self, * args, ** kwargs):
        self.first.stop(* args, ** kwargs)
    
    def join(self, * args, ** kwargs):
        for cons in self.consumers: cons.join(* args, ** kwargs)
    
    def wait(self, * args, ** kwargs):
        for cons in self.consumers: cons.wait(* args, ** kwargs)

    def plot(self, * args, ** kwargs):
        return self.first.plot(* args, ** kwargs)
        