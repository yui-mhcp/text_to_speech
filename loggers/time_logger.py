
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
import logging
import collections

try:
    from utils.generic_utils import time_to_string
except ImportError as e:
    from loggers.utils import time_to_string

def format_time(** kwargs):
    if kwargs.get('n_exec', 1) == 1:
        return '- {name} : {total_time}'.format(** kwargs)
    return '- {name} executed {n_exec} times : {total_time} ({mean_time} / exec)'.format(** kwargs)

TIME_LEVEL  = 15

_str_indent     = '  '
_default_format   = format_time

class Timer:
    def __init__(self, name, format = _default_format, parent = None):
        self.name = name
        self.start_time = -1
        self._format = format
        
        self.parent     = parent
        self.children   = {}
        self.runs   = collections.deque()
    
    @property
    def is_root(self):
        return self.parent is None
    
    @property
    def running(self):
        return self.start_time != -1
    
    @property
    def total_time(self):
        return 0. if len(self.runs) == 0 else sum(self.runs)
    
    @property
    def mean_time(self):
        return 0. if len(self.runs) == 0 else sum(self.runs) / len(self.runs)
    
    @property
    def infos(self):
        return {
            'name'      : self.name,
            'n_exec'    : len(self.runs),
            'mean_time' : time_to_string(self.mean_time),
            'total_time'    : time_to_string(self.total_time)
        }
    
    def __str__(self, indent = 0):
        if self.running: return "timer {} not stopped yet".format(self.name)
        
        des = _str_indent * indent + self.format()
        
        for child in self.children.values():
            des += '\n' + child.__str__(indent = indent + 1)
        
        if indent == 0: des = 'timers :\n' + des 
        
        return des

    def format(self):
        if callable(self._format): return self._format(** self.infos)
        return self._format.format(** self.infos)
    
    def add_child(self, name):
        if name not in self.children: self.children[name] = Timer(name, parent = self)
        return self.children[name]
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.runs.append(time.time() - self.start_time)
        self.start_time = -1
    
class RootTimer(Timer):
    def __init__(self, name):
        super().__init__(name = name, parent = None)
        self._timers = collections.deque()
    
    @property
    def running(self):
        return len(self._timers) > 0
    
    @property
    def lowest(self):
        return self._timers[-1] if len(self._timers) > 0 else self
    
    def __str__(self, names = None):
        if self.running: return "timer {} not stopped yet ({} running)".format(
            self.name, self._timers[0].name
        )
        
        if names is None: names = list(self.children.keys())
        elif not isinstance(names, (list, tuple)): names = [names]
    
        if len(names) == 0 or all([n not in self.children for n in names]):
            return 'no logged time for {}'.format(self.name)
        
        des = 'timers :'
        for n in names:
            child = self.children.pop(n, Timer(n, self))

            des += '\n' + child.__str__(indent = 1)
        
        return des

    def start_timer(self, name):
        timer = self.lowest.add_child(name)
        self._timers.append(timer)
        timer.start()
        return timer
    
    def stop_timer(self, name):
        timer = self._timers.pop()
        timer.stop()
        
        while len(self._timers) > 0 and timer.name != name:
            timer = self._timers.pop()
            timer.stop()
        return timer
    
def timer(fn = None, name = None, logger = 'timer', log_if_root = True, force_logging = False):
    if fn is None:
        return lambda fn: timer(
            fn, name = name, logger = logger, log_if_root = log_if_root,
            force_logging = force_logging
        )
    
    if isinstance(logger, str): logger = logging.getLogger(logger)
    elif logger is None: logger = logging.getLogger()
    if name is None: name = fn.__name__
    
    def fn_with_timer(* args, ** kwargs):
        if not logger.isEnabledFor(TIME_LEVEL):
            return fn(* args, ** kwargs)
        
        logger.start_timer(name)
        try:
            result = fn(* args, ** kwargs)
        except Exception as e:
            raise e
        finally:
            logger.stop_timer(name)

        if log_if_root and not logger.timer.running: logger.log_time(name)
        elif force_logging: logger.log_time(timer)
        return result
    
    wrapper = fn_with_timer
    wrapper.__doc__     = fn.__doc__
    wrapper.__name__    = fn.__name__
    
    return wrapper

def start_timer(self, name, * args, ** kwargs):
    if not self.isEnabledFor(TIME_LEVEL): return
    if not hasattr(self, 'timer'): self.timer = RootTimer(self.name)
    self.timer.start_timer(name)
    
def stop_timer(self, name, * args, ** kwargs):
    if not self.isEnabledFor(TIME_LEVEL): return
    if not hasattr(self, 'timer'): self.timer = RootTimer(self.name)
    self.timer.stop_timer(name)

def log_time(self, names = None, * args, ** kwargs):
    if not self.isEnabledFor(TIME_LEVEL): return
    if not hasattr(self, 'timer'): self.timer = RootTimer(self.name)
    des = self.timer.__str__(names) if not isinstance(names, Timer) else str(names)
    self.log(TIME_LEVEL, des, * args, ** kwargs)


logging.addLevelName(TIME_LEVEL, 'TIME')

logging.Logger.start_timer  = start_timer
logging.Logger.stop_timer   = stop_timer
logging.Logger.log_time     = log_time

