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
import logging
import threading
import collections

from functools import wraps

logger = logging.getLogger(__name__)

TIME_LEVEL      = 15
TIME_DEBUG_LEVEL    = 13
TIME_LOGGER_NAME    = None

_str_indent     = '  '

class RootTimer:
    """
        Main Timer class which stores recursively information on the execution time of functions / nested functions
        The timer is thread-safe and stores each thread's timer in a different dict
        
        `_timers` stores the different timers in a tree-like structured dict : 
        {thread : timer} wher `timer` is a dict {name :, runs : [], start :, children {timer...}}
        
        `_runnings` stores the running timer reference in a list : {thread : references...[]}
        
        Note that `_runnings` only contains references to the corresponding timer in the `_timers` structure, meaning that modifying the `_runnings`'s timers will automatically modify the timer in `_timers`
    """
    def __init__(self):
        self.mutex     = threading.Lock()
        self._timers   = collections.OrderedDict()
        self._runnings = {}
        self._thread_names  = {}
    
    def __str__(self, names = None):
        def thread_to_str(thread_timers, ** kwargs):
            des = ''
            for _, timer in thread_timers.items():
                des += '\n' + _timer_to_str(timer, ** kwargs)
            return des
        
        if self.is_running(): return "timer {} not stopped yet".format(self.name)
        
        with self.mutex:
            _timers = self._timers
            self._timers, self._runnings, self._thread_names = collections.OrderedDict(), {}, {}
        
        des = 'Timers :'
        if len(_timers) == 1:
            des += thread_to_str(list(_timers.values())[0])
        else:
            for thread, timers in _timers.items():
                if len(timers) == 0: continue
                
                des += '\n- Timers in thread {} (id {}) :{}'.format(
                    self._thread_names[thread], thread, thread_to_str(timers, indent = 1)
                )
        
        return des
    
    def get_thread_timers(self):
        """
            This method is the only one required to be thread-safe as it (possibly creates) and returns the timers for a specific thread
            Once returned, the manipulation of these timers will be, by design, thread-safe as functions are executed in a single thread then will modify in a sequential order
        """
        thread_id   = _get_thread_id()
        if thread_id not in self._timers:
            with self.mutex:
                self._timers[thread_id]   = collections.OrderedDict()
                self._runnings[thread_id] = collections.deque()
                self._thread_names[thread_id] = threading.current_thread().name
        
        return self._timers[thread_id], self._runnings[thread_id]
    
    def is_running(self):
        return any(len(runnings) > 0 for t, runnings in self._runnings.items())

    def start_timer(self, name, level):
        """
            Add a new timer to the current running timer
            If no timer is running for this thread (i.e., `len(runnings) == 0`, it is added to the main thread timer `timers`)

            Arguments :
                - name  : the name of the timer to start
            Return :
                - new_timer : a new `dict` representing the new current timer

            Note : this function is thread-safe by design because it manipulates data structures belonging to the current thread. This means that other threads will necessarly modify different structures.
        """
        if not _should_track(level): return
        
        timers, runnings = self.get_thread_timers()
        
        if runnings:
            current = runnings[-1]
            if 'children' not in current: current['children'] = collections.OrderedDict()
            current = current['children']
        else:
            current = timers

        if name not in current: current[name] = new_timer = _create_timer(name)
        else: new_timer = current[name]

        new_timer['start'] = time.time()
        runnings.append(new_timer)
        return new_timer
    
    def stop_timer(self, name):
        timers, runnings = self.get_thread_timers()
        if not runnings:
            raise RuntimeError('No timer is running when stopping `{}` for thread `{}`'.format(
                name, self._thread_names[_get_thread_id()]
            ))
        
        timer = runnings.pop()
        timer['runs'].append(time.time() - timer['start'])
        if timer['name'] != name:
            raise RuntimeError('The currently running timer `{}` is not the stopped one `{}`. Make sure to stop them in the correct order'.format(
                timer['name'], name
            ))

        return timer
    

class Timer:
    def __init__(self, name, debug = False):
        self.name   = name
        self.level  = TIME_LEVEL if not debug else TIME_DEBUG_LEVEL
        self._timer = None
    
    @property
    def timers(self):
        if not self._timer: return None
        times = {self.name : sum(self._timer['runs'])}
        return self.get_times(times, self._timer['children'])
    
    def __enter__(self):
        self._timer = start_timer(self.name, self.level)
        return self
    
    def __exit__(self, * args):
        if self._timer: stop_timer(self.name)
    
    def log(self):
        if not self._timer: return
        time_logger.log(TIME_LEVEL, _timer_to_str(self._timer))
    
    def save(self, filename, overwrite = False):
        if os.path.exists(filename) and not overwrite:
            print('This file already exists ! To overwrite, pass `overwrite = True`')
            return
        
        infos = json.dumps(self.timers)
        with open(filename, 'w') as file:
            file.write(infos)
    
    @staticmethod
    def get_times(times, timers):
        for name, timer in timers.items():
            times.setdefault(name, []).extend(timer['runs'])
            Timer.get_times(times, timer.get('children', {}))
        return times

def timer(name = None, *, debug = False, log_if_root = True, fn = None):
    """
        Decorator that will track execution time for the decorated function `fn`
        Arguments :
            - name  : the name of the timer (by default, the name of the function)
            - debug : whether to use TIME_LEVEL or TIME_DEBUG_LEVEL
            - log_if_root   : whether to print if it is the root timer
            - fn    : the function to decorate
    """
    def wrapper(fn):
        @wraps(fn)
        def fn_with_timer(* args, ** kwargs):
            if not start_timer(timer_name, level = level): return fn(* args, ** kwargs)
            try:
                return fn(* args, ** kwargs)
            finally:
                stop_timer(timer_name)

                if log_if_root and not is_timer_running(): log_time(level)
        
        
        timer_name = name if name else (fn.name if hasattr(fn, 'name') else fn.__name__)
        return fn_with_timer
    
    level = TIME_LEVEL if not debug else TIME_DEBUG_LEVEL
    if callable(name): fn, name = name, None
    return wrapper if fn is None else wrapper(fn)

def log_time(level = TIME_LEVEL):
    time_logger.log(level, _root_timer)

def time_to_string(seconds):
    """ Returns a string representation of a time (given in seconds) """
    if seconds < 0.001: return '{} \u03BCs'.format(int(seconds * 1000000))
    if seconds < 0.01:  return '{:.3f} ms'.format(seconds * 1000)
    if seconds < 1.:    return '{} ms'.format(int(seconds * 1000))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = ((seconds % 3600) % 60)
    
    return '{}{}{}'.format(
        '' if h == 0 else '{}h '.format(h),
        '' if m == 0 else '{}min '.format(m),
        '{:.3f} sec'.format(s) if m + h == 0 else '{}sec'.format(int(s))
    )

def _create_timer(name):
    return {'name' : name, 'runs' : [], 'start' : -1}

def _timer_to_str(timer, indent = 0, str_indent = _str_indent):
    _indent = indent * str_indent

    if len(timer['runs']) > 1:
        _infos = 'executed {} times : {} ({} / exec)'.format(
            len(timer['runs']),
            time_to_string(sum(timer['runs'])),
            time_to_string(sum(timer['runs']) / len(timer['runs']))
        )
    elif len(timer['runs']) == 0:
        _infos = 'not finished yet'
    else:
        _infos = ': {}'.format(time_to_string(timer['runs'][0]))

    des = '{}- {} {}'.format(_indent, timer['name'], _infos)
    for child_name, child_timer in timer.get('children', {}).items():
        des += '\n' + _timer_to_str(child_timer, indent = indent + 1, str_indent = str_indent)
    return des

_get_thread_id  = threading.get_ident

time_logger = logging.getLogger(TIME_LOGGER_NAME)
_should_track   = time_logger.isEnabledFor

_root_timer = RootTimer()
start_timer = _root_timer.start_timer
stop_timer  = _root_timer.stop_timer
is_timer_running    = _root_timer.is_running

logging.Logger.start_timer  = start_timer
logging.Logger.stop_timer   = stop_timer
logging.Logger.log_time     = log_time
logging.Logger.timer        = Timer

logging.start_timer  = time_logger.start_timer
logging.stop_timer   = time_logger.stop_timer
logging.log_time     = time_logger.log_time
logging.timer        = time_logger.timer

