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

import sys
import keras
import importlib
import threading

from functools import cache

get_backend     = keras.backend.backend
is_tensorflow_backend   = lambda: get_backend() == 'tensorflow'
is_torch_backend        = lambda: get_backend() == 'torch'
is_jax_backend          = lambda: get_backend() == 'jax'

_is_executing_xla = {}
_should_execute_eagerly = {}

def get_backend_module():
    return importlib.import_module(get_backend())

def get_backend_version():
    return get_backend_module().__version__

@cache
def is_tensorflow_available():
    try:
        import tensorflow as tf
        return True
    except:
        return False

def is_tensorflow_graph():
    """
        This function is equivalent to `tf.executing_eagerly` while enabling to not import tensorflow by default
    """
    if is_tensorflow_backend() or  'tensorflow' in sys.modules:
        import tensorflow as tf
        return not tf.executing_eagerly()
    return False

def executing_eagerly():
    """
        This function returns whether the code is executing eagerly or not (i.e., XLA compiled)
        Note that there is no equivalent to `tf.executing_eagerly()` in other backends. To overcome this, the `graph_compile` function calls the `set_xla_execution` when running a code in XLA
        This function will only detect XLA-codes executed by `graph_compile`, and not regular compilation (like `jax.jit` or `torch.compile`)
        In the `tensorflow` backend, this function is equivalent to `tf.executing_eagerly()`
        
        Note that the function is thread-safe (for other backends), meaning that executing it in a separate thread that is executing eagerly will correctly return True, no matter if another thread is running XLA code at the same time
    """
    if is_tensorflow_backend(): # shortcut to speed up for tensorflow backend
        return not is_tensorflow_graph()
    elif is_tensorflow_graph(): # should return False if executing in `tf.data` (for all backends)
        return False
    return _get_thread_id() not in _is_executing_xla

def set_xla_execution(use_xla):
    if is_tensorflow_backend(): return
    if use_xla: _is_executing_xla[_get_thread_id()] = True
    else:       _is_executing_xla.pop(_get_thread_id())

def set_eager_execution(eager):
    if eager:   _should_execute_eagerly[_get_thread_id()] = True
    else:       _should_execute_eagerly.pop(_get_thread_id())

def should_execute_eagerly():
    """
        This function returns whether a graph-compiled function should be executed eagerly or not
        The typical use case is to execute eagerly all nested functions if it has been required via the `execute_eagerly` argument from a `graph_compile`d function
        
        Note that the function is thread-safe, meaning that if one function thread A is requested to run eagerly, it will not afect functions from other threads
    """
    return _get_thread_id() in _should_execute_eagerly

class XLAExecution:
    def __init__(self, force_tensorflow = False):
        if not force_tensorflow:
            self.enter = lambda: set_xla_execution(True)
            self.exit  = lambda * _: set_xla_execution(False)
        else:
            self.enter = self.exit = lambda: None
    
    def __enter__(self):
        self.enter()
        return self
    
    def __exit__(self, * args):
        self.exit()
    

class EagerExecution:
    def __enter__(self):
        set_eager_execution(True)
        return self
    
    def __exit__(self, * args):
        set_eager_execution(False)

def _get_thread_id():
    return threading.current_thread().ident

