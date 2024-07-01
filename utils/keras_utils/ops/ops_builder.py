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
import logging
import importlib
import threading
import numpy as np
import keras.ops as K

from functools import wraps

from loggers import timer

logger  = logging.getLogger(__name__)

get_backend     = keras.backend.backend
is_tensorflow_backend   = lambda: get_backend() == 'tensorflow'
is_torch_backend        = lambda: get_backend() == 'torch'
is_jax_backend          = lambda: get_backend() == 'jax'

_is_executing_xla = {}

def get_backend_module():
    if is_tensorflow_backend():
        import tensorflow as tf
        return tf
    elif is_torch_backend():
        import torch
        return torch
    elif is_jax_backend():
        import jax
        return jax
    elif get_backend() == 'numpy':
        return np

def get_backend_version():
    return get_backend_module().__version__

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
        
def build_op(k_op,
             tf_op = None,
             np_op = None,
             tf_kwargs  = {},
             disable_np = False,
             is_numpy_check = None,
             nested = False,
             name   = None
            ):
    """
        Builds a custom operation that internally calls the `keras.ops` equivalent in normal mode, while forcing the equivalent tensorflow operation when running in `tensorflow graph` (e.g., `tf.data` pipeline)
        By default, this function also enables the numpy version (if available), to avoid unnecessary convertion CPU - GPU for small data
        
        Arguments :
            - k_op  : the `keras.ops` operation name, or `callable`
            - tf_op : the `tensorflow` operation name, or `callable` (if `None`, set to `k_op`)
            - np_op : the `numpy` operation name (only relevant if `disable_np = False`)
            - disable_np    : whether to allow numpy operation or not
            - nested    : whether the op takes nested `Tensor` (e.g., `list of Tensor`), in order to check whether to use numpy or not
        Return :
            - ops   : a callable that wraps the original functions
    """
    if tf_op is None: tf_op = k_op
    if np_op is None and not disable_np: np_op = k_op if isinstance(k_op, str) else name
    
    root_module = K
    if isinstance(k_op, str) and k_op.startswith('keras.'):
        root_module, k_op = keras, k_op.replace('keras.', '')
    keras_fn    = _import_function(root_module, k_op) if isinstance(k_op, str) else k_op
    np_fn       = _import_function(np, np_op) if isinstance(np_op, str) else np_op
    if disable_np: np_fn = None
    elif np_fn is None: logger.debug('The `{}` does not exist for `numpy`'.format(np_op))
    if is_numpy_check is None: is_numpy_check = _is_numpy
    
    if name is not None:
        try:
            keras_fn.__name__ = name
        except AttributeError:
            pass
    
    @timer(debug = True, name = name if name else keras_fn.__name__)
    def ops(* args, allow_np = True, ** kwargs):
        if is_tensorflow_graph():
            import tensorflow as tf
            if isinstance(tf_op, str):
                tf_fn = tf.autograph.experimental.do_not_convert(_import_function)(tf, tf_op)
            else:
                tf_fn = tf_op
            if tf_fn is None: tf_fn = keras_fn
            return tf_fn(* args, ** tf_kwargs, ** kwargs)
        elif allow_np and np_fn is not None and is_numpy_check(args, kwargs, nested):
            return np_fn(* args, ** kwargs)
        return keras_fn(* args, ** kwargs)

    ops.function    = keras_fn
    ops.numpy_function  = np_fn
    
    ops.__name__ = keras_fn.__name__
    ops.__doc__  = keras_fn.__doc__
    return ops

def build_custom_op(tf_fn   = None,
                    torch_fn    = None,
                    jax_fn  = None,
                    keras_fn    = None,
                    np_fn   = None,
                    name    = None,
                    ** kwargs
                   ):
    if is_tensorflow_backend(): fn = tf_fn
    elif is_torch_backend():    fn = torch_fn
    elif is_jax_backend():      fn = jax_fn
    
    if fn is None and name is not None: fn = name
    if isinstance(fn, str):
        _fn = _import_function(get_backend_module(), fn)
        fn = _fn if _fn is not None else _import_function(K, fn)
    
    if fn is None:
        if keras_fn is not None: fn = keras_fn
        else:
            def _raise_undefined(* args, ** kwargs):
                raise NotImplementedError('This operation is not implemented for the `{}` backend'.format(get_backend()))
                
            logger.debug('The operation {} is not available in {} backend. Calling it will raise an exception'.format(name, get_backend()))
            fn = _raise_undefined
    
    return build_op(fn, tf_op = tf_fn, np_op = np_fn, name = name, ** kwargs)

def _import_function(module, fn):
    if '.' in fn:
        parts   = fn.split('.')
        module, fn = importlib.import_module('.'.join([module.__name__] + parts[:-1])), parts[-1]
    return getattr(module, fn, None)

def _is_numpy(args, kwargs, nested = False):
    if nested: args = args[0]
    return not (
        any(K.is_tensor(a) for a in args) or any(K.is_tensor(v) for v in kwargs.values())
    )

def _get_thread_id():
    return threading.current_thread().ident
