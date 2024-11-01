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

import keras
import logging
import importlib
import numpy as np
import keras.ops as K

from functools import wraps

from loggers import timer
from .execution_contexts import *

logger  = logging.getLogger(__name__)

_is_np_backend_available    = False

def _import_functions(module, _globals = set()):
    return [
        k for k, v in vars(module).items()
        if not k.startswith('_') and getattr(v, '__module__', '').startswith(module.__name__) and not isinstance(v, type) and k not in _globals
    ]

def _is_numpy(args, kwargs, nested = False):
    if nested: args = args[0]
    return not (
        any(K.is_tensor(a) for a in args) or any(K.is_tensor(v) for v in kwargs.values())
    )

def build_op(k_op,
             tf_op = None,
             np_op = None,
             *,
             
             tf_kwargs  = {},
             
             nested = False,
             disable_np = False,
             is_numpy_check = _is_numpy,

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
    keras_fn    = k_op
    np_fn       = np_op if not disable_np else None
    if isinstance(k_op, str):
        keras_fn    = _import_function(keras if k_op.startswith('random') else K, k_op)

    if not disable_np:
        if np_op is None and isinstance(k_op, str):
            if _is_np_backend_available:
                np_fn   = _get_backend_function(k_op, keras_fn, 'numpy')
            else:
                np_fn   = _import_function(np, k_op)
        elif isinstance(np_op, str):
            np_fn   = _import_function(np, np_op)
    
    if keras_fn is None:
        raise ValueError('The operation `{}` is not available in `keras` !'.format(k_op))
    
    if name:
        try:
            keras_fn.__name__ = name
        except AttributeError:
            pass
    else:
        name = keras_fn.__name__
    
    if disable_np:
        assert np_fn is None
    elif np_fn is None and logger.isEnabledFor(logging.DEBUG):
        logger.debug('The `{}` operation does not exist for `numpy`'.format(name))

    @timer(debug = True, name = name)
    def ops(* args, allow_np = True, ** kwargs):
        if is_tensorflow_graph():
            if ops.tf_function is None:
                if tf_op is None:
                    if is_tensorflow_backend():
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('Using the keras function for {}'.format(name))
                        tf_fn = keras_fn
                    elif isinstance(k_op, str):
                        tf_fn = _get_backend_function(k_op, keras_fn, 'tensorflow')
                    else:
                        raise RuntimeError('This operation does not exist in `tensorflow`')
                    
                elif isinstance(tf_op, str):
                    import tensorflow as tf
                    tf_fn = tf.autograph.experimental.do_not_convert(_import_function)(tf, tf_op)
                else:
                    tf_fn = tf_op

                ops.tf_function = tf_fn
            
            return ops.tf_function(* args, ** tf_kwargs, ** kwargs)
        
        elif allow_np and np_fn is not None and is_numpy_check(args, kwargs, nested):
            return np_fn(* args, ** kwargs)
        
        return keras_fn(* args, ** kwargs)

    ops.function    = keras_fn
    ops.tf_function = None
    ops.numpy_function  = np_fn
    
    if tf_op is None:
        ops = wraps(keras_fn)(ops)
    else:
        ops.__name__ = name
        ops.__doc__  = keras_fn.__doc__
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Op `{}` created successfully'.format(name))
    
    return ops

def build_custom_op(tf_fn   = None,
                    torch_fn    = None,
                    jax_fn  = None,
                    keras_fn    = None,
                    np_fn   = None,
                    name    = None,
                    ** kwargs
                   ):
    if is_tensorflow_backend(): fn, tf_fn = tf_fn, None
    elif is_torch_backend():    fn = torch_fn
    elif is_jax_backend():      fn = jax_fn
    
    if np_fn is None:
        np_fn = keras_fn if isinstance(keras_fn, str) else name
        
    if fn is None: fn = name if keras_fn is None else keras_fn
    if isinstance(fn, str):
        _fn = _import_function(get_backend_module(), fn)
        fn = _fn if _fn is not None else _import_function(K, fn)
    
    if fn is None:
        def _raise_undefined(* args, ** kwargs):
            raise NotImplementedError('This operation is not implemented for the `{}` backend'.format(get_backend()))

        logger.debug('The operation {} is not available in {} backend. Calling it will raise an exception'.format(name, get_backend()))
        fn = _raise_undefined
    
    return build_op(fn, tf_op = tf_fn, np_op = np_fn, name = name, ** kwargs)

def _get_backend_function(op_name, op_fn, backend):
    _keras_module = op_fn.__module__.split('.')[-1]
    _keras_module = '{}.{}'.format(_keras_module, op_name.split('.')[-1])

    backend_fn = keras.src.utils.backend_utils.DynamicBackend(backend)
    for part in _keras_module.split('.'):
        if not part: continue
        backend_fn = getattr(backend_fn, part)
    return backend_fn

def _import_function(module, fn):
    if isinstance(fn, (list, tuple)):
        for _fn in fn:
            _fn = _import_function(module, _fn)
            if _fn is not None: return _fn
        return None
    
    if '.' in fn:
        parts   = fn.split('.')
        module, fn = importlib.import_module('.'.join([module.__name__] + parts[:-1])), parts[-1]
    return getattr(module, fn, None)

