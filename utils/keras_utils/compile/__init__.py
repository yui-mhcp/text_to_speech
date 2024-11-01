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

import inspect
import numpy as np
import keras.ops as K

from functools import wraps

from .. import ops, tree
from .exporter import *
from .compiled_function import *
from .saved_model_function import *

_py_types   = (str, int, float, bool, np.ndarray)

def graph_compile(fn = None, ** kwargs):
    def wrapper(fn):
        if isinstance(fn, CompiledFunction): return fn
        return CompiledFunction(fn, ** kwargs)
    
    return wrapper if fn is None else wrapper(fn)

def execute_eagerly(fn  = None,
                    *,
                    
                    Tout    = None,
                    signature   = None,
                    default_key = (),
                    numpy   = False,
                    name    = None
                   ):
    """
        This function wraps `fn`, a regular python function, such that it can be used transparently inside a ``tf.function` executing by automatically calling `tf.py_function` (default) or `tf.numpy_function` (if `numpy == True`)
        
        Arguments :
            - fn    : the python function to wrap
            
            - Tout  : output types of `fn`
            - signature : (list of) `TensorSpec` that gives both shapes and types information
            - default_key   : the key to use if the 1st argument is a `dict`
            - numpy : whether to use `numpy_function` instead of `py_function`
            - name  : the operation name
        Return :
            If `fn is None`:
                - wraper    : a decorator function
            Else :
                - decorated : the function wrapped
        
        Note : if the function is executed eagerly (i.e. `tf.executing_eagerly() == True`), `fn` is simply called, meaning that the output is **not** forced to be a `tf.Tensor` !
        If `numpy == True`, all `Tensor `args / kwargs are converted to `ndarray`, no matter the backend. 
        
        Note 2 : the benefit of passing `signature` instead of `Tout` is that it will fix the static shape of the output tensor with `tf.ensure_shape`, which may be required for some future usage
        
        Note 3 : when calling the decorated function, 2 new arguments are supported :
            - key       : to override the `default_key` argument
            - shape     : to specify a more precise shape (or in complement to `Tout`), see Note 2
    """
    def wrapper(fn):
        @wraps(fn)
        def inner(* args, shape = Sout, key = default_key, ** kwargs):
            _is_graph = ops.is_tensorflow_graph()
            if _is_graph and is_class_method:
                idx = 0
                function, args = getattr(args[0], fn.__name__), args[1:]
            else:
                idx = 1 if is_class_method else 0
                function = fn
            
            if len(args) > idx and isinstance(args[idx], dict):
                if not isinstance(key, (list, tuple)): key = [key]
                for k in key:
                    if k in args[idx]:
                        args = args[:idx] + (args[idx][k], ) + args[idx + 1 :]
                        break

            if not _is_graph:
                if numpy:
                    args = [
                        ops.convert_to_numpy(a) if not isinstance(a, _py_types) and K.is_tensor(a) else a
                        for a in args
                    ]
                    kwargs  = {
                        k : ops.convert_to_numpy(v) if not isinstance(v, _py_types) and K.is_tensor(v) else v
                        for k, v in kwargs.items()
                    }
                return function(* args, ** kwargs)

            def fn_with_cast(* args, ** kwargs):
                out = function(* args, ** kwargs)
                if isinstance(out, (list, tuple)):
                    if not isinstance(Tout, list):  dtypes = [Tout] * len(out)
                    elif len(Tout) == 1:            dtypes = Tout * len(out)
                    else:                           dtypes = Tout
                    return [ops.cast(out_i, dtype) for out_i, dtype in zip(out, dtypes)]
                else:
                    return ops.cast(out, Tout if not isinstance(Tout, list) else Tout[0])

            import tensorflow as tf
            
            python_function = tf.py_function if not numpy else tf.numpy_function
            
            if not kwargs:
                result = python_function(fn_with_cast, args, Tout = Tout)
            else:
                def fn_with_kwargs(n, * args_and_kwargs):
                    args    = args_and_kwargs[:n]
                    keys    = args_and_kwargs[n]
                    if not isinstance(keys, np.ndarray): keys = keys.numpy()
                    
                    kwargs  = {k.decode() : v for k, v in zip(keys, args_and_kwargs[n + 1 :])}
                    return fn_with_cast(* args, ** kwargs)

                keys    = list(kwargs.keys())
                vals    = [kwargs[k] for k in keys]
                args_with_kv = list(args) + [tf.cast(keys, tf.string)] + vals
                result = python_function(
                    fn_with_kwargs, [len(args)] + args_with_kv, Tout = Tout
                )

            if shape is not None:
                if isinstance(shape, tuple):
                    shape = tf.TensorShape(shape)
                elif isinstance(shape, list):
                    shape = tuple(tf.TensorShape(s) if isinstance(s, tuple) else s for s in shape)
                
                if isinstance(result, list):
                    if not isinstance(shape, tuple):    shape = tuple([shape] * len(result))
                    elif len(shape) == 1:               shape = shape * len(result)
                    
                    result = tuple(tf.ensure_shape(res, s) for res, s in zip(result, shape))
                else:
                    result = tf.ensure_shape(
                        result, shape if not isinstance(shape, tuple) else shape[0]
                    )
            elif isinstance(result, list):
                result = tuple(result)
            
            return result
        
        is_class_method     = 'self' == list(inspect.signature(fn).parameters.keys())[0]

        inner.Tout      = Tout
        inner.signature = signature
        inner.default_key   = default_key

        return inner
    
    assert Tout is not None or signature is not None

    Sout = None
    if signature is not None:
        Tout    = tree.map_structure(lambda s: s.dtype, signature)
        Sout    = tree.map_structure(lambda s: s.shape, signature)
    elif isinstance(Tout, tuple):
        Tout    = list(Tout)
    
    if not isinstance(default_key, (list, tuple)): default_key = [default_key]
    return wrapper if fn is None else wrapper(fn)

def tensorflow_only_function(fn):
    if ops.is_tensorflow_backend(): return fn
    
    @wraps(fn)
    def inner(* args, ** kwargs):
        if ops.is_tensorflow_graph(): return fn(* args, ** kwargs)
        
        try:
            import tensorflow as tf
            args = tf.nest.map_structure(
                lambda x: ops.convert_to_tf_tensor(x) if ops.is_tensor(x) else x, args
            )
            kwargs = tf.nest.map_structure(
                lambda x: ops.convert_to_tf_tensor(x) if ops.is_tensor(x) else x, kwargs
            )
        except ImportError:
            raise ops.TensorflowNotAvailable()
        
        return fn(* args, ** kwargs)
    return inner

