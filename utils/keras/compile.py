# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import enum
import inspect
import logging
import warnings
import numpy as np

from functools import wraps
from typing import Union, Tuple
from dataclasses import dataclass

from . import ops, timer

logger  = logging.getLogger(__name__)

@dataclass
class TensorSpec:
    shape   : Union[None, Tuple[int]] = None
    dtype   : str   = None
    name    : str   = None
    static  : bool  = False
    
    def __hash__(self):
        return hash((self.shape, self.dtype))
    
    def __eq__(self, o):
        return self.name == o.name and self.dtype == o.dtype

class ExecutionMode(enum.IntEnum):
    EAGER   = 0
    GRAPH   = 1
    XLA     = 2

def graph_compile(fn    = None,
                  *,
                  
                  kwargs_annots  = None,
                  follow_type_hints  = True,

                  support_xla    = True,
                  prefer_xla     = None,
                  force_tensorflow   = False,

                  prepare    = None,
                  prepare_for_xla    = None,
                  prepare_for_graph  = None,

                  static_args    = 'auto',
                  input_signature    = None,
                  reduce_retracing   = True,

                  ** compile_kwargs
                 ):
    def wrapper(fn):
        @wraps(fn if not hasattr(fn, 'call') else fn.call)
        def inner(* args, run_eagerly = None, use_xla = None, recompile = False, ** kwargs):
            other_kwargs = {}
            if not _supports_kwargs:
                other_kwargs    = {k : v for k, v in kwargs.items() if k not in _signature.parameters}
                kwargs = {k : v for k, v in kwargs.items() if k in _signature.parameters}

            inputs = _signature.bind(* args, ** kwargs)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Unprocessed kwargs : {}'.format(inputs))

            inputs.apply_defaults()
            if prepare is not None:
                inputs.arguments.update(_call_from_bounded_args(
                    prepare, inputs, other_kwargs
                ))

            if force_tensorflow and not ops.is_tensorflow_available():
                warnings.warn(
                    'Tensorflow is not available, running the function {} eagerly'.format(fn_name)
                )
                return _call_from_bounded_args(fn, inputs)
            elif not ops.executing_eagerly():
                return _call_from_bounded_args(fn, inputs)

            execution_mode, ctx_manager = _infer_execution_mode(
                run_eagerly = run_eagerly,

                use_xla = use_xla,
                prefer_xla  = prefer_xla,
                support_xla = support_xla,

                force_tensorflow    = force_tensorflow
            )
            
            if execution_mode == ExecutionMode.EAGER:
                if ctx_manager is not None:
                    with ctx_manager: return _call_from_bounded_args(fn, inputs)
                return _call_from_bounded_args(fn, inputs)
            
            if follow_type_hints:
                inputs.arguments.update(_cast_arg(
                    inputs.arguments, _annotations, force_tensorflow, execution_mode
                ))

            if kwargs_annots and inputs.arguments.get('kwargs', None):
                inputs.arguments['kwargs'].update(_cast_arg(
                    inputs.arguments['kwargs'], kwargs_annots, force_tensorflow, execution_mode
                ))

            if prepare_for_graph is not None:
                inputs.arguments.update(_call_from_bounded_args(
                    prepare_for_graph, inputs, other_kwargs
                ))

            key = 'graph'
            _compile_kwargs = compile_kwargs.copy()
            if execution_mode == ExecutionMode.XLA:
                key = 'xla'
                if prepare_for_xla is not None:
                    inputs.arguments.update(_call_from_bounded_args(
                        prepare_for_xla, inputs, other_kwargs
                    ))

                if (ops.is_jax_backend()
                    and not force_tensorflow
                    and static_args == 'auto'):

                    _compile_kwargs['static_args'] = _get_static_args(inputs)

            if recompile or key not in _compiled:
                _compiled[key] = timer(fn = compile_function(
                    fn,
                    jit_compile = execution_mode == ExecutionMode.XLA,
                    force_tensorflow    = force_tensorflow,
                    ** _compile_kwargs
                ), name = '{}_{}'.format(key, _name), log_if_root = False)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('The function {} is executed in {} mode with {}'.format(
                    _name, key, inputs
                ))

            if ctx_manager is not None:
                with ctx_manager: return _call_from_bounded_args(_compiled[key], inputs)
            return _call_from_bounded_args(_compiled[key], inputs)
        
        inner.fn    = fn
        
        _name   = inner.__name__
        _signature  = inspect.signature(inner.__wrapped__)
        _supports_kwargs    = 'kwargs' in _signature.parameters
        _annotations    = _get_annotations(inner.__wrapped__)
        
        _compiled   = {'eager' : fn}
        
        return inner
    
    compile_kwargs['reduce_retracing'] = reduce_retracing
    return wrapper(fn) if fn is not None else wrapper

def execute_eagerly(fn  = None,
                    *,
                    
                    name    = None,
                    Tout    = None,
                    numpy   = False,
                    signature   = None
                   ):
    """
        This function wraps `fn`, a regular python function, such that it can be used transparently inside a ``tf.function` executing by automatically calling `tf.py_function` (default) or `tf.numpy_function` (if `numpy == True`)
        
        Arguments :
            - fn    : the python function to wrap
            
            - Tout  : output types of `fn`
            - signature : (list of) `TensorSpec` that gives both shapes and types information
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
        
        Note 3 : when calling the decorated function, the special `shape` argument is supported :
            - shape     : to specify a more precise shape (or in complement to `Tout`), see Note 2
    """
    def wrapper(fn):
        @wraps(fn)
        def inner(* args, shape = Sout, ** kwargs):
            if _skip_kwargs: kwargs = {k : v for k, v in kwargs.items() if k in _argnames}
            
            _is_graph = ops.is_tensorflow_graph()
            if _is_graph and _is_class_method:
                function, args = getattr(args[0], fn.__name__), args[1:]
            else:
                function = fn

            if not _is_graph:
                if numpy:
                    args = [ops.convert_to_numpy(a) if ops.is_tensor(a) else a for a in args]
                    kwargs  = {
                        k : ops.convert_to_numpy(v) if ops.is_tensor(v) else v
                        for k, v in kwargs.items()
                    }
                return function(* args, ** kwargs)

            def fn_with_cast(* args, ** kwargs):
                out = function(* args, ** kwargs)
                if isinstance(out, (list, tuple)):
                    if not isinstance(Tout, list):  dtypes = [Tout] * len(out)
                    elif len(Tout) == 1:            dtypes = Tout * len(out)
                    else:                           dtypes = Tout
                    return [tf.cast(out_i, dtype) for out_i, dtype in zip(out, dtypes)]
                else:
                    return tf.cast(out, Tout if not isinstance(Tout, list) else Tout[0])

            import tensorflow as tf
            
            python_function = tf.py_function if not numpy else tf.numpy_function
            
            kwargs = {
                k : v for k, v in kwargs.items()
                if (v is not None) and (hasattr(v, 'shape') or v != _defaults.get(k, None))
            }
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
                
                if not isinstance(result, list):
                    result = tf.ensure_shape(
                        result, shape if not isinstance(shape, tuple) else shape[0]
                    )
                else:
                    if not isinstance(shape, tuple):    shape = tuple([shape] * len(result))
                    elif len(shape) == 1:               shape = shape * len(result)
                    
                    result = tuple(tf.ensure_shape(res, s) for res, s in zip(result, shape))
            elif isinstance(result, list):
                result = tuple(result)
            
            return result
        
        _params = inspect.signature(fn).parameters
        _defaults   = {k : v.default for k, v in _params.items() if v.default is not inspect._empty}
        _argnames   = list(_params.keys())
        _skip_kwargs    = 'kwargs' not in _argnames
        _is_class_method    = _argnames[0] == 'self'
        
        sign_with_shape = list(_params.values())
        if sign_with_shape[-1].kind == inspect.Parameter.VAR_KEYWORD:
            sign_with_shape.insert(-2, inspect.Parameter(
                name = 'shape', default = Sout, kind = inspect.Parameter.KEYWORD_ONLY
            ))
        else:
            sign_with_shape.append(inspect.Parameter(
                name = 'shape', default = Sout, kind = inspect.Parameter.KEYWORD_ONLY
            ))
        
        inner.Tout      = Tout
        inner.signature = signature
        inner.__signature__ = inspect.Signature(
            sign_with_shape, return_annotation = inspect.signature(fn).return_annotation
        )

        return inner
    
    assert Tout is not None or signature is not None

    Sout = None
    if signature is not None:
        Tout    = signature.dtype if not isinstance(signature, list) else [s.dtype for s in signature]
        Sout    = signature.shape if not isinstance(signature, list) else [s.shape for s in signature]
    elif isinstance(Tout, tuple):
        Tout    = list(Tout)
    
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

def compile_function(fn, jit_compile, force_tensorflow = False, ** kwargs):
    """
        Compile `fn` with the appropriate backend function
        - tensorflow    : `tf.function`
        - torch         : `torch.compile`
        - jax           : `jax.jit`
    """
    compile_fn = None
    if ops.is_tensorflow_backend() or force_tensorflow:
        compile_fn = sys.modules['tensorflow'].function
        kwargs['jit_compile'] = jit_compile
    elif not jit_compile:
        return fn
    elif ops.is_torch_backend():
        compile_fn = sys.modules['torch'].compile
    elif ops.is_jax_backend():
        compile_fn = sys.modules['jax'].jit
    
    kwargs = {
        k : v for k, v in kwargs.items() if k in inspect.signature(compile_fn).parameters
    }
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Compiling {} with kwargs {}'.format(fn, kwargs))
    
    return compile_fn(fn, ** kwargs)

def replace_kwargs(fn, nested_fn = None, ignore = ()):
    """
        Replaces `fn.__signature__` by removing the `kwargs` parameter, and replacing it by all new arguments defined in `nested_fn`
        In practice, this will **not** affect `fn`, and you will still have the `kwargs` variable. This will simply modify the signature so that the `graph_compile` method will assume that the function has no "kwargs", enabling to remove unknown kwargs (useful for XLA/graph modes)
        Note that it can also be used as a decorator
    """
    if nested_fn is None:
        return lambda wrapped_fn: replace_kwargs(wrapped_fn, fn, ignore = ignore)
        
    original_params, updated_params = list(zip(* inspect.signature(fn).parameters.items()))
    original_params, updated_params = original_params[:-1], list(updated_params)[:-1] # remove "kwargs"
    
    for name, param in inspect.signature(nested_fn).parameters.items():
        if name not in original_params and name not in ignore and param.default is not inspect._empty:
            updated_params.append(inspect.Parameter(
                name = name, default = param.default, kind = inspect.Parameter.KEYWORD_ONLY
            ))
    
    fn.__signature__ = inspect.Signature(updated_params)
    fn.__annotations__.update(_get_annotations(nested_fn))
    return fn

def _infer_execution_mode(run_eagerly, use_xla, prefer_xla, support_xla, force_tensorflow):
    if run_eagerly:
        return ExecutionMode.EAGER, ops.EagerExecution()
    elif ops.should_execute_eagerly() and run_eagerly is not False:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('The function is requested to run eagerly by another function')
        return ExecutionMode.EAGER, None

    if use_xla is None:
        use_xla = (prefer_xla) or (support_xla and not ops.is_tensorflow_backend())
    if use_xla and not support_xla:
        warnings.warn('`use_xla = True` but the function does not support XLA\nSet `support_xla = True` in the decorator if you want to enable it')
        use_xla = False

    if not use_xla and not (ops.is_tensorflow_backend() or force_tensorflow):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('The function should be executed in graph without XLA which is not supported by this backend')
        return ExecutionMode.EAGER, None
    
    ctx_manager = ops.XLAExecution(force_tensorflow = force_tensorflow)
    return (ExecutionMode.XLA if use_xla else ExecutionMode.GRAPH), ctx_manager

def _cast_arg(value, annot, force_tensorflow = False, mode = None):
    if value is None: return None
    
    if isinstance(annot, list):
        return [_cast_arg(v, a, force_tensorflow, mode) for v, a in zip(value, annot)]
    elif isinstance(annot, dict):
        return {
            k : _cast_arg(value[k], v, force_tensorflow, mode)
            for k, v in annot.items()
            if k in value
        }
    elif not isinstance(annot, TensorSpec):
        return value
    elif annot.static and mode == ExecutionMode.XLA:
        return value
    elif isinstance(value, tuple):
        return tuple(_cast_arg(v, annot, force_tensorflow, mode) for v in value)
    elif force_tensorflow:
        return ops.convert_to_tf_tensor(value, annot.dtype)
    else:
        return ops.convert_to_tensor(value, annot.dtype)

def _get_static_args(inputs):
    if inputs.arguments.get('kwargs', {}):
        return _get_static_args(
            {k : v for k, v in inputs.arguments.items() if k != 'kwargs'}
        ) +  _get_static_args(inputs.arguments['kwargs'])
    return [k for k, v in inputs.arguments.items() if not ops.is_tensor(v)]

def _call_from_bounded_args(function, inputs, kwargs = {}):
    if 'kwargs' in inputs.arguments:
        kwargs = {** kwargs, ** inputs.arguments.pop('kwargs')}
    
    if 'self' in inputs.arguments:
        return function(inputs.arguments.pop('self'), ** inputs.arguments, ** kwargs)
    
    return function(** inputs.arguments, ** kwargs)

def _get_annotations(fn):
    if hasattr(inspect, 'get_annotations'):
        return inspect.get_annotations(fn)
    else:
        return getattr(fn, '__annotations__', {})
