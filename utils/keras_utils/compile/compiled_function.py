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

import enum
import keras
import inspect
import logging
import warnings

from keras import tree
from typing import Union, Tuple
from dataclasses import dataclass
from functools import cached_property, cache, wraps

from .. import ops
from loggers import timer
from .exporter import export_function

logger  = logging.getLogger(__name__)

_jit_compile    = not ops.is_tensorflow_backend()

def jit_compile():
    global _jit_compile
    return _jit_compile

def set_jit_compile(jit_compile):
    global _jit_compile
    _jit_compile = jit_compile

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

class CompiledFunction:
    def __init__(self,
                 fn,
                 
                 cast_kwargs    = True,
                 cast_defaults  = True,
                 follow_type_hints  = True,
                 eager_convertion   = True,
                 internal_functions = None,
                 add_unspecified_kwargs = True,
                 
                 support_xla    = True,
                 prefer_xla     = None,
                 force_tensorflow   = False,
                 xla_recompile_on_new   = False,
                 
                 prepare    = None,
                 prepare_for_xla    = None,
                 prepare_for_graph  = None,
                 
                 static_args    = 'auto',
                 input_signature    = None,
                 reduce_retracing   = True,
                 ** compile_kwargs
                ):
        if isinstance(input_signature, TensorSpec): input_signature = [input_signature]
        
        self.fn = fn
        self.name   = fn.__name__ if hasattr(fn, '__name__') else fn.__class__.__name__
        self.sign_fn    = fn.call if isinstance(fn, keras.Model) else fn
        self.internal_functions = internal_functions
        
        self.cast_kwargs    = cast_kwargs
        self.cast_defaults  = cast_defaults
        self.follow_type_hints  = follow_type_hints
        self.eager_convertion   = eager_convertion
        self.add_unspecified_kwargs = add_unspecified_kwargs
        
        self.support_xla    = support_xla
        self.prefer_xla     = prefer_xla
        self.force_tensorflow   = force_tensorflow
        self.xla_recompile_on_new   = xla_recompile_on_new
        
        self.prepare    = prepare
        self.prepare_for_xla    = prepare_for_xla
        self.prepare_for_graph  = prepare_for_graph
        
        self.static_args    = static_args
        self.input_signature    = input_signature
        self.reduce_retracing   = reduce_retracing
        self.compile_kwargs = compile_kwargs
        self.compile_kwargs.update({
            'reduce_retracing'  : reduce_retracing
        })
        
        if self.args and self.args[0] == 'self':
            raise RuntimeError('The `fn` argument must be a bound method, got a class method. See `help(graph_compile)` to see the expected design pattern for this situation\n  Function : {}'.format(self.fn))

        self._compiled  = {}
        
        self._functions = None
        self.dispatcher = None
        self.dispatch_arg   = None
        self.dispatch_default   = None
        
        self.__wrapped__ = self.fn
    
    def _init_internal_functions(self):
        functions = [self.sign_fn]
        if self.internal_functions is not None:
            if isinstance(self.internal_functions, CompiledFunction):
                self.internal_functions = self.internal_functions.functions
            elif callable(self.internal_functions) and 'lambda' in self.internal_functions.__name__:
                self.internal_functions = self.internal_functions()
            if not isinstance(self.internal_functions, (list, tuple)):
                self.internal_functions = [self.internal_functions]
            
            for fn in self.internal_functions:
                if hasattr(fn, 'dispatcher') and getattr(fn, 'methods', None) is not None:
                    self.dispatch_arg   = fn.dispatcher
                    self.dispatch_default   = get_kwargs(fn).get(self.dispatch_arg, None)
                    self.dispatcher = fn.methods
                else:
                    functions.append(fn)
    
        return functions

    @property
    def functions(self):
        if not self._functions:
            self._functions = self._init_internal_functions()
        return self._functions
    
    @cached_property
    def args(self):
        return get_args(self.sign_fn)
    
    @cached_property
    def kwargs(self):
        kwargs = {}
        for fn in reversed(self.functions):
            if isinstance(fn, CompiledFunction):
                kwargs.update(fn.kwargs)
            else:
                kwargs.update(get_kwargs(fn))
        return kwargs
    
    @cached_property
    def annotations(self):
        annots = {}
        for fn in reversed(self.functions):
            while is_timed_fn(fn):
                fn = fn.__wrapped__

            if isinstance(fn, CompiledFunction):
                annots.update(fn.annotations)
            else:
                annots.update(get_tensor_annots(fn, self.force_tensorflow))
        
        if self.input_signature:
            if isinstance(self.input_signature, dict):
                annots.update(self.input_signature)
            elif isinstance(self.input_signature, (list, tuple)):
                annots.update({
                    name : sign for name, sign in zip(self.args, self.input_signature)
                })
            else:
                raise ValueError('Unsupported `input_signature` : {}'.format(self.input_signature))
        
        return annots
    
    @cached_property
    def arg_names(self):
        return set(self.args + list(self.kwargs.keys()))
    
    @cached_property
    def supports_kwargs(self):
        return len(self.functions) == 1 and has_kwargs(self.sign_fn)
    
    @cached_property
    def prepare_fn_kwargs(self):
        kwargs = {}
        if self.prepare is not None:           kwargs.update(get_kwargs(self.prepare))
        if self.prepare_for_xla is not None:   kwargs.update(get_kwargs(self.prepare_for_xla))
        if self.prepare_for_graph is not None: kwargs.update(get_kwargs(self.prepare_for_graph))
        return {k : v for k, v in kwargs.items() if k not in self.arg_names}
    
    def _get_dispatch_method(self, args_dict, kwargs):
        if self.dispatch_arg in args_dict:  return args_dict[self.dispatch_arg]
        elif self.dispatch_arg in kwargs:   return kwargs[self.dispatch_arg]
        else:   return self.dispatch_default
    
    def _get_dispatched_args(self, method):
        return set(self._get_dispatched_kwargs(method).keys())
    
    @cache
    def _get_dispatched_kwargs(self, method):
        if not self.dispatcher: return {}
        return get_kwargs(self.dispatcher[method])
    
    @cache
    def _get_dispatched_annotations(self, method):
        if not self.dispatcher: return {}
        return get_tensor_annots(self.dispatcher[method], self.force_tensorflow)
    
    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, signature_to_str(self.sign_fn))
    
    def __call__(self, * args, run_eagerly = None, use_xla = None, recompile = False, ** kwargs):
        dispatch_method, prep_kwargs, inputs = self.prepare_inputs(args, kwargs)
        
        if self.prepare is not None: inputs = self.prepare(** inputs, ** prep_kwargs)
        
        if self.force_tensorflow and not ops.is_tensorflow_available():
            warnings.warn(
                'Tensorflow is not available, running the function {} eagerly'.format(fn_name)
            )
            return self.fn(** inputs)
        elif not ops.executing_eagerly():
            return self.fn(** inputs)
        
        execution_mode, ctx_manager = infer_execution_mode(
            run_eagerly = run_eagerly,

            use_xla = use_xla,
            prefer_xla  = self.prefer_xla,
            support_xla = self.support_xla,

            force_tensorflow    = self.force_tensorflow
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Unprocessed kwargs : {}'.format(_inp_to_str(inputs)))
        
        if execution_mode != ExecutionMode.EAGER or self.eager_convertion:
            if self.follow_type_hints:
                inputs = self.cast_known_kwargs(dispatch_method, execution_mode, inputs)

            if self.cast_kwargs and self.supports_kwargs:
                inputs = self.cast_additional_kwargs(dispatch_method, execution_mode, inputs)

            if self.cast_defaults:
                inputs = self.cast_default_kwargs(dispatch_method, execution_mode, inputs)
        
        if execution_mode == ExecutionMode.EAGER:
            key = mode = 'eager'
            if key not in self._compiled: self._compiled[key] = self.fn
        else:
            _compile_kwargs = self.compile_kwargs.copy()
            
            if self.add_unspecified_kwargs:
                for k, v in self.kwargs.items():
                    if k not in inputs: inputs[k] = v
            
            if self.prepare_for_graph is not None:
                inputs = self.prepare_for_graph(** inputs, ** prep_kwargs)
            
            if execution_mode == ExecutionMode.GRAPH:
                key = mode = 'graph'
            else:
                mode = 'xla'
                key = 'xla' if not self.xla_recompile_on_new else _inp_to_str(inputs)
                if self.prepare_for_xla is not None:
                    inputs = self.prepare_for_xla(** inputs, ** prep_kwargs)

                if (ops.is_jax_backend()
                    and not self.force_tensorflow
                    and self.static_args == 'auto'):
                    
                    _compile_kwargs['static_args'] = self.get_static_args(inputs)

            if recompile or key not in self._compiled:
                self._compiled[key] = timer(compile_function(
                    self.fn, jit_compile = execution_mode == ExecutionMode.XLA, ** _compile_kwargs
                ), name = '{}_{}'.format(key, self.name), debug = True)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('The function {} is executed in {} mode with {}'.format(
                self.name, mode, _inp_to_str(inputs)
            ))

        if ctx_manager is not None:
            with ctx_manager: return self._compiled[key](** inputs)
        return self._compiled[key](** inputs)
    
    @timer(debug = True)
    def prepare_inputs(self, args, kwargs):
        args_dict   = {name : arg for name, arg in zip(self.args, args)}

        prep_kwargs = {k : v for k, v in kwargs.items() if k in self.prepare_fn_kwargs}
        
        dispatch_method = None
        if not self.supports_kwargs:
            supported = self.arg_names
            if self.dispatch_arg:
                dispatch_method = self._get_dispatch_method(args_dict, kwargs)
                supported = supported.copy()
                supported.update(self._get_dispatched_args(dispatch_method))
            
            unsupported = set(kwargs.keys()) - supported
            if unsupported:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Removing kwargs {}'.format(unsupported))
                for k in unsupported: kwargs.pop(k)
        
        kwargs.update(args_dict)
        return dispatch_method, prep_kwargs, kwargs
    
    @timer(name = 'follow_type_hints', debug = True)
    def cast_known_kwargs(self, method, execution_mode, kwargs):
        annots = self.annotations
        if self.dispatch_arg:
            annots.update(self._get_dispatched_annotations(method))
        
        for k, v in kwargs.items():
            if k in annots:
                kwargs[k] = cast_arg(v, annots[k], self.force_tensorflow, execution_mode)
        return kwargs

    @timer(debug = True)
    def cast_additional_kwargs(self, method, execution_mode, kwargs):
        for k, v in kwargs.items():
            if k not in self.arg_names:
                kwargs[k] = cast_arg(v, None, self.force_tensorflow)
        return kwargs
    
    @timer(debug = True)
    def cast_default_kwargs(self, method, execution_mode, kwargs):
        _kwargs, _annots = self.kwargs, self.annotations
        if self.dispatch_arg:
            _kwargs.update(self._get_dispatched_kwargs(method))
            _annots.update(self._get_dispatched_annotations(method))
        
        casted_defaults = {
            k : cast_arg(v, _annots[k], self.force_tensorflow, execution_mode)
            for k, v in _kwargs.items()
            if v is not None and k in _annots and k not in kwargs
        }

        if casted_defaults:
            kwargs.update(casted_defaults)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Add casted values for keys : {}'.format(
                    tuple(casted_defaults.keys())
                ))
        
        return kwargs

    def get_static_args(self, kwargs):
        return [k for k, v in kwargs.items() if not ops.is_tensor(v)]
    
    def export(self, directory, endpoints = None, ** kwargs):
        if not endpoints: endpoints = {}
        
        for name, signature in endpoints.items():
            if not isinstance(signature, dict): continue
            
            endpoints[name] = {k : v for k, v in signature.items() if k in self.arg_names}
        
        if len(endpoints) == 0:
            endpoints['serve'] = {
                k : v for k, v in self.annotations.items()
                if self.kwargs.get(k, -1) is not None
            }
        
        return export_function(self.fn, directory, endpoints = endpoints, ** kwargs)


def get_tensor_annots(fn, force_tensorflow):
    _is_tensorflow = force_tensorflow or ops.is_tensorflow_backend()
    return {
        k : v for k, v in get_annotations(fn).items() if _is_tensor_spec(v, _is_tensorflow)
    }

def _is_tensor_spec(annot, _is_tensorflow):
    if isinstance(annot, TensorSpec):
        return _is_tensorflow or not annot.static
    elif isinstance(annot, (list, tuple)):
        return all(_is_tensor_spec(v, _is_tensorflow) for v in annot)
    elif isinstance(annot, dict):
        return all(_is_tensor_spec(v, _is_tensorflow) for v in annot.values())
    else:
        return False
    

def infer_execution_mode(run_eagerly, use_xla, prefer_xla, support_xla, force_tensorflow):
    if run_eagerly:
        return ExecutionMode.EAGER, ops.EagerExecution()
    elif ops.should_execute_eagerly() and run_eagerly is not False:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('The function is requested to run eagerly by another function')
        return ExecutionMode.EAGER, None

    if use_xla is None:
        use_xla = (support_xla and jit_compile()) or prefer_xla
    elif use_xla and not support_xla:
        warnings.warn('`use_xla = True` but the function {} does not support XLA\nSet `support_xla = True` in the decorator if you want to enable it'.format(fn))
        use_xla = False

    if not use_xla and not (ops.is_tensorflow_backend() or force_tensorflow):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('The function should be executed in graph without XLA which is not supported by this backend')
        return ExecutionMode.EAGER, None
    
    ctx_manager = ops.XLAExecution(force_tensorflow = force_tensorflow)
    return (ExecutionMode.XLA if use_xla else ExecutionMode.GRAPH), ctx_manager
    
@cache
def cached_cast_arg(value, annot, force_tensorflow = False, mode = None):
    return cast_arg(value, annot, force_tensorflow, mode, cache = False)

def cast_arg(value, annot, force_tensorflow = False, mode = None, *, cache = True):
    if value is None: return None
    
    if isinstance(annot, list):
        return [
            cast_arg(v, a, force_tensorflow, mode, cache = cache)
            for v, a in zip(value, annot)
        ]
    elif isinstance(annot, dict):
        return {
            k : cast_arg(value[k], v, force_tensorflow, mode, cache = cache)
            for k, v in annot.items()
        }
    elif isinstance(annot, TensorSpec) and annot.static and mode == ExecutionMode.XLA:
        return value
    elif cache and isinstance(value, (int, float, bool)):
        return cached_cast_arg(value, annot, force_tensorflow, mode)
    
    if not force_tensorflow:
        convert_to_tensor = ops.convert_to_tensor
    else:
        convert_to_tensor = ops.convert_to_tf_tensor
    
    if isinstance(annot, TensorSpec):
        return convert_to_tensor(value, annot.dtype)
    elif _should_cast_kwarg(value):
        return convert_to_tensor(value)
    return value

def compile_function(fn, jit_compile, force_tensorflow = False, ** kwargs):
    compile_fn = None
    if ops.is_tensorflow_backend() or force_tensorflow:
        import tensorflow as tf
        compile_fn = tf.function
        kwargs['jit_compile'] = jit_compile
    elif ops.is_torch_backend() and jit_compile:
        import torch
        compile_fn = torch.compile
    elif ops.is_jax_backend() and jit_compile:
        import jax
        compile_fn = jax.jit
    
    if compile_fn is None: return fn
    
    kwargs = {
        k : v for k, v in kwargs.items() if k in inspect.signature(compile_fn).parameters
    }
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Compiling {} with kwargs {}'.format(fn, kwargs))
    return compile_fn(fn, ** kwargs)


def _should_cast_kwarg(x):
    if isinstance(x, dict): return all(_should_cast_kwarg(vi) for vi in x.values())
    if isinstance(x, list): return all(_should_cast_kwarg(xi) for xi in x)
    return not isinstance(x, (str, bool)) and not callable(x)

def is_timed_fn(fn):
    return getattr(fn, '__name__', None) == 'fn_with_timer'

def get_args(fn, ** kwargs):
    """ Returns a `list` of the positional argument names (even if they have default values) """
    return [
        name for name, param in inspect.signature(fn, ** kwargs).parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    
def get_kwargs(fn, ** kwargs):
    """ Returns a `dict` containing the kwargs of `fn` """
    return {
        name : param.default for name, param in inspect.signature(fn, ** kwargs).parameters.items()
        if param.default is not inspect._empty
    }

def get_annotations(fn):
    if hasattr(inspect, 'get_annotations'):
        return inspect.get_annotations(fn)
    elif hasattr(fn, '__annotations__'):
        return fn.__annotations__
    else:
        return {}

def has_kwargs(fn, ** kwargs):
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD and name == 'kwargs'
        for name, param in inspect.signature(fn, ** kwargs).parameters.items()
    )

def signature_to_str(fn, add_doc = False, ** kwargs):
    return '{}{}{}'.format(
        fn.__name__,
        str(inspect.signature(fn, ** kwargs)),
        '\n{}'.format(fn.__doc__) if add_doc else ''
    )

def _inp_to_str(kwargs):
    if isinstance(kwargs, dict):
        return {k : _inp_to_str(v) for k, v in kwargs.items()}
    elif isinstance(kwargs, (list, tuple)):
        return [_inp_to_str(v) for v in kwargs]
    elif not hasattr(kwargs, 'shape') or len(kwargs.shape) == 0:
        return kwargs
    return kwargs.shape
