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

import inspect
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from functools import wraps

from utils.wrapper_utils import update_signature
from utils.generic_utils import get_args, get_kwargs, convert_to_str

logger  = logging.getLogger(__name__)

_limited_memory = False

def split_gpus(n, memory = 2048):
    """ Splits each physical GPU into `n` virtual devices with `memory` available gpu memory """
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(gpu, [
                tf.config.LogicalDeviceConfiguration(memory_limit = memory)
                for _ in range(n)
            ])
    except RuntimeError as e:
        logger.error(e)
    
    logger.info("# physical GPU : {}\n# logical GPU : {}".format(
        len(tf.config.list_physical_devices('GPU')), len(tf.config.list_logical_devices('GPU'))
    ))

def limit_gpu_memory(limit = 4096):
    """ Limits the tensorflow visible GPU memory on each available physical device """
    global _limited_memory
    if _limited_memory or not limit: return
    
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(gpu, [
                tf.config.LogicalDeviceConfiguration(memory_limit = limit)
            ])
        _limited_memory = True
    except Exception as e:
        logger.error("Error while limiting tensorflow GPU memory : {}".format(e))

def set_memory_growth(memory_growth = True):
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, memory_growth)
    except Exception as e:
        logger.error("Error while setting memory growth : {}".format(e))

def show_memory(gpu = 'GPU:0', message = ''):
    """ Displays the memory stats computed by `tensorflow`, then resets the stats """
    mem_usage = tf.config.experimental.get_memory_info(gpu)
    logger.info('{}{}'.format(message if not message else message + '\t: ', {
        k : '{:.3f} Gb'.format(v / 1024 ** 3) for k, v in mem_usage.items()
    }))
    tf.config.experimental.reset_memory_stats(gpu)
    return mem_usage

def convert_to_tensor(data):
    """
        Converts `data` ((list / dict) of `str` or `np.ndarray`) by inferring the output dtype :
            - bool      --> tf.bool
            - str       --> tf.string
            - uint8     --> tf.uint8    # useful for images
            - int / int32 / int64       --> tf.int32
            - float / float32 / float64 --> tf.float32
        This function uses the `tf.cast` with the inferred output type
        The other benefit is the standardization between types (e.g., floating --> tf.float32)
    """
    if isinstance(data, tf.Tensor) or data is None: return data
    if isinstance(data, (list, tuple, dict)):
        return tf.nest.map_structure(convert_to_tensor, data)
    
    if isinstance(data, str):                       conv_dtype = tf.string
    elif isinstance(data, int):                     conv_dtype = tf.int32
    elif isinstance(data, float):                   conv_dtype = tf.float32
    elif isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.floating):  conv_dtype = tf.float32
        elif data.dtype == bool:                    conv_dtype = tf.bool
        elif data.dtype == np.uint8:                conv_dtype = tf.uint8
        elif np.issubdtype(data.dtype, np.integer): conv_dtype = tf.int32
        else: raise ValueError('Unknown data dtype : {}'.format(data.dtype))
    else: raise ValueError('Unknown data type : {}\n{}'.format(type(data), data))
    
    return tf.cast(data, conv_dtype)

def execute_eagerly(fn  = None,
                    Tout    = None,
                    signature   = None,
                    default_key = None,
                    numpy   = False,
                    name    = None
                   ):
    """
        This function wraps `fn`, a regular python function, such that it can be used transparently inside a ``tf.function` executing by automatically calling `tf.py_function` or `tf.numpy_function`
        
        Arguments :
            - fn    : the python function to wrap
            - Tout  : output types of `fn`
            - signature : (list of) `tf.TensorSpec` that gives both shapes and types information
            - default_key   : the key to use if the 1st argument is a `dict / pd.Series`
            - numpy : whether to use `numpy_function` instead of `py_function`
            - name  : the operation name
        Return :
            If `fn is None`:
                - wraper    : a decorator function
            Else :
                - decorated : the function wrapped
        
        Note : if the function is executed eagerly (i.e. `tf.executing_eagerly() == True`), `fn` is simply called, meaning that the output is **not** forced to be a `tf.Tensor` !
        
        Note 2 : the benefit of passing `signature` instead of `Tout` is that it will fix the static shape of the output tensor with `tf.ensure_shape`, which may be required for some future usage
        
        Note 3 : when calling the decorated function, 2 new arguments are supported :
            - key       : to override the `default_key` argument
            - shape     : to specify a more precise shape (or in complement to `Tout`), see Note 2
        
        Known limitation : the decorator cannot be used on class method directly
        Example :
        ```python
        # This will raise an error when called in graph-mode due to the `self` parameter
        # which is not convertible to `tf.Tensor`, and thus not usable in `tf.numpy_function`
        class TextEncoderV1(object):
            @execute_eagerly(...)
            def encode(self, text, ...):
                ...
        
        # This will work properly both in eager and graph modes !
        # This way, the `self` argument is not *explicitely* passed, which makes `tf.numpy_function` happy :)
        class TextEncoderV2(object):
            def __init__(...):
                ...
                self.encode = execute_eagerly(self.encode, ...)
            
            def encode(self, text, ...):
                ...
        ```
    """
    def wrapper(fn):
        @wraps(fn)
        def inner(* args, shape = Sout, key = default_key, ** kwargs):
            if not tf.executing_eagerly() and is_class_method:
                function, args = getattr(args[0], fn.__name__), args[1:]
            else:
                function = fn
            
            if len(args) > 0 and isinstance(args[0], (dict, pd.Series)):
                if key and key in args[0]:
                    args = (args[0][key], ) + args[1:]
                elif default_key and default_key in args[0]:
                    args = (args[0][default_key], ) + args[1:]

            if tf.executing_eagerly(): return fn(* args, ** kwargs)

            if not kwargs:
                result = python_function(function, args, Tout = Tout)
            else:
                def fn_with_kwargs(n, * args_and_kwargs):
                    args    = args_and_kwargs[:n]
                    keys    = convert_to_str(args_and_kwargs[n])
                    kwargs  = {k : v for k, v in zip(keys, args_and_kwargs[n + 1 :])}
                    return function(* args, ** kwargs)

                keys    = list(kwargs.keys())
                vals    = [kwargs[k] for k in keys]
                args_with_kv = list(args) + [tf.cast(keys, tf.string)] + vals
                result = python_function(
                    fn_with_kwargs, [len(args)] + args_with_kv, Tout = Tout
                )

            if shape is not None:
                result = tf.nest.map_structure(tf.ensure_shape, result, shape)
            elif Sout is not None:
                result = tf.nest.map_structure(tf.ensure_shape, result, Sout)

            return result
        
        is_class_method     = 'self' == list(inspect.signature(fn).parameters.keys())[0]
        inner.__signature__ = update_signature(fn, shape = Sout, key = default_key)
        python_function     = tf.py_function if not numpy else tf.numpy_function

        inner.signature = signature
        inner.Tout      = Tout
        inner.default_key   = default_key
        
        return inner
    
    assert Tout is not None or signature is not None
    
    Sout = None
    if signature is not None:
        Tout    = tf.nest.map_structure(lambda s: s.dtype, signature)
        Sout    = tf.nest.map_structure(lambda s: s.shape, signature)
    
    return wrapper if fn is None else wrapper(fn)

def cast_arg(value, annot, force = False):
    if value is None: return None
    if isinstance(annot, tf.TensorSpec):
        value = tf.cast(value, annot.dtype)
        if annot.shape is not None: value = tf.ensure_shape(value, annot.shape)
        return value
    elif annot == tf.Tensor:
        return convert_to_tensor(value)
    elif force and not isinstance(value, (str, bool)) and not callable(value):
        return convert_to_tensor(value)
    return value

def tf_compile(fn = None,
               support_xla  = False,
               cast_kwargs  = False,
               cast_defaults    = True,
               follow_type_hints    = False,
               ** compile_kwargs
              ):
    """
        This function is equivalent to `tf.function` except that it pre-processes the arguments to cast them to `tf.Tensor` (only if `experimental_follow_type_hints == True`)
        If `experimental_follow_type_hints = False` it simply returns `tf.function` call
        
        Note that when the wrapped function is executing within another graph-function (i.e. `tf.executing_eagerly() == False`), no pre-processing is performed as the args / kwargs are by design `tf.Tensor`
        
        /!\ WARNING /!\ The output function is a regular function calling a graph function, and is **not** a graph function. It means that passing this function as argument to a `tf.function` will raise retracing (as if you pass any regular function as argument)
        In this specific case, either set `experimental_follow_type_hints = False`, either use `tf.function` directly
        Note that it is a really specific use case, and you may not care about this in the vast majority of usage ;)
        
        Example :
        ```python
            @tf_compile(experimental_follow_type_hints = True, reduce_retracing = True)
            def square(x : tf.Tensor):
                return x ** 2

            for i in range(5): print(square(i))
            # Is strictly equivalent to
            for i in range(5): print(square(tf.cast(i, tf.int32)))
        ```
        This feature has been removed after tensorflow 2.10 but it is useful to avoid retracing
        
        Another fancy feature which was not implemented originally is that you can specify expected shapes and types by setting `tf.TensorSpec` as annotation
        This will automatically cast the input to the right dtype, which may be useful to standardize the inputs and further reduce retracing !
        It is a bit equivalent to a partially known `input_signature` in the `tf.function`
        ```python
            @tf_compile(experimental_follow_type_hints = True, reduce_retracing = True)
            def square(x : tf.TensorSpec(shape = None, dtype = tf.float32)):
                return x ** 2

            for i in range(5): print(square(i))
            # Is strictly equivalent to
            for i in range(5): print(square(tf.cast(i, tf.float32)))
        ```
    """
    def wrapper(fn):
        @wraps(fn)
        def inner(* args, run_eagerly = False, use_xla = None, recompile = False, ** kwargs):
            if run_eagerly or not tf.executing_eagerly(): return fn(* args, ** kwargs)
            
            if follow_type_hints and fn_annots:
                args = tuple([
                    cast_arg(arg, fn_annots.get(name, None))
                    for name, arg in zip(fn_args[: len(args)], args)
                ])
                kwargs  = {
                    k : cast_arg(v, fn_annots.get(k, None), cast_kwargs)
                    for k, v in kwargs.items()
                }
            if cast_defaults:
                for k, annot in defaults_to_cast.items():
                    if k not in fn_args[:len(args)] and k not in kwargs:
                        kwargs[k] = cast_arg(fn_kwargs[k], annot)
            
            if use_xla is None: use_xla = support_xla
            if use_xla and not support_xla:
                warnings.warn('`use_xla = True` while the function {} does not support XLA\nSet `support_xla = True` in the decorator if you want to enable it'.format(fn.__name__))
                use_xla = False
            
            if not use_xla:
                if recompile or 'graph' not in _compiled:
                    _compiled['graph'] = tf.function(fn, ** compile_kwargs)
                return _compiled['graph'](* args, ** kwargs)
            else:
                if recompile or 'xla' not in _compiled:
                    _compiled['xla'] = tf.function(fn, jit_compile = True, ** compile_kwargs)
                return _compiled['xla'](* args, ** kwargs)
        
        _compiled   = {}
        
        fn_args     = get_args(fn)
        fn_kwargs   = get_kwargs(fn)
        if hasattr(inspect, 'get_annotations'):
            fn_annots   = inspect.get_annotations(fn)
        else:
            fn_annots   = fn.__annotations__
        fn_annots = {
            k : v for k, v in fn_annots.items()
            if isinstance(v, tf.TensorSpec) or v is tf.Tensor
        }
        defaults_to_cast    = {
            k : v for k, v in fn_annots.items() if fn_kwargs.get(k, None) is not None
        }
        
        return inner
    
    follow_type_hints = compile_kwargs.pop('experimental_follow_type_hints', follow_type_hints)
    return wrapper if fn is None else wrapper(fn)

def unstack_and_flatten(struct):
    """
        Unstack nested 1D tensor and flatten them all to a single list of scalar Tensors
        Useful to map nested metrics to their names
    """
    if isinstance(struct, tf.Tensor):
        return tf.unstack(tf.reshape(struct, [-1]))
    return tf.nest.flatten(
        tf.nest.map_structure(lambda t: tf.unstack(t) if tf.rank(t) > 0 else t, struct)
    )

def map_output_names(values, names):
    mapping, idx = {}, 0
    for i, v in enumerate(values):
        if isinstance(v, tf.Tensor):
            if len(tf.shape(v)) == 0:
                v = {names[idx] : v}
            else:
                v = tf.reshape(v, [-1])
                v = {n : vi for n, vi in zip(names[idx : idx + len(v)], tf.unstack(v))}
        idx += len(v)
        mapping.update(v)

    return mapping

def infer_downsampling_factor(model):
    """ Based on a sequential model, computes an estimation of the downsampling factor """
    from tensorflow.keras.layers import (
        Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, MaxPooling3D,
        AveragePooling1D, AveragePooling2D, AveragePooling3D
    )
    _downsampling_types = [
        Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, MaxPooling3D,
        AveragePooling1D, AveragePooling2D, AveragePooling3D
    ]
    try:
        from custom_layers import MaskedConv1D, MaskedMaxPooling1D, MaskedAveragePooling1D
        _downsampling_types.extend([MaskedConv1D, MaskedMaxPooling1D, MaskedAveragePooling1D])
    except Exception as e:
        pass
    
    def _get_factor(model):
        factor = 1
        for l in model.layers:
            if type(l) in _downsampling_types:
                factor = factor * np.array(l.strides)
            elif hasattr(l, 'layers'):
                factor = factor * _get_factor(l)
        
        return factor
    return _get_factor(model)

def infer_upsampling_factor(model):
    """ Based on a sequential model, computes an estimation of the upsampling factor """
    from tensorflow.keras.layers import (
        UpSampling1D, UpSampling2D, UpSampling3D,
        Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
    )
    _downsampling_types = [
        UpSampling1D, UpSampling2D, UpSampling3D,
        Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
    ]
    try:
        from custom_architectures.unet_arch import UpSampling2DV1
        _downsampling_types.append(UpSampling2DV1)
    except Exception as e:
        pass
    
    def _get_factor(model):
        factor = 1
        for l in model.layers:
            if type(l) in _downsampling_types:
                if hasattr(l, 'strides'):
                    strides = l.strides
                elif hasattr(l, 'size'):
                    strides = l.size
                elif hasattr(l, 'scale_factor'):
                    strides = l.scale_factor
                factor = factor * np.array(strides)
            elif hasattr(l, 'layers'):
                factor = factor * _get_factor(l)
        
        return factor
    return _get_factor(model)
