
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

import os
import enum
import json
import queue
import timeit
import inspect
import logging
import datetime
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

_limited_memory = False

def _is_filename(data):
    if not isinstance(data, str): return False
    return (
        len(data) < 512 and len(data) > 1 and any(c.isalnum() for c in data) and os.path.exists(data)
    )
    
def time_to_string(seconds):
    """ return a string representation of a time (given in seconds) """
    if seconds < 0.0001: return '{} \u03BCs'.format(int(seconds * 1000000))
    if seconds < 1.: return '{} ms'.format(int(seconds * 1000))
    h = int(seconds // 3600)
    h = "" if h == 0 else "{}h ".format(h)
    m = int((seconds % 3600) // 60)
    m = "" if m == 0 else "{}min ".format(m)
    s = ((seconds % 3600) % 60)
    s = "{:.3f} sec".format(s) if m == "" and h == "" else "{}sec".format(int(s))
    return "{}{}{}".format(h, m, s)        

def convert_to_str(x):
    """ Convert different string formats (bytes, tf.Tensor, ...) to a `str` object """
    if isinstance(x, str): return x
    elif isinstance(x, tf.Tensor) and x.dtype != tf.string: return x
    elif isinstance(x, np.ndarray) and x.dtype in (np.int32, np.float32, np.int64, np.float64): return x
    if hasattr(x, 'numpy'): x = x.numpy()
    
    if isinstance(x, np.ndarray) and x.ndim == 0: x = str(x)
    if isinstance(x, bytes): x = x.decode('utf-8')
    
    if isinstance(x, (list, tuple, np.ndarray)):
        return [convert_to_str(xi) for xi in x]
    elif isinstance(x, dict):
        return {convert_to_str(k) : convert_to_str(v) for k, v in x.items()}
    
    return x

def create_iterator(generator, ** kwargs):
    if isinstance(generator, pd.DataFrame):
        def _df_iterator():
            for idx, row in generator.iterrows():
                yield row
        return _df_iterator()
    elif isinstance(generator, (queue.Queue, multiprocessing.queues.Queue)):
        def _queue_iterator():
            try:
                while True:
                    item = generator.get(** kwargs)
                    if item is not None:
                        yield item
            except queue.Empty:
                pass
        
        return _queue_iterator()
    elif callable(generator):
        return generator(** kwargs)
    return generator

def split_gpus(n, memory = 2048):
    gpus = tf.config.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu, [
                    tf.config.LogicalDeviceConfiguration(memory_limit = memory) for _ in range(n)
                ]
            )
    except RuntimeError as e:
        print(e)
    
    logger.info("# physical GPU : {}\n# logical GPU : {}".format(
        len(tf.config.list_physical_devices('GPU')),
        len(tf.config.list_logical_devices('GPU'))
    ))

def limit_gpu_memory(limit = 2048):
    """ Limit gpu memory for tensorflow """
    global _limited_memory
    if _limited_memory: return
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu, 
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2048)]
            )
        _limited_memory = True
    except:
        logger.error("Error while limiting tensorflow GPU memory")

def show_memory(gpu = 'GPU:0', message = ''):
    mem_usage = tf.config.experimental.get_memory_info(gpu)
    print('{}{}'.format(message if not message else message + '\t: ', {
        k : '{:.3f} Gb'.format(v / 1024 ** 3) for k, v in mem_usage.items()
    }))
    tf.config.experimental.reset_memory_stats(gpu)
    return mem_usage

def infer_downsampling_factor(model):
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

def benchmark(f, inputs, number = 30, force_gpu_sync = True, display_memory = False):
    if isinstance(f, dict):
        return {name : benchmark(f_i, inputs, number, force_gpu_sync, display_memory) for name, f_i in f.items()}
    times = []
    if display_memory: show_memory(message = 'Before')
    for inp in inputs:
        if not isinstance(inp, tuple): inp = (inp, )
        def _g():
            if force_gpu_sync: one = tf.ones(())
            f(* inp)
            if force_gpu_sync: one = one.numpy()
        
        _g() # warmup 
        t = timeit.timeit(_g, number = number)
        times.append(t * 1000. / number)
    if display_memory: show_memory(message = 'After')
    return times

def get_enum_item(value, enum, upper_names = True):
    if isinstance(value, enum): return value
    if isinstance(value, str):
        if upper_names: value = value.upper()
        if not hasattr(enum, value):
            raise KeyError('{} is not a valid {} : {}'.format(value, enum.__name__, tuple(enum)))
        return getattr(enum, value)
    return enum(value)
    
def get_object(available_objects, obj_name, * args,
               print_name = 'object', err = False, 
               allowed_type = None, ** kwargs):
    """
        Get corresponding object based on a name and dict of object names with associated class / function to call
        Arguments : 
            - available_objects : dict of objects names with their associated class / function
            - obj_name      : the objectto construct (either a list, str or instance of 'allowed_type')
            - args / kwargs : the args and kwargs to pass to the constructor
            - print_name    : name for printing if object was not found
            - err   : whether to raise error if object is not available or not
            - allowed_type  : expected return type
        Return : 
            - instance (or list of instance) of the object created
    """
    if allowed_type is not None and isinstance(obj_name, allowed_type):
        return obj_name
    elif obj_name is None:
        return [get_object(
            available_objects, n, *args, print_name = print_name, 
            err = err, allowed_type = allowed_type, ** kw
        ) for n, kw in kwargs.items()]
    
    elif isinstance(obj_name, (list, tuple)):
        return [get_object(
            available_objects, n, *args, print_name = print_name, 
            err = err, allowed_type = allowed_type, ** kwargs
        ) for n in obj_name]
    
    elif isinstance(obj_name, dict):
        return [get_object(
            available_objects, n, *args, print_name = print_name, 
            err = err, allowed_type = allowed_type, ** kwargs
        ) for n, args in obj_name.items()]
    
    elif isinstance(obj_name, str) and obj_name.lower() in to_lower_keys(available_objects):
        return to_lower_keys(available_objects)[obj_name.lower()](*args, **kwargs)
    else:
        if err:
            raise ValueError("{} is not available !\n  Accepted : {}\n  Got :{}".format(
                print_name, tuple(available_objects.keys()), obj_name
            ))
        else:
            logger.warning("{} : '{}' is not available !".format(print_name, obj_name))
        return obj_name

def print_objects(objects, print_name = 'objects'):
    print("Available {} : {}".format(print_name, sorted(list(objects.keys()))))
    
def to_lower_keys(dico):
    return {k.lower() : v for k, v in dico.items()}

def to_json(data):
    """ Convert a given data to json-serializable (if possible) """
    if isinstance(data, enum.Enum): data = data.value
    if isinstance(data, (tf.Tensor, tf.Variable)):  data = data.numpy()
    if isinstance(data, bytes): data = data.decode('utf-8')
    if isinstance(data, bool): return data
    elif isinstance(data, datetime.datetime):    return data.strftime("%Y-%m-%m %H:%M:%S")
    elif isinstance(data, (float, np.floating)): return float(data)
    elif isinstance(data, (int, np.integer)): return int(data)
    elif isinstance(data, (tuple, list, np.ndarray)):
        return [to_json(d) for d in data]
    elif isinstance(data, dict):
        return {to_json(k) : to_json(v) for k, v in data.items()}
    elif isinstance(data, str) and _is_filename(data):
        return data.replace(os.path.sep, '/')
    elif hasattr(data, 'get_config'):
        return to_json(data.get_config())
    elif data is None or isinstance(data, str):
        return data
    else:
        logger.warning("Unknown json data (type : {}) : {}".format(
            type(data), data
        ))
        return str(data)

def var_from_str(v):
    """ Try to get the value interpreted as json-notation """
    if not isinstance(v, str): return v
    try:
        v = json.loads(v)
    except:
        pass
    return v

    

def get_metric_names(obj, default_if_not_list = None):
    if isinstance(obj, dict):
        default_if_not_list = list(obj.keys())
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        if not isinstance(default_if_not_list, (list, tuple)):
            default_if_not_list = [default_if_not_list] * len(obj)
        return tf.nest.flatten(tf.nest.map_structure(
            get_metric_names, obj, default_if_not_list
        ))
    if hasattr(obj, 'metric_names'):
        return obj.metric_names
    elif hasattr(obj, 'loss_names'):
        return obj.loss_names
    elif default_if_not_list is not None:
        return default_if_not_list
    elif hasattr(obj, 'name'):
        return obj.name
    elif hasattr(obj, '__name__'):
        return obj.__name__
    elif hasattr(obj, '__class__'):
        return obj.__class__.__name__
    else:
        raise ValueError("Cannot extract name from {} !".format(obj))
    
def flatten(struct):
    """ Flatten nested python lists """
    return tf.nest.flatten(struct)

def unstack_and_flatten(struct):
    """
        Unstack nested 1D tensor and flatten all as a single list of scalar
        Useful to map nested metrics to their names
    """
    if isinstance(struct, tf.Tensor):
        return tf.unstack(tf.reshape(struct, [-1]))
    return tf.nest.flatten(
        tf.nest.map_structure(lambda t: tf.unstack(t) if tf.rank(t) > 0 else t, struct)
    )

def map_output_names(values, names):
    mapping = {}
    idx = 0
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

def get_kwargs(fn):
    sign = inspect.getfullargspec(fn)
    kwargs = {}
    if sign.defaults:
        kwargs.update({
            k : v for k, v in zip(sign.args[- len(sign.defaults) :], sign.defaults)
        })
    if sign.kwonlydefaults:
        kwargs.update(kwonlydefaults)
    return kwargs

def parse_args(* args, allow_abrev = True, add_unknown = False, ** kwargs):
    """
        Not tested yet but in theory it parses arguments :D
        Arguments : 
            - args  : the mandatory arguments
            - kwargs    : optional arguments with their default values
            - allow_abrev   : whether to allow abreviations or not (will automatically create abreviations as the 1st letter of the argument if it is the only argument to start with this letter)
    """
    def get_abrev(keys):
        abrev_count = {}
        for k in keys:
            abrev = k[0]
            abrev_count.setdefault(abrev, 0)
            abrev_count[abrev] += 1
        return [k for k, v in abrev_count.items() if v == 1 and k != 'h']
    
    parser = argparse.ArgumentParser()
    for arg in args:
        name, config = arg, {}
        if isinstance(arg, dict):
            name, config = arg.pop('name'), arg
        parser.add_argument(name, ** config)
    
    allowed_abrev = get_abrev(kwargs.keys()) if allow_abrev else {}
    for k, v in kwargs.items():
        abrev = k[0]
        names = ['--{}'.format(k)]
        if abrev in allowed_abrev: names += ['-{}'.format(abrev)]
        
        config = v if isinstance(v, dict) else {'default' : v}
        if not isinstance(v, dict) and v is not None: config['type'] = type(v)
        
        parser.add_argument(* names, ** config)
    
    parsed, unknown = parser.parse_known_args()
    
    parsed_args = {}
    for a in args + tuple(kwargs.keys()): parsed_args[a] = getattr(parsed, a)
    if add_unknown:
        k, v = None, None
        for a in unknown:
            if not a.startswith('--'):
                if k is None:
                    raise ValueError("Unknown argument without key !\n  Got : {}".format(unknown))
                a = var_from_str(a)
                if v is None: v = a
                elif not isinstance(v, list): v = [v, a]
                else: v.append(a)
            else: # startswith '--'
                if k is not None:
                    parsed_args.setdefault(k, v if v is not None else True)
                k, v = a[2:], None
        if k is not None:
            parsed_args.setdefault(k, v if v is not None else True)
    
    return parsed_args
