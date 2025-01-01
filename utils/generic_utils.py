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
import json
import uuid
import timeit
import logging
import argparse
import datetime
import numpy as np

from .parser import *
from .keras_utils import ops

logger = logging.getLogger(__name__)

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

def convert_to_str(x):
    """ Convert different string formats (bytes, tf.Tensor, ...) to regular `str` object """
    if isinstance(x, str) or x is None: return x
    elif hasattr(x, 'dtype') and getattr(x.dtype, 'name', None) == 'string':
        x = x.numpy()
    elif isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number): return x
    
    if isinstance(x, np.ndarray) and x.ndim == 0: x = x.item()
    if isinstance(x, bytes): return x.decode('utf-8')
    
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [convert_to_str(xi) for xi in x]
    elif isinstance(x, dict):
        return {convert_to_str(k) : convert_to_str(v) for k, v in x.items()}
    
    return x

def get_entry(data, keys):
    if isinstance(data, str):        return data
    elif not isinstance(data, dict): return None
    elif isinstance(keys, str):      return data.get(keys, None)
    for k in keys:
        if k in data: return data[k]
    return None

def normalize_keys(kwargs, key_alternatives):
    kwargs = kwargs.copy()
    for k, alternatives in key_alternatives.items():
        if k in kwargs: continue
        
        for ki in alternatives:
            if ki in kwargs:
                kwargs[k] = kwargs[ki]
                break
        
    return kwargs

def to_json(data):
    """ Converts a given data to json-serializable (if possible) """
    if data is None: return data
    if hasattr(data, '__dataclass_fields__'):
        return {k : to_json(v) for k, v in data.__dict__.items()}
    if isinstance(data, enum.Enum): data = data.value
    if ops.is_tensor(data):         data = ops.convert_to_numpy(data)
    if isinstance(data, bytes):     data = data.decode('utf-8')
    if isinstance(data, np.ndarray) and len(data.shape) == 0: data = data.item()
    
    if isinstance(data, bool): return data
    elif isinstance(data, uuid.UUID):            return str(data)
    elif isinstance(data, datetime.datetime):    return data.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(data, (float, np.floating)): return float(data)
    elif isinstance(data, (int, np.integer)):    return int(data)
    elif isinstance(data, (list, tuple, set, np.ndarray)):
        return [to_json(d) for d in data]
    elif isinstance(data, argparse.Namespace):
        return {k : to_json(v) for k, v in data.__dict__.items()}
    elif isinstance(data, dict):
        return {to_json(k) : to_json(v) for k, v in data.items()}
    elif isinstance(data, str):
        from .file_utils import is_path, path_to_unix
        return data if not is_path(data) else path_to_unix(data)
    elif hasattr(data, 'get_config'):
        return to_json(data.get_config())
    else:
        logger.warning("Unknown json data ({}) : {}".format(type(data), data))
        return str(data)

def var_from_str(v):
    """ Try to get the value interpreted as json-notation """
    if not isinstance(v, str): return v
    try:
        return json.loads(v)
    except:
        return v

def to_lower_keys(data):
    """ Returns the same dict with lowercased keys"""
    return {k.lower() : v for k, v in data.items() if k.lower() not in data}

def print_objects(objects, print_name = 'objects', _logger = logger):
    """ Displays the list of available objects (i.e. simply prints `objects.keys()` :D ) """
    _logger.info("Available {} : {}".format(print_name, sorted(list(objects.keys()))))

def get_object(objects,
               obj,
               * args,
               err  = True,
               types    = (type, ),
               print_name   = 'object',
               function_wrapper = None,
               ** kwargs
              ):
    """
        Get corresponding object based on a name (`obj`) and dict of object names with associated class / function to call (`objects`)
        
        Arguments : 
            - objects   : mapping (`dict`) of names with their associated class / function
            - obj       : the object to build (either a list, str or instance of `types`)
            - args / kwargs : the args and kwargs to pass to the object / function
            - print_name    : name for printing if object is not found
            - err   : whether to raise error if object is not available
            - types : expected return type
            - function_wrapper  : wrapper that takes the stored function as single argument
        Return : 
            - (list of) object instance(s) or function result(s)
    """
    if obj is None:
        return [get_object(
            objects, n, * args, print_name = print_name, err = err, types = types, ** kw
        ) for n, kw in kwargs.items()]
    
    elif isinstance(obj, (list, tuple)):
        return [get_object(
            objects, n, * args, print_name = print_name, err = err, types = types, ** kwargs
        ) for n in obj]
    
    elif isinstance(obj, dict):
        if 'class_name' not in obj:
            return [get_object(
                objects, n, * args, print_name = print_name,  err = err, types = types, ** kwargs
            ) for n, args in obj.items()]
        
        obj, args, kwargs = obj['class_name'], (), obj.get(
            'config', {k : v for k, v in obj.items() if k != 'class_name'}
        )
    
    if isinstance(obj, str):
        _lower_objects = to_lower_keys(objects)
        if obj in objects:
            obj = objects[obj]
        elif obj.lower() in _lower_objects:
            obj = _lower_objects[obj.lower()]
        elif ''.join([c for c in obj.lower() if c.isalnum()]) in _lower_objects:
            obj = _lower_objects[''.join([c for c in obj.lower() if c.isalnum()])]
        elif err:
            raise ValueError("{} is not available !\n  Accepted : {}\n  Got : {}".format(
                print_name, tuple(objects.keys()), obj
            ))
    elif types is not None and isinstance(obj, types) or callable(obj):
        pass
    elif err:
        raise ValueError("{} is not available !\n  Accepted : {}\n  Got : {}".format(
            print_name, tuple(objects.keys()), obj
        ))
    else:
        logger.warning("Unknown {} !\n  Accepted : {}\n  Got : {}".format(
            print_name, tuple(objects.keys()), obj
        ))
        return obj
    
    if is_object(obj):
        return obj
    elif not isinstance(obj, type) and function_wrapper is not None:
        return function_wrapper(obj, ** kwargs)
    return obj(* args, ** kwargs)

def get_enum_item(value, enum, upper_names = True):
    if isinstance(value, enum): return value
    if isinstance(value, str):
        if upper_names: value = value.upper()
        if not hasattr(enum, value):
            raise KeyError('{} is not a valid {} : {}'.format(value, enum.__name__, tuple(enum)))
        return getattr(enum, value)
    return enum(value)

def benchmark(f, inputs, number = 30, force_gpu_sync = True, display_memory = False):
    """
        Computes the average time to compute the result of `f` on multiple `inputs`
        
        Arguments :
            - f : the function to call
            - inputs    : list of inputs for `f`
            - number    : the number of times to apply `f` on each input
            - force_gpu_sync    : whether to sync gpu (useful for graph-mode calls)
            - display_memory    : whether to display the tensorflow memory stats
        Return :
            - times : list of average execution time for the different inputs
    """
    if isinstance(f, dict):
        return {
            name : benchmark(f_i, inputs, number, force_gpu_sync, display_memory)
            for name, f_i in f.items()
        }
    
    times = []
    for i, inp in enumerate(inputs):
        if display_memory: show_memory(message = 'Before round #{}'.format(i + 1))

        if not isinstance(inp, tuple): inp = (inp, )
        def _g():
            if force_gpu_sync: one = ops.ones(())
            f(* inp)
            if force_gpu_sync: one = ops.convert_to_numpy(one)
        
        _g() # warmup 
        t = timeit.timeit(_g, number = number)
        times.append(t * 1000. / number)
        
        if display_memory: show_memory(message = 'After round #{}'.format(i + 1))
        
    return times

    
