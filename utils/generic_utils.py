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
import timeit
import inspect
import logging
import datetime
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

def time_to_string(seconds):
    """ Returns a string representation of a time (given in seconds) """
    if seconds < 0.001: return '{} \u03BCs'.format(int(seconds * 1000000))
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
    elif isinstance(x, tf.Tensor):
        if x.dtype != tf.string: return x
        x = x.numpy()
    elif isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number): return x
    
    if isinstance(x, np.ndarray) and x.ndim == 0: x = x.item()
    if isinstance(x, bytes): x = x.decode('utf-8')
    
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [convert_to_str(xi) for xi in x]
    elif isinstance(x, dict):
        return {convert_to_str(k) : convert_to_str(v) for k, v in x.items()}
    
    return x

def to_json(data):
    """ Converts a given data to json-serializable (if possible) """
    if data is None: return data
    if isinstance(data, enum.Enum): data = data.value
    if isinstance(data, (tf.Tensor, tf.Variable)):  data = data.numpy()
    if isinstance(data, bytes): data = data.decode('utf-8')
    
    if isinstance(data, bool): return data
    elif isinstance(data, datetime.datetime):    return data.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(data, (float, np.floating)): return float(data)
    elif isinstance(data, (int, np.integer)):    return int(data)
    elif isinstance(data, (list, tuple, set, np.ndarray)):
        return [to_json(d) for d in data]
    elif isinstance(data, dict):
        return {to_json(k) : to_json(v) for k, v in data.items()}
    elif isinstance(data, str):
        from utils.file_utils import is_path, path_to_unix
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
    return {k.lower() : v for k, v in data.items()}

def get_args(fn):
    """ Returns a `list` of the positional arguments (even if they have default values) """
    return inspect.getfullargspec(fn).args

def get_kwargs(fn):
    """ Returns a `dict` containing the kwargs of `fn` """
    sign = inspect.getfullargspec(fn)
    
    kwargs = {}
    if sign.defaults:
        kwargs.update({
            k : v for k, v in zip(sign.args[- len(sign.defaults) :], sign.defaults)
        })
    
    if sign.kwonlydefaults:
        kwargs.update(sign.kwonlydefaults)
    
    return kwargs

def signature_to_str(fn):
    return '{}{}'.format(fn.__name__, str(inspect.signature(fn)))

def print_objects(objects, print_name = 'objects', _logger = logger):
    """ Displays the list of available objects (i.e. simply prints `objects.keys()` :D ) """
    _logger.info("Available {} : {}".format(print_name, sorted(list(objects.keys()))))

def get_object(objects, obj, * args, print_name = 'object', err = False, types = None, ** kwargs):
    """
        Get corresponding object based on a name (`obj`) and dict of object names with associated class / function to call (`objects`)
        
        Arguments : 
            - objects   : mapping (`dict`) of names with their associated class / function
            - obj       : the object to build (either a list, str or instance of `types`)
            - args / kwargs : the args and kwargs to pass to the object / function
            - print_name    : name for printing if object is not found
            - err   : whether to raise error if object is not available
            - types : expected return type
        Return : 
            - (list of) instance(s) or function results
    """
    if types is not None and isinstance(obj, types): return obj
    elif obj is None:
        return [get_object(
            objects, n, * args, print_name = print_name, err = err, types = types, ** kw
        ) for n, kw in kwargs.items()]
    
    elif isinstance(obj, (list, tuple)):
        return [get_object(
            objects, n, * args, print_name = print_name, err = err, types = types, ** kwargs
        ) for n in obj]
    
    elif isinstance(obj, dict):
        if 'class_name' in obj:
            return get_object(
                objects, obj['class_name'], * args, print_name = print_name, err = err, types = types, ** obj.get('config', {})
            )
        return [get_object(
            objects, n, * args, print_name = print_name,  err = err, types = types, ** kwargs
        ) for n, args in obj.items()]
    
    elif isinstance(obj, str) and obj.lower() in to_lower_keys(objects):
        return to_lower_keys(objects)[obj.lower()](* args, ** kwargs)
    elif err:
        raise ValueError("{} is not available !\n  Accepted : {}\n  Got :{}".format(
            print_name, tuple(objects.keys()), obj
        ))
    else:
        logger.warning("{} : `{}` is not available !".format(print_name, obj))
        return obj

def get_enum_item(value, enum, upper_names = True):
    if isinstance(value, enum): return value
    if isinstance(value, str):
        if upper_names: value = value.upper()
        if not hasattr(enum, value):
            raise KeyError('{} is not a valid {} : {}'.format(value, enum.__name__, tuple(enum)))
        return getattr(enum, value)
    return enum(value)
    
def should_predict(predicted,
                   data,
                   data_key = 'filename',
                   overwrite    = False,
                   timestamp    = -1,
                   required_keys    = []
                  ):
    """
        Returns whether `data` has already been predicted or not
        
        Arguments :
            - predicted : mapping (`dict`) of filename to a `dict` of information (the result)
            - data  : the data to predict
            - data_key  : the key to use if `data` is a `dict` / `pd.Series`
            - overwrite : whether to overwrite if `data` is already in `predicted`
            - timestamp : if provided, only overwrites if `predicted[data]['timestamp'] < timestamp`
            - required_keys : the keys that must be in `predicted[data]`
        Return :
            - should_predict    : whether the data should be predicted or not
        
        /!\ The keys in `predicted` should be in Unix style (i.e. with '/' instead of '\')
    """
    if isinstance(data, (dict, pd.Series)) and data_key in data: data = data[data_key]
    if not isinstance(data, str): return True
    
    from utils.file_utils import path_to_unix
    
    data = path_to_unix(data)
    if data in predicted and all(k in predicted[data] for k in required_keys):
        if not overwrite or (timestamp != -1 and timestamp <= predicted[data].get('timestamp', -1)):
            return False
    return True

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
            if force_gpu_sync: one = tf.ones(())
            f(* inp)
            if force_gpu_sync: one = one.numpy()
        
        _g() # warmup 
        t = timeit.timeit(_g, number = number)
        times.append(t * 1000. / number)
        
        if display_memory: show_memory(message = 'After round #{}'.format(i + 1))
        
    return times

    
def get_metric_names(obj, default_if_not_list = None):
    """ Returns the associated name for `obj` (e.g., `metric_names`, `loss_names`, `name`, ...) """
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

flatten = tf.nest.flatten

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
