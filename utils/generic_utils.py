import os
import json
import datetime
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

_limited_memory = False

def time_to_string(secondes):
    """ return a string representation of a time (given in seconds) """
    h = int(secondes // 3600)
    h = "" if h == 0 else "{}h ".format(h)
    m = int((secondes % 3600) // 60)
    m = "" if m == 0 else "{}min ".format(m)
    s = ((secondes % 300) % 60)
    s = "{:.3f} sec".format(s) if m == "" and h == "" else "{}sec".format(int(s))
    return "{}{}{}".format(h, m, s)        

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
        print("[ERROR] while limiting tensorflow GPU memory")

def get_object(available_objects, obj_name, *args,
               print_name = 'object', err = False, 
               allowed_type = None, **kwargs):
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
        return [get_object(available_objects, n, *args, print_name = print_name, 
                           err = err, allowed_type = allowed_type, **kw) 
                for n, kw in kwargs.items()]
    elif isinstance(obj_name, (list, tuple)):
        return [get_object(available_objects, n, *args, print_name = print_name, 
                           err = err, allowed_type = allowed_type, **kwargs) 
                for n in obj_name]
    elif isinstance(obj_name, dict):
        return [get_object(available_objects, n, *args, print_name = print_name, 
                           err = err, allowed_type = allowed_type, **kwargs) 
                for n, args in obj_name.items()]
    elif isinstance(obj_name, str) and obj_name.lower() in to_lower_keys(available_objects):
        return to_lower_keys(available_objects)[obj_name.lower()](*args, **kwargs)
    else:
        if err:
            raise ValueError("{} is not available !\n  Accepted : {}\n  Got :{}".format(print_name, tuple(available_objects.keys()), obj_name))
        else:
            print("[WARNING]\t{} : '{}' is not available !".format(print_name, obj_name))
        return obj_name

def print_objects(objects, print_name = 'objects'):
    print("Available {} : {}".format(print_name, sorted(list(objects.keys()))))
    
def to_lower_keys(dico):
    return {k.lower() : v for k, v in dico.items()}

def to_json(data):
    """ Convert a given data to json-serializable (if possible) """
    if isinstance(data, tf.Tensor): data = data.numpy()
    if isinstance(data, bytes): data = data.decode('utf-8')
    if isinstance(data, bool): return data
    elif isinstance(data, datetime.datetime): return data.strftime("%Y-%m-%m %H:%M:%S")
    elif isinstance(data, (float, np.floating)): return float(data)
    elif isinstance(data, (int, np.integer)): return int(data)
    elif isinstance(data, (tuple, list, np.ndarray)):
        return [to_json(d) for d in data]
    elif isinstance(data, dict):
        return {to_json(k) : to_json(v) for k, v in data.items()}
    elif isinstance(data, str) and len(data) < 512 and os.path.exists(data):
        return data.replace(os.path.sep, '/')
    elif hasattr(data, 'get_config'):
        return to_json(data.get_config())
    elif data is None or isinstance(data, str):
        return data
    else:
        print("Type inconnu : {}".format(type(data)))
        return str(data)

def var_from_str(v):
    """ Try to get the value interpreted as json-notation """
    try:
        v = json.loads(v)
    except:
        pass
    return v

    
def load_json(filename, default = {}):
    """ Safely load data from a json file """
    if not os.path.exists(filename): return default
    with open(filename, 'r', encoding = 'utf-8') as file:
        result = file.read()
    return json.loads(result)

def dump_json(filename, data, ** kwargs):
    """ Safely save data to a json file """
    data = to_json(data)
    data = json.dumps(data, ** kwargs)
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.write(data)

def normalize_filename(filename, invalid_mode = 'error'):
    """
        Return (list of) str filenames extracted from multiple formats
        
        Arguments :
            - filename  : (list of) filenames of different types
                - str   : return filename or list of filenames (if directory)
                - pd.DataFrame  : must have a 'filename' column
                - dict      : must have a 'filename' entry
                - tf.Tensor / np.ndarray    : string / bytes
    """
    if isinstance(filename, (dict, pd.Series)): filename = filename['filename']
    if isinstance(filename, tf.Tensor): filename = filename.numpy()
    if isinstance(filename, bytes): filename = filename.decode('utf-8')
    
    if isinstance(filename, (list, tuple)):
        outputs = flatten([normalize_filename(f) for f in filename])
    elif isinstance(filename, pd.DataFrame):
        outputs = flatten([normalize_filename(row) for _, row in filename.iterrows()])
    elif isinstance(filename, str):
        if not os.path.isdir(filename): return filename
        outputs = flatten([normalize_filename(
            os.path.join(filename, f)
        ) for f in os.listdir(filename)])
    else:
        if invalid_mode == 'skip': return None
        if invalid_mode == 'keep' : return filename
        else:
            raise ValueError("Unknown type for `filename` : {}\n  {}".format(type(filename), filename))
        
    return [o for o in outputs if o is not None]

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
    flattened_values    = unstack_and_flatten(values)
    flattened_names     = tf.nest.flatten(names)
    
    if len(flattened_values) != len(flattened_names):
        raise ValueError("Try to associate {} values with {} names !\n  Values : {}\n  Names : {}".format(len(flattened_values), len(flattened_names), flattened_values, flattened_names))
    return {n : v for n, v in zip(flattened_names, flattened_values)}

def pad_batch(batch, pad_value = 0, max_length = None, dtype = None):
    """
        Create a padded version of batch in a single np.ndarray
        Note that this function allows to have different shapes on different dimensions and will pad all of them. 
        However, all data must have the same rank
        
        Arguments : 
            - batch         : list of np.ndarray / tf.Tensor
            - pad_value     : the value to add as padding
            - max_length    : maximum length for each dimension. If not given, take the max length of datas 
            - dtype : dtype of the final output
        Return : 
            - padded_batch : np.ndarray of same rank as data
    """
    if not hasattr(batch[0], 'shape'): return np.array(batch)
    
    if dtype is None:
        b0 = batch[0] if not hasattr(batch[0], 'numpy') else batch[0].numpy()
        dtype = b0.dtype
    
    max_shape = batch[0].shape
    for b in batch:
        max_shape = [max(max_s, s) for max_s, s in zip(max_shape, b.shape)]
    if max_length is not None: max_shape[0] = min(max_shape[0], max_length)
    length = max_shape[0]
    max_shape = [len(batch)] + max_shape
    
    padded_batch = np.zeros(max_shape, dtype = dtype) + pad_value
    
    for i, b in enumerate(batch):
        if b.ndim == 1:
            padded_batch[i, :min(length, len(b))] = b[:length]
        elif b.ndim == 2:
            padded_batch[i, :min(length, len(b)), : b.shape[1]] = b[:length]
        elif b.ndim == 3:
            padded_batch[i, :min(length, len(b)), : b.shape[1], : b.shape[2]] = b[:length]
        elif b.ndim == 4:
            padded_batch[i, :min(length, len(b)), : b.shape[1], : b.shape[2], : b.shape[3]] = b[:length]
        
    return padded_batch

def compare(v1, v2, verbose = True):
    """
        Compare np.ndarray's (or list of) and see if they are close to each other or not
        Quite useful to see if model outputs can be considered assame or not and print useful message if not
    """
    if isinstance(v1, (list, tuple)):
        return all([compare(v1_i, v2_i, verbose = verbose) for v1_i, v2_i in zip(v1, v2)])
    all_close = np.allclose(v1, v2) if v1.shape == v2.shape else False
    if verbose:
        if v1.shape != v2.shape:
            print("Shape mismatch ({} vs {}) !".format(v1.shape, v2.shape))
        else:
            err = np.abs(v1 - v2)
            print("All close : {} - min : {} - max : {} - mean : {} - sum : {}".format(
                all_close, np.min(err), np.max(err), np.mean(err), np.sum(err)
            ))
    return all_close

def parse_args(* args, allow_abrev = True, ** kwargs):
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
    
    parser.parse_args()
    print(vars(parser))
    parsed_args = {}
    for a in args + tuple(kwargs.keys()): parsed_args[a] = getattr(parser, a)
    
    return parsed_args
