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
import json
import uuid
import queue
import logging
import inspect
import argparse
import datetime
import importlib
import multiprocessing.queues
import numpy as np

from math import prod

logger = logging.getLogger(__name__)

""" These functions are related to time convertion """

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

def timestamp_to_str(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime("%d %B %Y %H:%M:%S")

""" These functions are related to data convertion """

def convert_to_str(x):
    """ Convert different string formats (bytes, tf.Tensor, ...) to regular `str` object """
    if isinstance(x, str) or x is None: return x
    elif hasattr(x, 'dtype') and getattr(x.dtype, 'name', None) == 'string': # tensorflow strings
        x = x.numpy()
    elif isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
        logger.warning('numerical array has not been converted to string')
        return x
    
    if isinstance(x, np.ndarray) and x.ndim == 0: x = x.item()
    if isinstance(x, bytes): return x.decode('utf-8')
    
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [convert_to_str(xi) for xi in x]
    elif isinstance(x, dict):
        return {convert_to_str(k) : convert_to_str(v) for k, v in x.items()}
    
    return x

def to_json(data):
    """ Converts a given data to json-serializable (if possible) """
    if data is None: return data
    elif hasattr(data, '__dataclass_fields__') or isinstance(data, argparse.Namespace):
        return {k : to_json(v) for k, v in data.__dict__.items()}
    elif isinstance(data, enum.Enum): data = data.name
    elif hasattr(data, 'shape'):
        data = _naive_convert_to_numpy(data)
        if data.ndim == 0: data = data.item()
        elif prod(data.shape) >= 256:
            logger.warning('Large array of shape {} is serialized to json !'.format(data.shape))
    
    if isinstance(data, bytes): data = data.decode('utf-8')
    
    if isinstance(data, bool):          return data
    elif isinstance(data, uuid.UUID):   return str(data)
    elif isinstance(data, datetime.datetime):    return data.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(data, (float, np.floating)): return float(data)
    elif isinstance(data, (int, np.integer)):    return int(data)
    elif isinstance(data, (list, tuple, set, np.ndarray)):
        return [to_json(d) for d in data]
    elif isinstance(data, dict):
        return {to_json(k) : to_json(v) for k, v in data.items()}
    elif isinstance(data, str):
        from .file_utils import is_path, path_to_unix
        return data if not is_path(data) else path_to_unix(data)
    elif hasattr(data, 'get_config'):
        return to_json(data.get_config())
    elif inspect.isfunction(data):
        return data.__name__
    else:
        logger.warning("Unknown json data ({}) : {}".format(type(data), data))
        return str(data)

def create_iterable(generator, *, timeout = None, ** kwargs):
    """
        Creates a regular iterator (usable in a `for` loop) based on multiple types
            - `pd.DataFrame`    : iterates on the rows
            - `{queue / multiprocessing.queues}.Queue`  : iterates on the queue items (blocking)
            - `callable`    : generator function
            - else  : returns `generator`
        
        Note : `kwargs` are forwarded to `queue.get` (if `Queue` instance) or to the function call (if `callable`)
    """
    if hasattr(generator, 'iterrows'):
        for idx, row in generator.iterrows():
            yield row
    
    elif isinstance(generator, (queue.Queue, multiprocessing.queues.Queue)):
        try:
            while True:
                item = generator.get(timeout = timeout)
                if item is None: break
                yield item
        except queue.Empty:
            pass
    else:
        if inspect.isgeneratorfunction(generator):
            generator = generator(** {
                k : v for k, v in kwargs.items()
                if k in inspect.signature(generator).parameters
            })
        for item in generator:
            yield item

def _naive_is_tensor(data):
    return hasattr(data, 'device')

def _naive_convert_to_numpy(data):
    if hasattr(data, 'detach'): return data.detach().cpu().numpy()
    return np.asarray(data)

""" These functions are `inspect` utilities """

def get_fn_name(fn):
    if hasattr(fn, 'func'): fn = fn.func
    if hasattr(fn, 'name'):         return fn.name
    elif hasattr(fn, '__name__'):   return fn.__name__
    return fn.__class__.__name__

def get_args(fn, include_args = True, ** kwargs):
    """ Returns a `list` of the positional argument names (even if they have default values) """
    return [
        name for name, param in inspect.signature(fn, ** kwargs).parameters.items()
        if (param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))
        or (include_args and param.kind == inspect.Parameter.VAR_POSITIONAL)
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

def has_args(fn, ** kwargs):
    return any(
        param.kind == inspect.Parameter.VAR_POSITIONAL
        for param in inspect.signature(fn, ** kwargs).parameters.values()
    )

def has_kwargs(fn, name = None, ** kwargs):
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD and (name is None or _name == name)
        for _name, param in inspect.signature(fn, ** kwargs).parameters.items()
    )

def signature_to_str(fn, add_doc = False, ** kwargs):
    return '{}{}{}'.format(
        fn.__name__,
        str(inspect.signature(fn, ** kwargs)),
        '\n{}'.format(fn.__doc__) if add_doc else ''
    )

def get_enum_item(value, enum, upper_names = True):
    if isinstance(value, enum): return value
    if isinstance(value, str):
        if upper_names: value = value.upper()
        if not hasattr(enum, value):
            raise KeyError('{} is not a valid {} : {}'.format(value, enum.__name__, tuple(enum)))
        return getattr(enum, value)
    return enum(value)


def get_module_version(module):
    return importlib.metadata.version(module)

""" These functions manipulates `pd.DataFrame` """

_base_aggregation = {
    'count' : len,
    'min'   : np.min,
    'mean'  : np.mean,
    'max'   : np.max,
    'total' : np.sum
}

def is_dataframe(data):
    if 'pandas' not in sys.modules: return False
    return isinstance(data, sys.modules['pandas'].DataFrame)

def set_display_options(columns = 25, rows = 25, width = 125, colwidth = 50):
    import pandas as pd
    
    pd.set_option('display.max_columns', columns)
    pd.set_option('display.max_row', rows)

    pd.set_option('display.width', width)
    pd.set_option('display.max_colwidth', colwidth)


def filter_df(data, on_unique = [], ** kwargs):
    """
        Filters a `pd.DataFrame`
        
        Arguments : 
            - data      : dataframe to filter
            - on_unique : column or list of columns on which to apply criterion on uniques values (see notes for details)
            - kwargs    : key-value pairs of `{column_id : criterion}`
                where criterion can be : 
                - callable (a function) : take as argument the column and return a boolean based on values
                    --> `mask = data[column].apply(fn)`
                - list / tuple  : list of possible values
                    --> `mask = data[column].isin(value)`
                - else  : expected value
                    --> `mask = data[column] == value`
        Return :
            - filtered_data : filtered dataframe
        
        Note : if `on_unique` is used and value is a callable, it is applied on the result of `data[column].value_counts()` that gives a pd.Series, where index are the unique values and the values are their respective occurences (sorted in decreasing order). 
        The function must return boolean values (useful to get only ids with a minimal / maximal number of occurences)
        You can also pass a string (min / max / mean) or an int which represents the index you want to keep (min = index -1, max = index 0, mean = len(...) // 2)
    """
    if not isinstance(on_unique, (list, tuple)): on_unique = [on_unique]
    
    for column, value in kwargs.items():
        if column not in data.columns: continue
        
        if column in on_unique:
            assert callable(value) or isinstance(value, (str, int))
            uniques = data[column].value_counts()
            if isinstance(value, str):
                if value == 'min':      uniques = [uniques.index[-1]]
                elif value == 'max':    uniques = [uniques.index[0]]
                elif value == 'mean':   uniques = [uniques.index[len(uniques) // 2]]
            elif isinstance(value, int):
                uniques = [uniques.index[value]]
            else:
                uniques = uniques[value(uniques)].index
            
            mask = data[column].isin(uniques)
        elif callable(value):
            mask = data[column].apply(value)
        elif isinstance(value, (list, tuple)):
            mask = data[column].isin(value)
        else:
            mask = data[column] == value
        
        data = data[mask]
    return data

def sample_df(data,
              on    = 'id',
              n     = 10,
              n_sample      = 10,
              min_sample    = None,
              random_state  = None,
              drop = True
             ):
    """
        Sample dataframe by taking `n_sample` for `n` different values of column `on`
        Default values means : 'taking 10 samples for 10 different ids'
        
        Arguments :
            - data  : `pd.DataFrame` to sample
            - on    : the `data`'s column to identify groups
            - n     : the number of groups to sample
            - n_sample  : the number of samples for each group (if <= 0, max samples per group)
            - min_sample    : the minimal number of samples for a group to be selected.
                Note that if less than `n` groups have at least `min_sample`, some groups can have less than `min_sample` in the final result.
            - random_state  : state used in the sampling of group's ids and samples (for reproducibility)
            - drop          : cf `drop` argument in `reset_index`, if `False`, tries to ad an `index` column
        Returns :
            - samples   : a pd.DataFrame with `n` different groups and (hopefully) at least `n_sample` for each group
        
        raise ValueError if `n` is larger than the number of groups
    """
    rnd = np.random.RandomState(random_state)

    uniques = data[on].value_counts()
    
    if n is None or n <= 0: n = len(uniques)
    if n_sample is None or n_sample <= 0: n_sample = len(data)
    
    if min_sample is not None:
        uniques = uniques[uniques >= min_sample]
    
    uniques = uniques.index
    if len(uniques) > n:
        uniques = rnd.choice(uniques, n, replace = False)
    
    indexes = []
    for u in uniques:
        samples_i = data[data[on] == u]
        
        n_sample_i = min(
            len(samples_i), n_sample
        ) if not isinstance(n_sample, float) else int(n_sample * len(samples_i))
        
        indexes.extend(rnd.choice(
            samples_i.index, size = n_sample_i, replace = False
        ))
    
    return data.loc[indexes].reset_index(drop = drop)

def aggregate_df(data, group_by, columns = [], filters = {}, merge = False, ** kwargs):
    """
        Computes some aggregation functions (e.g., `np.sum`, `np.mean`, ...) on `data`
        
        Arguments :
            - data  : the original `pd.DataFrame`
            - group_by  : the columns to group for the aggregation
            - columns   : the columns on which to apply the aggregation functions
            - filters   : mapping `{column : filter}` to apply (see `filter_df`)
            - kwargs    : mapping `{aggregation_name : aggregation_fn}`, the aggregation to perform
        Return :
            - aggregated_data   : `pd.DataFrame` with columns `group_by + list(kwargs.keys())`
        
        Example usage :
        ```python
        dataset = get_dataset('common_voice') # contains columns ['id', 'filename', 'time']
        aggregated = aggregate_df(
            dataset,                # audio dataset
            group_by    = 'id',     # groups by the 'id' column
            columns     = 'time',   # computes the functions on the 'time' column
            total   = np.sum,       # computes the total time for each 'id'
            mean    = np.mean       # computes the average time for each 'id'
        )
        print(aggregated.columns)   # ['id', 'total', 'mean']
        ```
        
        Note : if no `kwargs` is provided, the default computation is `count, min, mean, max, total`
    """
    import pandas as pd
    
    if not isinstance(group_by, (list, tuple)): group_by = [group_by]
    if not isinstance(columns, (list, tuple)): columns = [columns]
    if len(columns) == 0: columns = [c for c in data.columns if c not in group_by]
    if len(kwargs) == 0: kwargs = _base_aggregation
    
    for k, v in kwargs.items():
        if isinstance(v, int): kwargs[k] = lambda x: x.values[v]
        elif isinstance(v, str): kwargs[k] = _base_aggregation[v]
    
    name_format = '{name}_{c}' if len(columns) > 1 else '{name}'
    
    data = filter_df(data, ** filters)
    
    result = []
    for group_values, grouped_data in data.groupby(group_by):
        if not isinstance (group_values, (list, tuple)): group_values = [group_values]
        
        grouped_values = {n : v for n, v in zip(group_by, group_values)}
        for c in columns:
            grouped_values.update({
                name_format.format(name = name, c = c) : fn(grouped_data[c])
                for name, fn in kwargs.items()
            })
        result.append(grouped_values)
    
    result = pd.DataFrame(result)
    
    if merge:
        result = pd.merge(data, result, on = group_by)
    
    return result
