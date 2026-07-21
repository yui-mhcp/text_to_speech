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

import os
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

_windows = sys.platform.lower().startswith('win')

""" These functions are related to time convertion """

def time_to_string(seconds):
    """ Returns a string representation of a time (given in seconds) """
    if seconds < 0.001: return '{} \u00B5s'.format(int(seconds * 1000000))
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

def timestamp_to_str(timestamp, include_time = True):
    return datetime.datetime.fromtimestamp(timestamp).strftime(
        "%d-%m-%Y %H:%M" if include_time else '%d %B %Y'
    )

""" These functions are related to data convertion """

def convert_to_str(x):
    """ Convert different string formats (bytes, tf.Tensor, ...) to regular `str` object """
    match x:
        case None | str():
            return x
        case bytes():
            return x.decode()
        case list() | tuple() | set():
            return [convert_to_str(xi) for xi in x]
        case dict():
            return {k : convert_to_str(v) for k, v in x.items()}
        case object(shape = _):
            if hasattr(x, 'numpy'): x = x.numpy()
            if np.issubdtype(x.dtype, np.number):
                logger.warning('numerical array has not been converted to string')
                return x
            else:
                # delegate to the recursive `bytes` / `str` cases : `astype(str)` on a
                # `bytes` array (dtype 'S') decodes as ASCII and raises on UTF-8 input,
                # whereas `.tolist()` yields python `bytes`, decoded as UTF-8 below.
                return convert_to_str(x.tolist())
        case _:
            return str(x)

def to_json(data):
    """ Converts a given data to json-serializable (if possible) """
    match data:
        case None | bool() | int() | float():
            return data
        case str():
            if _windows and 1 < len(data) < 512 and '\\' in data and os.path.exists(data):
                return data.replace('\\', '/')
            else:
                return data
        case bytes():
            return data.decode('utf-8')
        case list() | tuple() | set():
            return [to_json(d) for d in data]
        case dict():
            return {k : to_json(v) for k, v in data.items()}
        case object(shape = _):
            return _naive_convert_to_numpy(data).tolist()
        case uuid.UUID():
            return str(data)
        case datetime.datetime():
            return data.strftime("%Y-%m-%d %H:%M:%S")
        case argparse.Namespace() | object(__dataclass_fields__ = _):
            return {k : to_json(v) for k, v in data.__dict__.items()}
        case object(get_config = _):
            return {k : to_json(v) for k, v in data.get_config().items()}
        case _:
            if inspect.isfunction(data):
                return '{}.{}'.format(data.__module__, data.__name__)
            else:
                logger.warning("Unknown json data ({}) : {}".format(type(data), data))
                return str(data)

def map_structure(fn, structure):
    """
        Recursively applies `fn` to every leaf of a nested structure (`list` / `tuple` / `dict`),
        preserving the container types, and returns the resulting structure.

        A "leaf" is any value that is neither a `list`, `tuple` nor `dict` (e.g. a `TensorSpec`).
        Used e.g. by `BaseModel` to derive `input_shape` / `input_dtype` / `unbatched_*` from the
        (possibly nested) `input_signature` / `output_signature`.
    """
    if isinstance(structure, dict):
        return {k : map_structure(fn, v) for k, v in structure.items()}
    elif isinstance(structure, (list, tuple)):
        return type(structure)(map_structure(fn, v) for v in structure)
    return fn(structure)

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
    return hasattr(data, 'device') and not isinstance(data, np.ndarray)

def _naive_convert_to_numpy(data):
    if not hasattr(data, 'shape'):
        return np.asarray(data)
    elif hasattr(data, 'detach'):
        return data.detach().cpu().numpy()
    elif hasattr(data, 'numpy'):
        return data.numpy()
    else:
        return np.asarray(data)

""" These functions manipulate raw text """

def dedent(text):
    """
        Removes the common leading indentation from every line of `text`

        Unlike `textwrap.dedent`, newlines located **inside a double-quoted string literal**
        (e.g. the `\\n` of `{{- "## Title\\n\\n" -}}` in a Jinja template) are *not* treated as
        line breaks, so the indentation of such literals is preserved. This makes it suitable to
        dedent the multi-line Jinja templates defined in `models.nlu.prompts.templates`.
    """
    lines   = _split_lines(text)
    if not lines: return text
    common  = min((_count_indent(l) for l in lines if l.strip()), default = 0)
    return '\n'.join([l.replace(' ' * common, '', 1) for l in lines]).strip()

def _split_lines(text):
    """
        Splits `text` on newlines, skipping newlines located inside a double-quoted string
        literal (tracked with a naive quote-parity heuristic). The leading `-1` sentinel keeps
        the text located before the first newline and makes the function safe on newline-free
        input (`indexes` is never empty).
    """
    is_string = False

    indexes = [-1]
    for i, c in enumerate(text):
        if (c == '"') and (i == 0 or i == len(text) - 1 or not text[i-1].isalnum() or not text[i+1].isalnum()):
            is_string = not is_string
        elif c == '\n' and not is_string:
            indexes.append(i)

    if indexes[-1] != len(text): indexes.append(len(text))
    return [
        text[idx + 1 : indexes[i + 1]].strip('\n') for i, idx in enumerate(indexes[:-1])
    ]

def _count_indent(line):
    for i, c in enumerate(line):
        if not c.isspace(): return i
    return len(line)

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

def import_submodules(package, *, filter_fn = None):
    """
        Imports and returns every importable submodule of `package` (auto-registration backbone)
        
        Arguments :
            - `package`     : the package name (`str`) or the package module itself
            - `filter_fn`   : an optional `callable (name) -> bool` to restrict which entries are imported
        Return :
            - modules       : the `list` of imported submodules

        Skips dotfiles, `_`-prefixed names and `*_old*` (dead code), as well as any non-`.py` file
        (e.g. `CLAUDE.md` / `README.md`), while keeping sub-directories (sub-packages).
    """
    if not isinstance(package, str): package = package.__name__

    modules = []
    for module in os.listdir(package.replace('.', os.path.sep)):
        if module.startswith(('.', '_')) or '_old' in module: continue
        elif '.' in module and not module.endswith('.py'): continue
        elif filter_fn is not None and not filter_fn(module): continue
        modules.append(importlib.import_module(package + '.' + module.removesuffix('.py')))
    return modules

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