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

import numpy as np
import pandas as pd

from functools import wraps

from loggers import timer
from utils.keras_utils import ops
from .generic_utils import get_args
from .wrapper_utils import args_to_kwargs

def convert_to_list(data, rank = None):
    """ Converts `data` to a `list` or a batched array of rank `rank` """
    if isinstance(data, list):              return data
    elif isinstance(data, pd.DataFrame):    return data.to_dict('records')
    elif isinstance(data, (dict, str)):     return [data]
    elif ops.is_array(data):
        if rank is None:    return data
        r = ops.rank(data)
        if r == rank:       return ops.expand_dims(data, axis = 0)
        elif r == rank + 1: return data
        else: raise ValueError('Expected rank = {} or {}, got {}'.format(rank, rank + 1, r))
    else:
        raise ValueError('Unsupported `data` type : {}\n{}'.format(type(data), data))

def stack_batch(batch, pad_value = 0., dtype = 'float32', maybe_pad = False):
    if len(batch) == 1: return ops.expand_dims(batch[0], axis = 0)
    elif maybe_pad and len(set(tuple(b.shape) for b in batch)):
        return ops.cast(pad_batch(batch, dtype = dtype, pad_value = pad_value), dtype)
    return ops.stack(batch, axis = 0)

def pad_batch(batch, pad_value = 0, max_length = None, dtype = None):
    """
        Create a padded version of batch in a single np.ndarray
        Note that this function allows to have different shapes on different dimensions and will pad all of them. 
        However, all data must have the same rank (number of dimensions)
        
        Arguments : 
            - batch         : list of np.ndarray / Tensor
            - pad_value     : the value to add as padding
            - max_length    : maximum length for each dimension. If not given, take the max length of datas 
            - dtype : dtype of the final output
        Return : 
            - padded_batch : np.ndarray of same rank as data
    """
    if len(batch) == 0: return batch
    if not hasattr(batch[0], 'shape'):
        if not isinstance(batch[0], list): return np.array(batch)
        batch = [np.array(b) for b in batch]
    
    if dtype is None: dtype = ops.dtype_to_str(batch[0].dtype)
    
    ndim = len(batch[0].shape)
    assert all(len(b.shape) == ndim for b in batch)
    
    max_shape = np.max(np.array([b.shape for b in batch], dtype = np.int32), axis = 0).tolist()
    if max_length is not None: max_shape[0] = min(max_shape[0], max_length)
    
    padded_batch = np.full([len(batch)] + max_shape, pad_value).astype(dtype)
    for i, b in enumerate(batch):
        slices = (i, ) + tuple([
            slice(0, min(s, max_l)) for s, max_l in zip(b.shape, max_shape)
        ])
        padded_batch[slices] = b[slices[1:]]
    
    return padded_batch

def pad_to_multiple(data, multiple, axis = -1, pad_mode = 'after', ** kwargs):
    """ Pad `seq[axis]` to the next multiple of `multiple` (if not a multiple of it) """
    if isinstance(axis, int):       axis = [axis]
    if isinstance(multiple, int):   multiple = [multiple]
    axis = [ax if ax >= 0 else len(data.shape) + ax for ax in axis]
    
    shape   = ops.shape(data)
    
    should_pad = False
    paddings = []
    for i in range(len(shape)):
        pad = 0
        if i in axis:
            mul  = multiple[axis.index(i)] if len(multiple) > 1 else multiple[0]
            rest = shape[i] % mul
            if rest != 0:
                should_pad  = True
                pad     = mul - rest
        
        if pad_mode == 'before':
            padding = (pad, 0)
        elif pad_mode == 'after':
            padding = (0, pad)
        elif pad_mode == 'even':
            pad_half = pad // 2
            padding = (pad_half, pad - pad_half)
        
        paddings.append(padding)

    return ops.pad(data, paddings, ** kwargs) if should_pad else data

def apply_on_batch(fn = None,
                   *,
                   
                   cond = None,
                   batched_arg  = 0,
                   default_batch_size = None,
                   
                   sort_fn  = None,
                   sort_key = None,
                   
                   concat_fn    = None,
                   concat_axis  = 0
                  ):
    def wrapper(fn):
        @timer(name = 'batched_{}'.format(fn.__name__))
        @wraps(fn)
        @args_to_kwargs(fn)
        def batched_fn(*,
                       
                       batch_size   = default_batch_size,
                       return_inputs    = False,
                       tqdm = None,
                       ** kwargs
                      ):
            if not isinstance(batched_argname, (list, tuple)):
                batch_size = kwargs.pop('batch_size_{}'.format(batched_argname), batch_size)
            
            if cond is not None:
                batch_size, kwargs = cond(batch_size, kwargs)
            
            if not batch_size: return fn(** kwargs)
            
            if tqdm is None: tqdm = lambda x: x
            
            if isinstance(batched_argname, str):
                inputs = _to_iterable(kwargs.pop(batched_argname))
                length = len(inputs)
            else:
                inputs = {n : _to_iterable(kwargs.pop(n)) for n in batched_argname}
                length = len(inputs[batched_argname[0]])
            
            if sort_key is not None:
                if isinstance(inputs, pd.DataFrame):
                    sorted_indexes = sorted(
                        range(length), key = lambda i: sort_key(inputs.iloc[i]), reverse = True
                    )
                    inputs = inputs.iloc[sorted_indexes]
                else:
                    sorted_indexes = sorted(
                        range(length), key = lambda i: sort_key(inputs[i]), reverse = True
                    )
                    inputs = [inputs[idx] for idx in sorted_indexes]
                invert_indexes = np.argsort(np.array(sorted_indexes, dtype = 'int32'))
            
            results = None
            for idx in tqdm(range(0, length, batch_size)):
                if isinstance(batched_argname, (list, tuple)):
                    kwargs.update(_get_batch(inputs, idx, batch_size))
                else:
                    kwargs[batched_argname] = _get_batch(inputs, idx, batch_size)
                
                out = fn(** kwargs)
                
                results = nested_append(results, out)

            if concat_fn is not None:
                results = concat_fn(results)
            else:
                results = nested_concat(results, axis = concat_axis)
            
            if sort_key is not None:
                results = nested_gather(results, invert_indexes, axis = concat_axis)
            
            return results
        
        batched_argname = batched_arg
        if isinstance(batched_arg, int):
            _args = get_args(fn)
            if _args[0] == 'self': _args = _args[1:]
            batched_argname = _args[batched_arg]
            
        return batched_fn
    return wrapper if fn is None else wrapper(fn)

def _to_iterable(value):
    if isinstance(value, str): return [value]
    return value

@timer
def nested_append(acc, value):
    if isinstance(value, (list, tuple)):
        return value.__class__(* [
            nested_append(acc[i] if acc else None, value[i]) for i in range(len(value))
        ])
    elif isinstance(value, dict):
        return {
            k : nested_append(acc[k] if acc else None, value[k]) for k in value.keys()
        }
    
    if acc is None: return [value] if ops.is_array(value) else value
    acc.append(value)
    return acc

@timer
def nested_concat(values, variable_length = None, pad_value = 0., axis = 0):
    if isinstance(values, dict):
        return {k : nested_concat(v, variable_length, pad_value, axis) for k, v in values.items()}
    elif isinstance(values, tuple):
        return values.__class__(* [
            nested_concat(v, variable_length, pad_value, axis) for v in values
        ])
    elif not isinstance(values, list):
        return values
    elif isinstance(values[0], list):
        return [nested_concat(v, variable_length, pad_value, axis) for v in values]
    elif len(values) == 1:
        return values[0]
    
    if variable_length is None:
        variable_length = len(set([
            tuple([s for i, s in enumerate(it.shape) if i != axis]) for it in values]
        )) > 1
    
    if variable_length:
        return concat_sequences(values, pad_value = pad_value, axis = axis)
    else:
        return ops.concatenate(values, axis = axis)

@timer
def nested_gather(values, indexes, axis):
    if isinstance(values, dict):
        return {k : nested_gather(v, indexes, axis) for k, v in values.items()}
    elif isinstance(values, list):
        return [nested_gather(v, indexes, axis) for v in values]
    elif ops.is_array(values):
        return ops.take(values, indexes, axis)
    return values[indexes]
    
def concat_sequences(sequences, pad_value = 0., pad_mode = 'after', axis = 0):
    shapes = ops.stack([
        ops.convert_to_numpy(ops.shape(item)) for item in sequences
    ], axis = 0)
    
    max_it_shape    = ops.max(shapes[:, 1:], axis = 0, keepdims = True)
    pad_shapes      = max_it_shape - shapes[:, 1:]
    
    pad_shapes  = ops.stack([pad_shapes, ops.zeros_like(pad_shapes)], axis = 2)
    if pad_mode == 'before': pad_shapes = ops.flip(pad_shapes, axis = 2)
    pad_shapes = ops.pad(pad_shapes, [(0, 0), (1, 0), (0, 0)], constant_values = 0)
    
    return ops.concatenate([
        ops.pad(it, pad_shapes[i], constant_values = pad_value)
        for i, it in enumerate(sequences)
    ], axis = axis)

def _get_batch(data, start, size):
    if isinstance(data, dict):
        return {k : _get_batch(d, start, size) for k, d in data.items()}
    if isinstance(data, pd.DataFrame): return data.iloc[start : start + size]
    return data[start : start + size]

def truncate(tokens, max_length, keep_mode = 'start'):
    """ Truncate a sequence of shape `(length, )` to `max_length` """
    assert mode in ('random', 'start', 'end')
    
    start = 0
    if len(tokens) > max_length:
        if keep_mode == 'random':
            start = ops.random.uniform(
                (), minval = 0, 
                maxval = len(tokens) - max_length,
                dtype = 'int32'
            )
        elif keep_mode == 'end':
            start = len(tokens) - max_length
        else:
            start = 0
                
    return tokens[start : start + max_length]
    
