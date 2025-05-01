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

import numpy as np

from .keras import ops

def pad_batch(batch, pad_value = 0, dtype = None, pad_mode = 'after'):
    """
        Create a padded version of batch in a single np.ndarray
        Note that this function allows to have different shapes on different dimensions, and will pad all of them. 
        However, all data must have the same rank (number of dimensions)
        
        Arguments : 
            - batch         : `llist` of data (`np.ndarray`, `Tensor`, `int`, `float`, `bool`)
            - pad_value     : the value to add as padding
            - dtype         : dtype of the final output
        Return : 
            - padded_batch : np.ndarray of same rank as data + 1 (the batch dimension)
    """
    assert pad_mode in ('before', 'after')
    
    if len(batch) == 0: return batch
    if not hasattr(batch[0], 'shape'):
        if not isinstance(batch[0], list): return np.asarray(batch, dtype = dtype)
        batch = [
            pad_batch(b, pad_value = pad_value, dtype = dtype, pad_mode = pad_mode)
            for b in batch
        ]
    
    if len(batch) == 1:
        return ops.expand_dims(ops.cast(batch[0], dtype) if dtype else batch[0], axis = 0)
    elif pad_value is None or len(set(tuple(b.shape) for b in batch)) == 1:
        return ops.stack(batch, axis = 0)
    
    batch = [ops.convert_to_numpy(b) for b in batch]
    if dtype is None:   dtype = ops.dtype_to_str(batch[0].dtype)
    else:               dtype = ops.dtype_to_str(dtype)
    
    rank = batch[0].ndim
    assert all(b.ndim == rank for b in batch), 'All ranks must be equal !'
    
    max_shape = np.max(np.array([b.shape for b in batch], dtype = np.int32), axis = 0).tolist()
    
    padded_batch = np.full([len(batch)] + max_shape, np.asarray(pad_value, dtype = dtype))
    for i, b in enumerate(batch):
        if pad_mode == 'after':
            slices = tuple(slice(0, s) for s in b.shape)
        else:
            slices = tuple(slice(max_s - s, max_s) for s, max_s in zip(b.shape, max_shape))
        
        padded_batch[(i, ) + slices] = b
    
    return padded_batch

def pad_to_multiple(data, multiple, axis = -1, pad_mode = 'after', ** kwargs):
    """ Pad `data[axis]` to the next multiple of `multiple` (if not already a multiple) """
    if isinstance(axis, int):       axis = [axis]
    if isinstance(multiple, int):   multiple = [multiple] * len(axis)
    axis = [ax if ax >= 0 else len(data.shape) + ax for ax in axis]
    
    shape   = ops.shape(data)
    
    should_pad, paddings = False, [(0, 0)] * len(shape)
    for ax, mul in zip(axis, multiple):
        if shape[ax] % mul == 0: continue
        
        should_pad = True
        
        pad = mul - shape[ax] % mul
        
        if pad_mode == 'before':
            paddings[ax] = (pad, 0)
        elif pad_mode == 'after':
            paddings[ax] = (0, pad)
        elif pad_mode == 'even':
            pad_half = pad // 2
            paddings[ax] = (pad_half, pad - pad_half)

    return ops.pad(data, paddings, ** kwargs) if should_pad else data
