
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

import numpy as np
import tensorflow as tf

def truncate(tokens, max_length, keep_mode = 'start'):
    """ Truncate a sequence of shape `(length, )` to `max_length` """
    assert mode in ('random', 'start', 'end')
    
    start = 0
    if tf.shape(tokens)[0] > max_length:
        if keep_mode == 'random':
            start = tf.random.uniform(
                (), minval = 0, 
                maxval = tf.shape(tokens)[0] - max_length,
                dtype = tf.int32
            )
        elif keep_mode == 'end':
            start = tf.shape(tokens)[0] - max_length
        else:
            start = 0
                
    return tokens[start : start + max_length]

def pad_batch(batch, pad_value = 0, max_length = None, dtype = None):
    """
        Create a padded version of batch in a single np.ndarray
        Note that this function allows to have different shapes on different dimensions and will pad all of them. 
        However, all data must have the same rank (number of dimensions)
        
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

def concat_sequences(seq1, seq2, pad_value):
    """
        Concat 2 sequences on the 0-axis
        Arguments :
            - seq1  : sequence with shape `(n1, len_1)`
            - seq2  : sequence with shape `(n2, len_2)`
            - pad_value : the padding value for the shortest sequence
        Returns :
            - concatenation with shape `(n1 + n2, max(len_1, len_2))`
    """
    len_1, len_2 = tf.shape(seq1)[1], tf.shape(seq2)[1]

    if len_1 != len_2:
        padding = [(0,0), (0, tf.abs(len_1 - len_2))]
        if len_1 > len_2:
            seq2 = tf.pad(seq2, padding, constant_values = pad_value)
        else:
            seq1 = tf.pad(seq1, padding, constant_values = pad_value)
    
    return tf.concat([seq1, seq2], axis = 0)

def pad_to_multiple(seq, multiple, axis = 1, pad_after = True, ** kwargs):
    """ Pad `seq[axis]` to the next multiple of `multiple` (if not a multiple of it) """
    if axis < 0: axis = len(tf.shape(seq)) + axis
    rest = tf.shape(seq)[axis] % multiple
    if rest != 0:
        pad = (0, multiple - rest) if pad_after else (multiple - rest, 0)
        padding = [
            (0, 0) for _ in range(axis)] + [pad] + [
            (0, 0) for _ in range(axis + 1, len(tf.shape(seq)))
        ]
        
        seq = tf.pad(seq, padding, ** kwargs)
        
    return seq

