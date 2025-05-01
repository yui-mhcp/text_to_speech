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

import logging
import numpy as np

from .. import convert_to_str, pad_batch
from ..keras import TensorSpec, ops, graph_compile

logger  = logging.getLogger(__name__)

def process_model_output(output, offset = None, lengths = None):
    if hasattr(output, 'lengths') or hasattr(output, 'offset'):
        lengths = ops.convert_to_numpy(output.lengths)
        if not hasattr(output, 'offset'):
            offset = np.zeros_like(lengths)
        else:
            offset = ops.convert_to_numpy(output.offset)
            if lengths.ndim == 2 and offset.ndim == 1:
                offset = np.tile(offset[:, None], [1, lengths.shape[1]])
        output  = output.tokens
    
    if lengths.ndim:
        return [
            process_model_output(out, off, length)
            for out, off, length in zip(output, offset, lengths)
        ]
    
    return ops.convert_to_numpy(output[offset : lengths])

def mask_tokens(logits    : TensorSpec(shape = (None, None), dtype = 'float'),
                indices   : TensorSpec(shape = (None, 2), dtype = 'int32'),
                value     : TensorSpec(shape = (), dtype = 'float') = float('-inf')
               ):
    """ Equivalent to `logits[indices] = value` compatible with `tensorflow graph` """
    return ops.scatter_update(
        logits, indices, ops.full((ops.shape(indices)[0], ), value)
    )

def mask_batch_tokens(logits    : TensorSpec(shape = (None, None), dtype = 'float'),
                      indices   : TensorSpec(shape = (None, ), dtype = 'int32'),
                      value     : TensorSpec(shape = (), dtype = 'float') = float('-inf')
                     ):

    """ Equivalent to `logits[:, indices] = value` compatible with `tensorflow graph` """
    indices = ops.stack([
        ops.repeat(ops.arange(ops.shape(logits)[0]), ops.shape(indices)[0]),
        ops.tile(indices, [ops.shape(logits)[0]])
    ], axis = 1)
    return mask_tokens(logits, indices, value)

def mask_slice_tokens(logits  : TensorSpec(shape = (None, None), dtype = 'float'),
                      index   : TensorSpec(shape = (), dtype = 'int32'),
                      remove_after    : TensorSpec(shape = (), dtype = 'bool', static = True),
                      value     : TensorSpec(shape = (), dtype = 'float') = float('-inf')
                     ):
    """
        Equivalent to :
        - `logits[:, :index] = value` (`remove_after = False`)
        - `logits[:, index:] = value` (`remove_after = True`)
        compatible in `tensorflow graph`
    """
    if remove_after:
        start_idx, length = index, ops.shape(logits)[1] - index
    else:
        start_idx, length = 0, index
    
    update = ops.full((ops.shape(logits)[0], length), value, dtype = logits.dtype)
    return ops.slice_update(
        logits, ops.array([0, 1], dtype = 'int32') * start_idx, update
    )

def filter_texts(encoded_texts,
                 lengths,

                 min_text_length    = -1,
                 max_text_length    = -1,
                 max_sentences  = -1,
                 max_sentence_length    = -1,

                 max_total_length   = -1,
                 sort_by_length = False,

                 max_texts  = -1,
                 select_mode    = 'start',
                 
                 required_idx   = -1,
                 flatten    = True,
                 
                 return_indices = False,

                 ** kwargs
                ):
    ####################
    # Filter on length #
    ####################
    
    required_idx    = int(required_idx)
    lengths     = pad_batch(lengths, dtype = np.int32, pad_value = 0)
    is_multi    = lengths.ndim == 2
    text_lengths    = lengths if not is_multi else np.sum(lengths, axis = -1)
    
    valid_mask  = np.ones((len(text_lengths), ), dtype = bool)
    if min_text_length > -1:
        valid_mask[text_lengths < min_text_length] = False
    
    if max_text_length > -1:
        valid_mask[text_lengths > max_text_length] = False
    
    if is_multi:
        if max_sentences > 0:
            valid_mask[np.sum(lengths > 0, axis = -1) > max_sentences] = False
        
        if max_sentence_length > -1:
            valid_mask[np.max(lengths, axis = -1) > max_sentence_length] = False

    ##############################
    #   Filter on total length   #
    ##############################
    
    if max_total_length > 0 and np.sum(text_lengths[valid_mask]) > max_total_length:
        if sort_by_length:
            indexes = np.argsort(text_lengths)
            indexes = indexes[valid_mask[indexes]]
        else:
            indexes = np.where(valid_mask)[0]
        
        if required_idx != -1:
            indexes = np.concatenate([[required_idx], indexes[indexes != required_idx]], axis = 0)
        
        skip_mask = np.cumsum(text_lengths[indexes]) > max_total_length
        if np.any(skip_mask):
            skip_indexes = indexes[skip_mask]
            valid_mask[skip_indexes] = False

    ##############################
    #   Filter on texts' number  #
    ##############################

    if max_texts > 0:
        if required_idx != -1: max_texts -= 1
        
        if np.sum(valid_mask) > max_texts:
            select_mode = convert_to_str(select_mode)
            
            indexes = np.where(valid_mask)[0]
            if required_idx != -1: indexes = indexes[indexes != required_idx]
            
            if select_mode == 'random':
                skip_indexes = np.random.choice(indexes, size = max_texts)
            elif select_mode == 'start':
                skip_indexes = indexes[max_texts:]
            elif select_mode == 'end':
                skip_indexes = indexes[:-max_texts]
            else:
                raise ValueError('Unknown `select_mode` : {}'.format(select_mode))
            
            valid_mask[skip_indexes] = False
    
    if required_idx != -1 and not valid_mask[required_idx]: valid_mask[:] = False
    
    lengths = lengths[valid_mask]
    if isinstance(encoded_texts, list):
        encoded_texts = [text for text, valid in zip(encoded_texts, valid_mask) if valid]
    else:
        encoded_texts = encoded_texts[valid_mask]
        
        if is_multi and flatten:
            encoded_texts   = np.reshape(encoded_texts, [-1, encoded_texts.shape[-1]])
            lengths         = np.reshape(lengths, [-1])

            encoded_texts   = encoded_texts[lengths > 0]
            lengths         = lengths[lengths > 0]
        
        if len(encoded_texts) > 0:
            encoded_texts = encoded_texts[..., : np.max(lengths)]
            
            if is_multi and not flatten:
                encoded_texts = encoded_texts[:, : np.max(np.sum(lengths > 0, axis = -1)), :]
    
    if return_indices: return encoded_texts, lengths, np.where(valid_mask)[0]
    return encoded_texts, lengths

