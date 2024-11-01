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

import keras
import logging
import numpy as np

from utils import convert_to_str, pad_batch
from utils.keras_utils import TensorSpec, ops, graph_compile

logger  = logging.getLogger(__name__)

_max_length = 150
_eos_chars  = ('...', '.', ' ?', ' !', '?', '!')

def get_pairs(text, n = 2):
    """ Creates a n-gram """
    return [tuple(text[i : i + n]) for i in range(0, len(text) - n + 1)]

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

def build_masking_filter(indices):
    indices = ops.reshape(ops.convert_to_tensor(indices, 'int32'), [-1])
    return lambda scores, ** _: remove_batch_tokens(scores, indices)

@graph_compile
def remove_tokens(logits    : TensorSpec(shape = (None, None), dtype = 'float'),
                  indices   : TensorSpec(shape = (None, 2), dtype = 'int32'),
                  value     : TensorSpec(shape = (), dtype = 'float') = float('-inf')
                 ):
    """ Equivalent to `logits[indices] = value` compatible with `tensorflow graph` """
    return ops.scatter_update(
        logits, indices, ops.full((ops.shape(indices)[0], ), value)
    )

@graph_compile
def remove_batch_tokens(logits    : TensorSpec(shape = (None, None), dtype = 'float'),
                        indices   : TensorSpec(shape = (None, ), dtype = 'int32'),
                        value     : TensorSpec(shape = (), dtype = 'float') = float('-inf')
                       ):

    """ Equivalent to `logits[:, indices] = value` compatible with `tensorflow graph` """
    indices = ops.stack([
        ops.repeat(ops.arange(ops.shape(logits)[0]), ops.shape(indices)[0]),
        ops.tile(indices, [ops.shape(logits)[0]])
    ], axis = 1)
    return remove_tokens(logits, indices, value)

@graph_compile
def remove_slice_tokens(logits  : TensorSpec(shape = (None, None), dtype = 'float'),
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

def extract_sentence(text, pattern):
    pattern = pattern.lower()
    return [sent for sent in split_sentence(text) if pattern in sent.lower()]

def split_sentence(text):
    patterns = [pat + ' ' for pat in _eos_chars] + [pat + '\n' for pat in _eos_chars]
    return [
        (part.strip() + end_char).strip() for part, end_char in multi_split(text, * patterns)
        if len(part.strip()) > 0
    ]

def split_and_join(text, pattern):
    splitted = text.split(pattern)
    for i in reversed(range(1, len(splitted))):
        splitted.insert(i, pattern)
    return splitted
        
def multi_split(text, * separators):
    """
        Split `text` based on multiple separators and returns a list of tuple (sub_text, sep)
        Note that the last part is a tuple with an empty `sep` and a possibly empty `sub_text` (depending if `text` ends with a separator or not)
    """
    result = [(text, '')]
    for sep in separators:
        if sep not in text: continue
        
        new_result = []
        for text, end_c in result:
            parts = text.split(sep)
            for sub_part in parts[:-1]:
                new_result.append((sub_part, sep))
            new_result.append((parts[-1], end_c))
        result = new_result
    return result
    
def simple_text_split(text, max_length = _max_length):
    """
        Split a text (word based) such that each part have at most 'max_length' characters
    """
    words = text.split()

    parts = []
    for word in words:
        if len(parts) == 0 or len(parts[-1]) + len(word) > max_length:
            parts.append(word)
        else:
            parts[-1] += ' ' + word
    
    return parts

def split_text(text, max_length = _max_length):
    """
        Split a text such that each parts have at most 'max_length' characters. 
        The split is based on different criteria : 
        1) Split based on sentence ('_eos_chars' used as delimiters)
        2) If sentences are longer than 'max_length', split them based on comma
        3) If parts are still too long, split them on words
    """
    if isinstance(text, list):
        return [split_text(t, max_length) for t in text]
    
    text = text.replace('\n', ' ').strip()
    if len(text) == 0: return []
    elif len(text) <= max_length: return [text]
    
    if text[-1] in _eos_chars: text += ' '

    parts = []
    for part, end_char in multi_split(text, * _eos_chars):
        part = part.strip()
        # Skip empty parts
        if len(part) == 0: continue
        
        if len(part) <= max_length:
            # If part <= max_length, directly add it
            if len(parts) == 0 or len(parts[-1]) + len(part) > max_length:
                parts.append(part + end_char)
            else:
                parts[-1] += ' ' + part + end_char
        
        elif ', ' in part:
            # If part is longer but contains comma, split it based on commas
            splitted_part = part.split(", ")
            for i, sub_part in enumerate(splitted_part):
                sub_part = sub_part.strip()
                
                end_sub_part = end_char if i == len(splitted_part) -1 else ","
                if len(sub_part) <= max_length:
                    if len(parts) == 0 or len(parts[-1]) + len(sub_part) > max_length:
                        parts.append(sub_part + end_sub_part)
                    else:
                        parts[-1] += ' ' + sub_part + end_sub_part
                else:
                    sub_splitted = simple_text_split(sub_part, max_length)
                    sub_splitted[-1] += end_sub_part
                    for sub in sub_splitted:
                        sub = sub.strip()
                        if len(parts) == 0 or len(parts[-1]) + len(sub) > max_length:
                            parts.append(sub)
                        else:
                            parts[-1] += ' ' + sub
        else:
            splitted_part = simple_text_split(part, max_length)
            splitted_part[-1] += end_char
            for sub_part in splitted_part:
                sub_part = sub_part.strip()
                if len(parts) == 0 or len(parts[-1]) + len(sub_part) > max_length:
                    parts.append(sub_part)
                else:
                    parts[-1] += ' ' + sub_part
    
    return [p for p in parts if len(p) > 0]

