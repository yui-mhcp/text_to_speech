
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

import logging
import numpy as np
import tensorflow as tf

logger  = logging.getLogger(__name__)

_max_length = 150
_eos_chars  = ('...', '.', ' ?', ' !', '?', '!')

def get_pairs(text, n = 2):
    """ Creates a n-gram """
    return [tuple(text[i : i + n]) for i in range(0, len(text) - n + 1)]

@tf.function(input_signature = [
    tf.TensorSpec(shape = (None, None), dtype = tf.float32),
    tf.TensorSpec(shape = (None, 2),    dtype = tf.int32),
    tf.TensorSpec(shape = (),           dtype = tf.float32)
])
def remove_tokens(logits, indices, value = tf.float32.min):
    """ Equivalent to `logits[indices] = value` compatible with `tensorflow graph` """
    return tf.tensor_scatter_nd_update(
        logits, indices, tf.fill((tf.shape(indices)[0], ), value)
    )

@tf.function(input_signature = [
    tf.TensorSpec(shape = (None, None), dtype = tf.float32),
    tf.TensorSpec(shape = (None, ),     dtype = tf.int32),
    tf.TensorSpec(shape = (),           dtype = tf.float32)
])
def remove_batch_tokens(logits, indices, value = tf.float32.min):
    """ Equivalent to `logits[:, indices] = value` compatible with `tensorflow graph` """
    indices = tf.stack([
        tf.repeat(tf.range(tf.shape(logits)[0]), tf.shape(indices)[0]),
        tf.tile(indices, [tf.shape(logits)[0]])
    ], axis = 1)
    return remove_tokens(logits, indices, value)

@tf.function(input_signature = [
    tf.TensorSpec(shape = (None, None), dtype = tf.float32),
    tf.TensorSpec(shape = (),           dtype = tf.int32),
    tf.TensorSpec(shape = (),           dtype = tf.bool),
    tf.TensorSpec(shape = (),           dtype = tf.float32)
])
def remove_slice_tokens(logits, index, remove_after, value = tf.float32.min):
    """
        Equivalent to :
        - `logits[:, :index] = value` (`remove_after = False`)
        - `logits[:, index:] = value` (`remove_after = True`)
        compatible in `tensorflow graph`
    """
    indices = tf.cond(
        remove_after,
        lambda: tf.range(index, tf.shape(logits)[-1]),
        lambda: tf.range(0, index)
    )
    return remove_batch_tokens(logits, indices, value)

@tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
def filter_texts(encoded_texts  : tf.Tensor,
                 lengths    : tf.Tensor,

                 min_length : tf.Tensor = -1,
                 max_length : tf.Tensor = -1,

                 max_total_length   : tf.Tensor = -1,
                 sort_by_length : tf.Tensor = False,

                 required_idx   : tf.Tensor = -1,

                 max_texts  : tf.Tensor = -1,
                 select_mode    = 'start',
                 
                 ** kwargs
                ):
    """
        Filter a batch of texts to only keep those with length in the range [min_length, max_length]
        
        Arguments :
            - encoded_texts : 2-D / 3-D `tf.Tensor`, the encoded batch of texts
            - lengths       : 1-D / 2-D `tf.Tensor`, the texts' lengths
            
            - {min / max}_length    : the minimal / maximal length
            
            - max_total_length      : the maximal total cumulated length
            - sort_by_length        : whether to sort by length when filtering on max_total_length
            
            - required_idx  : index to keep even if its length is not in the expected range
            
            - max_texts     : maximum number of texts to keep
            - select_mode   : selection mode if there are too much texts
        Returns :
            - filtered_texts    : the filtered batch of texts
            - filtered_lengths  : the filtered lengths
        
        Warning : if no texts respects the constraints, the 1st outputs' dimension can be 0 !
        
        Note : if `encoded_texts` is a 3-D `tf.Tensor`, it represents a *splitted* version, meaning that :
        - The 1st dimension is the number of paragraphs
        - The 2nd dimension is the number of sentences
        - The 3rd dimension is the encoded sentence
        In this case, `lengths` is a 2-D `tf.Tensor` and the filters on length are applied on the paragraphs' lengths (i.e. `tf.reduce_sum(lengths, axis = -1)`)
    """
    ####################
    # Filter on length #
    ####################
    
    text_lengths    = lengths if len(tf.shape(encoded_texts)) == 2 else tf.reduce_sum(lengths, axis = -1)
    
    valid_mask  = tf.ones_like(text_lengths, dtype = tf.bool)
    if min_length > -1:
        valid_mask = tf.math.logical_and(valid_mask, text_lengths >= min_length)
    
    if max_length > -1:
        valid_mask = tf.math.logical_and(valid_mask, text_lengths <= max_length)
    
    ##############################
    #   Filter on total length   #
    ##############################
    
    if max_total_length > 0 and tf.reduce_any(valid_mask):
        if required_idx != -1:
            valid_mask = tf.tensor_scatter_nd_update(
                valid_mask, [[required_idx]], [True]
            )
        
        if sort_by_length:
            indexes = tf.argsort(text_lengths)
            indexes = tf.boolean_mask(indexes, tf.gather(valid_mask, indexes))
        else:
            indexes = tf.squeeze(tf.where(valid_mask), axis = 1)
        
        skip_mask = tf.math.cumsum(tf.gather(text_lengths, indexes)) > max_total_length
        if tf.reduce_any(skip_mask):
            bad_indexes = tf.boolean_mask(indexes, skip_mask)
            valid_mask  = tf.tensor_scatter_nd_update(
                valid_mask,
                tf.expand_dims(bad_indexes, axis = 1),
                tf.zeros((tf.shape(bad_indexes)[0], ), dtype = tf.bool)
            )

    ##############################
    #   Filter on texts' number  #
    ##############################

    if max_texts > 0:
        if required_idx != -1: max_texts -= 1
        
        if tf.reduce_sum(tf.cast(valid_mask, tf.int32)) > max_texts:
            indexes = tf.squeeze(tf.where(valid_mask), axis = 1)
            
            if select_mode == 'random':
                skip_indexes = tf.random.shuffle(indexes)[max_texts :]
            elif select_mode == 'start':
                skip_indexes = indexes[max_texts:]
            elif select_mode == 'end':
                skip_indexes = indexes[:-max_texts]
            
            valid_mask  = tf.tensor_scatter_nd_update(
                valid_mask,
                tf.expand_dims(skip_indexes, axis = 1),
                tf.zeros((max_texts, ), dtype = tf.bool)
            )
    
    if required_idx != -1:
        valid_mask = tf.tensor_scatter_nd_update(
            valid_mask, [[required_idx]], [True]
        )
    
    encoded_texts   = tf.boolean_mask(encoded_texts, valid_mask)
    lengths         = tf.boolean_mask(lengths,       valid_mask)
    
    if tf.reduce_any(valid_mask):
        encoded_texts = encoded_texts[..., : tf.reduce_max(lengths)]

    return encoded_texts

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
        Split a text (word based) such that each part have at most 'max_length' caracters
    """
    mots = text.split(" ")

    text_parts = []
    length, parts = 0, []
    for mot in mots:
        parts.append(mot)
        length += len(mot)

        if length >= max_length:
            text_parts.append(" ".join(parts))
            length, parts = 0, []
    if length > 0: text_parts.append(" ".join(parts))
    
    return text_parts

def split_text(text, max_length = _max_length):
    """
        Split a text such that each parts have at most 'max_length' caracters. 
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

