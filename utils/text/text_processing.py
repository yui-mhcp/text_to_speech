
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

def filter_texts(encoded_texts,
                 lengths,

                 min_length     = -1,
                 max_length     = -1,

                 max_total_length   = -1,
                 sort_by_length     = False,

                 required_idx   = -1,

                 max_texts      = -1,
                 select_mode    = 'start',

                 ** kwargs
                ):
    """
        Filter a batch of texts with length in the range [min_length, max_length]
        
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
    
    if required_idx != -1:
        valid_mask = tf.math.logical_or(valid_mask, tf.range(tf.shape(lengths)[0]) == required_idx)

    if not tf.reduce_all(valid_mask):
        logger.debug('Valid texts for length-based filtering : {}'.format(valid_mask))
        
        encoded_texts   = tf.boolean_mask(encoded_texts, valid_mask)
        text_lengths    = tf.boolean_mask(text_lengths, valid_mask)
        lengths         = tf.boolean_mask(lengths, valid_mask)
    
    ##############################
    #   Filter on total length   #
    ##############################
    
    if max_total_length > 0 and tf.shape(lengths)[0] > 0 and tf.reduce_sum(text_lengths) > max_total_length:
        indexes = tf.range(tf.shape(lengths)[0]) if not sort_by_length else tf.argsort(text_lengths)
        
        if required_idx != -1:
            indexes     = tf.concat([
                [required_idx], tf.boolean_mask(indexes, indexes != required_idx)
            ], axis = -1)
        
        cum_text_lengths = tf.math.cumsum(tf.gather(text_lengths, indexes))

        valid_indexes  = tf.boolean_mask(indexes, cum_text_lengths <= max_total_length)
        if sort_by_length: valid_indexes = tf.sort(valid_indexes)
        
        logger.debug('Selected indexes (max_total_length filtering) : {}'.format(valid_indexes))
        encoded_texts   = tf.gather(encoded_texts, valid_indexes)
        text_lengths    = tf.gather(text_lengths, valid_indexes)
        lengths         = tf.gather(lengths, valid_indexes)

    ##############################
    #   Filter on texts' number  #
    ##############################

    if max_texts > 0 and tf.shape(lengths)[0] > 0:
        if required_idx != -1: max_texts -= 1
        
        if tf.shape(lengths)[0] > max_texts:
            indexes = tf.range(tf.shape(lengths)[0])
            if required_idx != -1: indexes = tf.boolean_mask(indexes, indexes != required_idx)
            
            if select_mode == 'random':
                indexes = tf.random.shuffle(indexes)[:max_texts]
            elif select_mode == 'random_sorted':
                indexes = tf.sort(tf.random.shuffle(indexes)[:max_texts])
            elif select_mode == 'start':
                indexes = indexes[:max_texts]
            elif select_mode == 'end':
                indexes = indexes[-max_texts:]
            
            if required_idx != -1: indexes = tf.concat([[required_idx], indexes], axis = 0)
            
            logger.debug('Selected indexes (max_texts filtering) : {}'.format(indexes))
            encoded_texts   = tf.gather(encoded_texts, indexes)
            text_lengths    = tf.gather(text_lengths, indexes)
            lengths         = tf.gather(lengths, indexes)
    
    if len(lengths) > 0:
        encoded_texts = encoded_texts[..., : tf.reduce_max(lengths)]

    return encoded_texts, lengths

def create_padding_mask(seq, seq_len = None, pad_value = 0, maxlen = None, dtype = tf.float32):
    """
        Return padding mask matching attention shape [batch_size, 1, 1, seq_len]
    """
    if seq_len is None:
        mask = tf.cast(tf.math.equal(seq, pad_value), dtype = dtype)
    else:
        if maxlen is None: maxlen = tf.shape(seq)[1]
        mask = 1. - tf.sequence_mask(
            seq_len, maxlen = maxlen, dtype = dtype
        )
    return tf.reshape(mask, [tf.shape(seq)[0], 1, 1, -1])

def create_look_ahead_mask(batch_size, size, dtype = tf.float32):
    """ Creates a `look ahead` mask with shape [batch_size, 1, size, size] """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.tile(tf.reshape(mask, [1, 1, size, size]), [batch_size, 1, 1, 1])
    
    return tf.cast(mask, dtype = dtype)

def create_combined_mask(target, seq_len, pad_value = 0):
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[0], tf.shape(target)[1])
    padding_mask    = create_padding_mask(
        target, seq_len = seq_len, pad_value = pad_value, dtype = look_ahead_mask.dtype
    )
    
    return tf.maximum(look_ahead_mask, padding_mask)
    
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

