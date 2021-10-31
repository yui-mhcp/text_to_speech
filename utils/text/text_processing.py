# /!\ COPYRIGHT FOR FUNCTIONS `bpe` and `bytes_to_unicode` only /!\
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import numpy as np
import tensorflow as tf

from utils.text.cleaners import collapse_whitespace, remove_tokens, remove_punctuation, lowercase

_max_length = 150

_end_sentence = ('...', '.', ' ?', ' !', '?', '!')

def _normalize_text_f1(text, exclude = []):
    return collapse_whitespace(remove_tokens(remove_punctuation(lowercase(text)), exclude)).strip()

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(text, n = 2):
    return [tuple(text[i : i + n]) for i in range(0, len(text) - n + 1)]

def bpe(token, bpe_ranks):
    word = tuple(token)
    pairs = get_pairs(word)
    
    if not pairs: return token
    
    while True:
        bigram = min(pairs, key = lambda pair: bpe_ranks.get(pair, float('inf')))
        
        if bigram not in bpe_ranks: break
        
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                new_word.extend(word[i:])
                break
            else:
                new_word.extend(word[i:j])
                i = j
            
            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1: break
        else: pairs = get_pairs(word)
    return word

def exact_match(y_true, y_pred):
    return int(y_true == y_pred)

def f1(y_true, y_pred, normalize = True, exclude = None):
    if isinstance(y_true, tf.Tensor): y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor): y_pred = y_pred.numpy()
    if isinstance(y_true, bytes): y_true = y_true.decode('utf-8')
    if isinstance(y_pred, bytes): y_pred = y_pred.decode('utf-8')
    if isinstance(y_true, (list, tuple, np.ndarray)):
        return [
            f1(true_i, pred_i, normalize = normalize, exclude = exclude)
            for true_i, pred_i in zip(y_true, y_pred)
        ]
    
    if normalize:
        y_true = _normalize_text_f1(y_true, exclude)
        y_pred = _normalize_text_f1(y_pred, exclude)
    elif exclude:
        y_true = collapse_whitespace(remove_tokens(y_true, exclude))
        y_pred = collapse_whitespace(remove_tokens(y_pred, exclude))
    
    true_tokens = y_true.split()
    pred_tokens = y_pred.split()
    
    common = collections.Counter(true_tokens) & collections.Counter(pred_tokens)
    nb_same = sum(common.values())
    
    em = exact_match(y_true, y_pred)
    
    if len(true_tokens) == 0 or len(pred_tokens) == 0:
        f1 = int(true_tokens == pred_tokens)
        return em, f1, f1, f1
    elif nb_same == 0:
        return 0, 0, 0, 0
    
    precision = 1. * nb_same / len(pred_tokens)
    recall    = 1. * nb_same / len(true_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return em, f1, precision, recall

def create_padding_mask(seq, seq_len = None, pad_value = 0, dtype = tf.float32):
    """
        Return padding mask matching attention shape [batch_size, 1, 1, seq_len]
    """
    if seq_len is None:
        mask = tf.cast(tf.math.equal(seq, pad_value), dtype = dtype)
    else:
        mask = 1. - tf.sequence_mask(
            seq_len, maxlen = tf.shape(seq)[1], dtype = dtype
        )
    return tf.reshape(mask, [tf.shape(seq)[0], 1, 1, -1])

def create_look_ahead_mask(batch_size, size, dtype = tf.float32):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.tile(tf.reshape(mask, [1, 1, size, size]), [batch_size, 1, 1, 1])
    
    return tf.cast(mask, dtype = dtype)

def create_combined_mask(target, seq_len, pad_value = 0):
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[0], tf.shape(target)[1])
    padding_mask    = create_padding_mask(
        target, seq_len = seq_len, pad_value = pad_value, dtype = look_ahead_mask.dtype
    )
    
    return tf.maximum(look_ahead_mask, padding_mask)

def create_transformer_masks(inp, target, input_pad_value = 0, target_pad_value = 0):
    """
        Return 2 masks used in Transformer (Encoder + Decoder) architecture
        
        Arguments : 
            - inp       : input sequence (encoder input)
            - target    : target sequence (decoder input)
            - {input / target}_pad_value    : padding value for input / target sequence
        Return : [enc_padding_mask, combined_mask]
            - enc_padding_mask  : padding mask for encoder attention blocks
            - combined_mask     : combination of look_ahead_mask + padding mask on decoder input (target) for the 1st attention block of decoder layer
        
        Note : enc_padding_mask is used in encoder's MHA but also in the 2nd block of MHA in decoders layers
    """
    padding_mask    = create_padding_mask(inp, pad_value = input_pad_value)
    
    combined_mask   = create_combined_mask(target, target_pad_value)
    
    return padding_mask, combined_mask
    
def split_and_join(text, pattern):
    splitted = text.split(pattern)
    for i in reversed(range(1, len(splitted))):
        splitted.insert(i, pattern)
    return splitted

def multi_split(text, * separators):
    """
        Split a text (str) based on multiple separators and return a list of tuple (part, separator) 
    """
    liste = [(text, '')]
    for sep in separators:
        new_liste = []
        for text, end_c in liste:
            parts = text.split(sep)
            for sub_part in parts[:-1]:
                new_liste.append((sub_part, sep))
            new_liste.append((parts[-1], end_c))
        liste = new_liste
    return liste
    
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

def split_sentence(text):
    patterns = [pat + ' ' for pat in _end_sentence]
    return [
        part.strip() + end_char for part, end_char in multi_split(text, * patterns) if len(part.strip()) > 0
    ]

def split_text(text, max_length = _max_length):
    """
        Split a text such that each parts have at most 'max_length' caracters. 
        The split is based on different criteria : 
        1) Split based on sentence ('_end_sentence' used as delimiters)
        2) If sentences are longer than 'max_length', split them based on comma
        3) If parts are still too long, split them on words
    """
    if isinstance(text, list):
        return [split_text(t, max_length) for t in text]
    
    text = text.replace('\n', ' ').strip()
    if len(text) == 0: return []
    elif len(text) <= max_length: return [text]
    
    if text[-1] in _end_sentence: text += ' '

    parts = []
    for part, end_char in multi_split(text, *_end_sentence):
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

