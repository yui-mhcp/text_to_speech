
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

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from hparams import HParams
from utils import pad_batch, convert_to_str
from utils.text import TextEncoder, get_encoder, filter_texts, random_mask
from models.interfaces.base_model import BaseModel

logger = logging.getLogger(__name__)

TextTrainingHParams = HParams(
    nb_mask   = 1,
    min_mask_length   = 1,
    max_mask_length   = 1
)

def _get_key_value_format(data, keys = None, keys_mapping = None, ** kwargs):
    """
        Returns a tuple of 2 lists (keys, values) where keys[i] is the name for values[i]
        
        Arguments :
            - data  : the values (either dict, pd.Series, list, tuple, str, ...)
            - keys  : the expected keys
                if `data` is a dict / pd.Series, `keys` are extracted from it (others are ignored)
            - keys_mapping  : maps a (list of) key(s) to another name
                Typically a dict where keys are the final expected name and values are a (list of) possible names
            - kwargs  : unused
        Returns :
            - keys      : the final names for the mapping
            - values    : the values
    """
    def _get_alternative_key(data, alternatives, default = ''):
        for k in alternatives:
            if k in data: return data[k]
        return default
    
    def _get_key_mapping(key):
        for k, alternatives in keys_mapping.items():
            if key == k or key in alternatives: return k
        return key
    
    if isinstance(data, (dict, pd.Series)):
        if not keys: keys = keys_mapping if keys_mapping else list(data.keys())
        if not isinstance(keys, (list, tuple, dict)): keys = [keys]
        
        if isinstance(keys, dict):
            keys = {k : v if isinstance(v, (list, tuple)) else [v] for k, v in keys.items()}
            keys, values = list(zip(* [(k, _get_alternative_key(data, v + [k])) for k, v in keys.items()]))
        else:
            values = [data.get(k, '') for k in keys]
    
    elif not isinstance(data, (list, tuple)):
        values = [data]
    else:
        values = data

    if keys_mapping and keys != keys_mapping:
        if not isinstance(keys_mapping, (list, tuple, dict)): keys_mapping = [keys_mapping]
        
        if not keys:
            keys = keys_mapping if not isinstance(keys_mapping, dict) else list(keys_mapping.keys())
        elif isinstance(keys_mapping, dict):
            keys_mapping    = {k : v if isinstance(v, (list, tuple)) else [v] for k, v in keys_mapping.items()}
            keys = [_get_key_mapping(k) for k in keys]
        else:
            assert len(keys) == len(keys_mapping), '{} vs {}'.format(keys, keys_mapping)
            if len(keys_mapping) > 1:
                logger.warning('Make sure that the order of `keys` ({}) match the order of `keys_mapping` ({}) !'.format(keys, keys_mapping))
            keys = keys_mapping
    
    if keys is None:
        raise ValueError('You must either specify `keys`, `keys_mapping` or pass `data` as a dict')
    elif len(keys) != len(values):
        raise ValueError('{} keys vs {} values !\n  Keys : {}\n  Values : {}'.format(
            len(keys), len(values), keys, values
        ))
    
    return keys, values

class BaseTextModel(BaseModel):
    def _init_text(self, lang, text_encoder = None, text_encoder_config = {}, ** kwargs):
        """ Init variables for text-based models """
        self.lang   = lang
        
        # Initialization of Text Encoder
        self.text_encoder_config    = text_encoder_config
        self.text_encoder = get_encoder(
            text_encoder = text_encoder, lang = lang, ** text_encoder_config
        )
    
    @property
    def text_encoder_file(self):
        return os.path.join(self.save_dir, 'text_encoder.json')

    @property
    def text_signature(self):
        return tf.TensorSpec(shape = (None, None), dtype = tf.int32)

    @property
    def multi_text_signature(self):
        return tf.TensorSpec(shape = (None, None, None), dtype = tf.int32)

    @property
    def training_hparams_text(self):
        return TextTrainingHParams()
    
    @property
    def vocab(self):
        return self.text_encoder.vocab

    @property
    def vocab_size(self):
        return self.text_encoder.vocab_size

    @property
    def model_tokens(self):
        return {
            'sos_token' : self.sos_token_idx,
            'eos_token' : self.eos_token_idx,
            'pad_token' : self.blank_token_idx,
        }
    
    @property
    def blank_token_idx(self):
        return self.text_encoder.blank_token_idx

    @property
    def sep_token(self):
        return self.text_encoder.sep_token

    @property
    def sep_token_idx(self):
        return self.text_encoder.sep_token_idx
    
    @property
    def mask_token_idx(self):
        return self.text_encoder.mask_token_idx
    
    @property
    def sos_token_idx(self):
        return self.text_encoder.sos_token_idx

    @property
    def eos_token_idx(self):
        return self.text_encoder.eos_token_idx

    def _str_text(self):
        des = "- Language : {}\n".format(self.lang)
        des += "- Vocabulary (size = {}) : {}\n".format(
            self.vocab_size,
            self.vocab if len(self.vocab) < 25 else '[{}, ...]'.format(str(self.vocab[:25])[1:-1])
        )
        return des
    
    def set_text_encoder(self, new_encoder, lang = None, ** kwargs):
        """
            Change the current `text_encoder` to `new_encoder` and possibly update the model's vocabulary (if it has the mothod `change_vocabulary`)
            
            Arguments :
                - new_encoder   : the new text encoder
                    - filename (str)    : the text encoder's config filename
                    - model_name (str)  : the model's name from which to get the text encoder's config file
                    - TextEncoder       : the instance
                    - BaseTextModel's subclass  : the model instance
                - lang  : possibly update the language (if needed)
                - kwargs    : forwarded to `self.get_model().change_vocabulary(...)`
            Returns :
                - new_encoder   : the new TextEncoder instance initialized
        """
        from models import is_model_name, get_pretrained
        
        if lang is None: lang = self.lang
        old_vocab   = self.vocab
        
        if isinstance(new_encoder, str):
            if is_model_name(new_encoder):
                new_encoder = get_pretrained(new_encoder)
            else:
                new_encoder = get_encoder(lang = lang, text_encoder = new_encoder)
        
        if isinstance(new_encoder, BaseTextModel):
            self.lang   = new_encoder.lang
            self.text_encoder   = new_encoder.text_encoder
            self.text_encoder_config    = new_encoder.text_encoder_config
        elif isinstance(new_encoder, TextEncoder):
            self.lang   = lang
            self.text_encoder   = new_encoder
            self.text_encoder_config    = {}
        else:
            raise ValueError('Unsupported TextEncoder (type {}) : {}'.format(
                type(new_encoder), new_encoder
            ))
        
        self.save_text_encoder(force = True)
        
        if hasattr(self.get_model(), 'change_vocabulary') and self.vocab != old_vocab:
            self.get_model().change_vocabulary(
                self.vocab,
                old_vocab = old_vocab,
                ** self.model_tokens,
                ** kwargs
            )
            
            self.save()
        
        return self.text_encoder
    
    def clean_text(self, text, * args, ** kwargs):
        """ Equivalent to `self.text_encoder.clean_text(...)` """
        return self.text_encoder.clean_text(text, * args, ** kwargs)

    def encode_text(self, text, * args, ** kwargs):
        """ Equivalent to `self.text_encoder.encode(...)` """
        return self.text_encoder.encode(text, * args, ** kwargs)
    
    def format_text(self, text_format, _keys = None, _values = None, ** kwargs):
        """
            Equivalent to `self.text_encoder.format(...)`
            _keys and _values can be list of names / values to be used as kwargs. Useful for `tf.numpy_function` call which requires positional arguments instead of kwargs.
            If they are provided : `kwargs = {k : v for k, v in zip(_keys, _values)}`
        """
        if not kwargs: kwargs = convert_to_str({k : v for k, v in zip(_keys, _values)})
        return self.text_encoder.format(text_format, ** kwargs)
    
    def split_and_format_text(self, text_format, split_key, _keys = None, _values = None,
                              max_length = -1, split_mode = 'sentence', ** kwargs):
        """
            Equivalent to `self.text_encoder.split_and_format(...)`
            _keys and _values can be list of names / values to be used as kwargs. Useful for `tf.numpy_function` call which requires positional arguments instead of kwargs.
            If they are provided : `kwargs = {k : v for k, v in zip(_keys, _values)}`
        """
        split_mode = convert_to_str(split_mode)
        if not kwargs:
            kwargs = convert_to_str({k : v for k, v in zip(_keys, _values)})

        encoded_parts = self.text_encoder.split_and_format(
            pattern     = text_format,
            split_key   = split_key,
            max_length  = max_length,
            split_mode  = split_mode,
            ** kwargs
        )
        return (
            pad_batch(encoded_parts, pad_value = self.blank_token_idx, dtype = np.int32),
            np.array([len(e) for e in encoded_parts], dtype = np.int32)
        )

    def multi_encode(self, texts, ** kwargs):
        """
            Encodes multiple texts and returns the 2-D ndarray (padded texts) with their length
            
            Arguments :
                - texts     : (list of) text to encode
                - kwargs    : forwarded to each `self.encode_text` call
            Return  :
                - encoded_texts : 2-D array of padded encoded texts (padded with `self.blank_token_idx`)
                - lengths       : 1-D array with the effective text length (at each index)
        """
        texts   = convert_to_str(texts)
        if not isinstance(texts, (list, tuple, np.ndarray)): texts = [texts]
        
        encoded = [
            self.encode_text(t, ** kwargs) for t in texts
        ]
        
        return (
            pad_batch(encoded, pad_value = self.blank_token_idx, dtype = np.int32),
            np.array([len(e) for e in encoded], dtype = np.int32)
        )

    def multi_format(self, text_format, _keys = None, _values = None, ** kwargs):
        """
            Formats multiple texts and returns the 2-D ndarray (padded texts) with their length
            
            Arguments :
                - text_format   : the text_format to use
                - kwargs    : keys are names (for formatting) and values are list of texts
            Return  :
                - encoded_texts : 2-D array of padded encoded (formatted) texts (padded with `self.blank_token_idx`)
                - lengths       : 1-D array with the effective text length (at each index)
            
            See `help(self.format_text)` for information about `_keys` and `_values`
        """
        if not kwargs:
            kwargs = convert_to_str({k : v for k, v in zip(_keys, _values)})
        kwargs = {k : v if isinstance(v, (list, tuple, np.ndarray)) else [v] for k, v in kwargs.items()}
        
        nb_texts    = max([len(v) for v in kwargs.values()])
        encoded     = [
            self.format_text(text_format, ** {k : v[i] for k, v in kwargs.items()})[0]
            for i in range(nb_texts)
        ]

        return (
            pad_batch(encoded, pad_value = self.blank_token_idx, dtype = np.int32),
            np.array([len(e) for e in encoded], dtype = np.int32)
        )

    def multi_split_and_format(self, text_format, split_key, _keys = None, _values = None,
                               max_length = -1, split_mode = 'sentence', ** kwargs):
        """
            Splits and ormats multiple texts and returns the 3-D ndarray (padded splitted texts) with their lengths
            
            Arguments :
                - text_format   : the text_format to use
                - kwargs    : keys are names (for formatting) and values are list of texts
            Return  :
                - encoded_texts : 3-D array of padded encoded (formatted) texts (padded with `self.blank_token_idx`)
                - lengths       : 2-D array with the effective text length (at each index)
            
            See `help(self.split_and_format_text)` for information about `_keys` and `_values`
        """
        split_mode = convert_to_str(split_mode)
        if not kwargs:
            kwargs = convert_to_str({k : v for k, v in zip(_keys, _values)})
        kwargs = {k : v if isinstance(v, (list, tuple, np.ndarray)) else [v] for k, v in kwargs.items()}

        nb_texts    = max([len(v) for v in kwargs.values()])
        encoded, lengths    = list(zip(* [
            self.split_and_format_text(
                text_format,
                split_key   = split_key,
                max_length  = max_length,
                split_mode  = split_mode, 
                ** {k : v[i] for k, v in kwargs.items()}
            ) for i in range(nb_texts)
        ]))

        return pad_batch(encoded, pad_value = self.blank_token_idx), pad_batch(lengths, pad_value = 0)

    def decode_text(self, encoded, ** kwargs):
        return self.text_encoder.decode(encoded, ** kwargs)
    
    def tf_encode_text(self, text, default_key = 'text'):
        """ Calls `self.encode_text` inside a `tf.numpy_function` (to be enable graph computation) """
        if isinstance(text, (dict, pd.Series)): text = text[default_key]
        
        encoded_text = tf.numpy_function(
            self.encode_text, [text], Tout = tf.int32
        )
        encoded_text.set_shape([None])
        
        return encoded_text

    def tf_format_text(self, text_format, data, keys = ['text'], return_types = False, ** kwargs):
        """
            Calls `self.format_text` inside a `tf.numpy_function` (to be enable graph computation)
            
            Arguments :
                - text_format   : the text format to use
                - data      : either list of data or dict / pd.Series
                    if dict, it is transformed into a list by taking keys in `keys`
                - return_types  : whether to return the types or not
                - kwargs    : forwarded to `_get_key_names`
            Returns :
                - encoded_text  : 1-D `tf.Tensor`, the result of `self.format_text`
                - token_types   : (if return_types), the 2nd return value of `self.format_text`
            
            Note that you can specify a `keys_mapping` argument to map `keys` to another name to match `format_text`. It is useful if the format's names do not match the `data`'s key names.
            For instance if `text_format = '{text}'` but the `data`'s key you want to use is 'label', you can set `keys = ['label'], keys_mapping = ['text']`, which will call :
            `self.format_text('{text}', text = data['label'])`
        """
        keys, values = _get_key_value_format(data, keys = keys, ** kwargs)
        
        encoded_text, token_types = tf.numpy_function(
            self.format_text, [text_format, keys, values], Tout = [tf.int32, tf.int32]
        )
        encoded_text.set_shape([None])
        
        if return_types:
            token_types.set_shape([None])
            return encoded_text, token_types
        
        return encoded_text
    
    def tf_split_and_format_text(self, text_format, split_key, data, keys = ['text'],
                                 max_sentences_length = -1, split_mode = 'sentence', ** kwargs):
        keys, values = _get_key_value_format(data, keys = keys, ** kwargs)

        encoded_text, lengths = tf.numpy_function(
            self.split_and_format_text,
            [text_format, split_key, keys, values, max_sentences_length, split_mode],
            Tout = [tf.int32, tf.int32]
        )
        encoded_text.set_shape([None, None])
        lengths.set_shape([None])
        
        if logger.isEnabledFor(logging.DEBUG):
            tf.print('Split + format lengths (total :', tf.reduce_sum(lengths), ') :', lengths)
        
        return encoded_text
    
    def tf_multi_encode(self, text, default_key = 'text', ** kwargs):
        """
            Calls `self.multi_encode` inside a `tf.numpy_function`
            kwargs are passed to `utils.text.filter_texts` allowing to filter the resulting texts based on multiple criteria. 
            **Warning** : if no texts respects the filtering criteria, the 1st output shape's dimension can be 0 !
        """
        if isinstance(text, (dict, pd.Series)): text = text[default_key]

        encoded_texts, lengths = tf.numpy_function(
            self.multi_encode, [text], Tout = [tf.int32, tf.int32]
        )
        encoded_texts.set_shape([None, None])
        lengths.set_shape([None])
        
        return filter_texts(encoded_texts, lengths, ** kwargs)

    def tf_multi_format(self, text_format, data, keys = ['text'], ** kwargs):
        """
            Calls `self.multi_format` inside a `tf.numpy_function`
            kwargs are passed to `utils.text.filter_texts` allowing to filter the resulting texts based on multiple criteria. 
            **Warning** : if no texts respects the filtering criteria, the 1st output shape's dimension can be 0 !
        """
        keys, values = _get_key_value_format(data, keys = keys, ** kwargs)

        encoded_texts, lengths = tf.numpy_function(
            self.multi_format, [text_format, keys, values], Tout = [tf.int32, tf.int32]
        )
        encoded_texts.set_shape([None, None])
        lengths.set_shape([None])
        
        return filter_texts(encoded_texts, lengths, ** kwargs)

    def tf_multi_split_and_format(self,
                                  text_format,
                                  split_key,
                                  data,
                                  keys  = ['text'],
                                  max_sentences_length  = -1,
                                  split_mode    = 'sentence',
                                  max_sentences = -1,
                                  ** kwargs
                                 ):
        keys, values = _get_key_value_format(data, keys = keys, ** kwargs)
        
        encoded_texts, lengths = tf.numpy_function(
            self.multi_split_and_format,
            [text_format, split_key, keys, values, max_sentences_length, split_mode],
            Tout = [tf.int32, tf.int32]
        )
        encoded_texts.set_shape([None, None, None])
        lengths.set_shape([None, None])
        
        if max_sentences > 0 and tf.shape(encoded_texts)[1] > max_sentences:
            nb_sent = tf.reduce_sum(tf.cast(lengths > 0, tf.int32), axis = -1)
            valids  = nb_sent <= max_sentences
            
            if logger.isEnabledFor(logging.DEBUG):
                tf.print('Valid texts (max_sentences filtering) :', valids)
            
            encoded_texts   = tf.boolean_mask(encoded_texts, valids)
            lengths         = tf.boolean_mask(lengths, valids)
        
        encoded_texts = filter_texts(encoded_texts, lengths, ** kwargs)
        
        if logger.isEnabledFor(logging.DEBUG):
            tf.print('Lengths (total :', tf.reduce_sum(lengths), ') :', lengths)
        
        encoded_texts   = tf.reshape(encoded_texts, [-1, tf.shape(encoded_texts)[-1]])
        encoded_texts   = tf.boolean_mask(encoded_texts, encoded_texts[0] != self.blank_token_idx)

        if logger.isEnabledFor(logging.DEBUG):
            tf.print('Multi input shape :', tf.shape(encoded_texts))

        return encoded_texts

    def augment_text(self, tokens, min_idx = 1, max_idx = -1, nb_mask = None,
                     min_mask_length = None, max_mask_length = None):
        if nb_mask is None: nb_mask = self.nb_mask
        if min_mask_length is None: min_mask_length = self.min_mask_length
        if max_mask_length is None: max_mask_length = self.max_mask_length
        
        tokens = tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: random_mask(
                tokens, self.mask_token_idx,
                min_idx = min_idx, max_idx = max_idx,
                nb_mask = nb_mask,
                min_mask_length = min_mask_length,
                max_mask_length = max_mask_length
            ),
            lambda: tokens
        )
        return tokens

    def save_text_encoder(self, filename = None, force = False):
        if filename is None: filename = self.text_encoder_file
        
        if not os.path.exists(filename) or force:
            self.text_encoder.save_to_file(filename)
        return filename
    
    def get_config_text(self, * args, ** kwargs):
        # Saving text encoder and mel fn (if needed)
        self.save_text_encoder()
        
        return {
            'lang'      : self.lang,
            'text_encoder'  : self.text_encoder_file
        }
        
