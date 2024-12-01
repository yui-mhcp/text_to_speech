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

import os
import enum
import glob
import json
import logging
import regex as re
import numpy as np

from keras import tree
from datetime import datetime
from functools import cached_property, cache

from utils import dump_json, load_json, pad_batch, get_enum_item, convert_to_str, download_file, is_dataframe
from utils.keras_utils import TensorSpec, execute_eagerly, ops
from utils.text import cleaners as cleaners_module
from .ctc_decoder import ctc_decode
from .byte_pair_encoding import bytes_to_unicode, bpe
from .text_processing import filter_texts, process_model_output
from .text_splitter import split_sentences, split_and_join

logger  = logging.getLogger(__name__)

text_signature          = TensorSpec(shape = (None, ), dtype = 'int32', name = 'text')
multi_text_signature    = TensorSpec(shape = (None, None), dtype = 'int32', name = 'text')
text_length_signature   = TensorSpec(shape = (None, ), dtype = 'int32', name = 'length')
multi_text_length_signature = TensorSpec(shape = (None, None), dtype = 'int32', name = 'length')

_clip_bpe_url   = 'https://raw.githubusercontent.com/openai/CLIP/master/clip/bpe_simple_vocab_16e6.txt.gz'
_whisper_url   = 'https://raw.githubusercontent.com/openai/whisper/master/whisper/assets/{}/{}'

_gpt_pattern    = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_clip_pattern   = r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
_llama_pattern  = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

WHISPER_LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}
        
class TextEncoderLevel(enum.IntEnum):
    CHAR    = 0
    TOKEN   = 1
    BPE     = 1
    WORD    = 2

class TextEncoder(object):
    def __init__(self,
                 vocab,
                 level,
                 template   = None,
                 vocab_size     = None,
                 
                 lstrip     = False,
                 rstrip     = False,
                 cleaners       = [],
                 split_pattern  = None,
                 bpe_pairs      = None,
                 byte_encoder   = None,
                 bpe_end_of_word    = None,
                 
                 pad_token      = '',       # blank token
                 ukn_token      = None,     # for unknown toekn (if not provided, skip them)
                 sep_token      = None,
                 mask_token     = None,
                 sos_token      = '[SOS]',  # Start Of Sequence
                 eos_token      = '[EOS]',  # End Of Sequence
                 additional_tokens  = None,
                 
                 sub_word_prefix    = '',   # Add for inner sub-word part
                 use_sos_and_eos    = False,
                 add_special_tokens_at_end  = True,
                 
                 name           = 'Text encoder'
                ):
        """
            Constructor for the `TextEncoder` class
            
            Arguments :
                - vocab     : list of tokens (words, sub-words or characters)
                - level     : tokenization level (char, token, word)
                - vocab_size    : special vocab_size (by default len(vocab))
                
                - lstrip / rstrip   : whether to strip left / right the text
                - cleaners      : cleaners to use for text-cleaning (see `cleaners.get_cleaners_fn`)
                - split_pattern : regex pattern used to split the text
                - bpe_pairs     : list containing the byte-pair encoding (BPE) pairs
                - byte_encoder  : mapping (`dict`) for byte-encoding
                - bpe_end_of_word   : special character to add at the end of words
                
                - pad_token     : token to add for padding (also called `blank_token`)
                - ukn_token     : token to use for unknown (out-of vocabulary) tokens
                - sep_token     : special token to separate different parts
                - mask_token    : special token to mask some tokens (e.g., data augmentation or MLM)
                - {sos / eos}_token : start / end of sequence tokens
                
                - sub_word_prefix   : prefix to add at the start of sub-words (used in BERT)
                - use_sos_and_eos   : whether to add SOS and EOS at the encoded text extrema
                - add_special_tokens_at_end : whether to add special tokens at the end of the vocabulary (or at the beginning)
                
                - name      : special name for this encoder
        """
        level = get_enum_item(level, TextEncoderLevel)
        if level != TextEncoderLevel.CHAR and 'detach_punctuation' not in cleaners and split_pattern is None and type(self) == TextEncoder:
            logger.warning("When using token / word-level tokenizer, it can be useful to add 'detach_punctuation' in cleaners")
        
        self.name       = name
        self._vocab     = list(vocab)
        self.level      = level
        self.template   = template
        
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.cleaners   = cleaners
        self.split_pattern  = split_pattern
        self.bpe_pairs      = bpe_pairs if bpe_pairs is None else [tuple(pair) for pair in bpe_pairs]
        self.byte_encoder   = byte_encoder if byte_encoder is None else {
            int(k) : v for k, v in byte_encoder.items()
        }
        self.byte_encoder_inv   = None if byte_encoder is None else {
            v : k for k, v in self.byte_encoder.items()
        }
        self.bpe_end_of_word    = bpe_end_of_word
        
        self.pad_token  = pad_token
        self.sep_token  = sep_token
        self.ukn_token  = ukn_token
        self.sos_token  = sos_token
        self.eos_token  = eos_token
        self.mask_token = mask_token
        self.sub_word_prefix    = sub_word_prefix
        self.use_sos_and_eos    = use_sos_and_eos
        self.add_special_tokens_at_end  = add_special_tokens_at_end

        if additional_tokens is None:            additional_tokens = []
        elif isinstance(additional_tokens, str): additional_tokens = [additional_tokens]
        if isinstance(additional_tokens, (list, tuple)):
            additional_tokens = {
                _create_token_name(tok) : tok for tok in additional_tokens
            }
        self.additional_tokens  = additional_tokens
        
        self.splitter   = re.compile(split_pattern) if split_pattern is not None else None
        if bpe_pairs is not None and not isinstance(bpe_pairs, dict):
            bpe_pairs   = {tuple(pair) : i for i, pair in enumerate(bpe_pairs)}
        self.bpe_ranks  = bpe_pairs
        self.cleaners_fn    = cleaners_module.get_cleaners_fn(cleaners)
        
        self._special_tokens    = list(self.tokens.values())
        self._bpe_cache     = {}
        self._symbol_to_id  = {}
        self._id_to_symbol  = {}
        self.__build_indexes(vocab_size, add_special_tokens_at_end)
        self._cleaned_tokens    = {
            self.clean_text(token) : token for token in self._special_tokens
        }
        
        for name, token in self.additional_tokens.items():
            setattr(self, name, token)
            setattr(self, name + '_idx', self[token])

    def __build_indexes(self, vocab_size = None, add_special_tokens_at_end = True):
        def _add_symbols(symbols):
            for s in symbols:
                self._symbol_to_id.setdefault(s, len(self._symbol_to_id))
        
        special_tokens = list(self.tokens.values())
        
        if vocab_size is not None:
            for spe in special_tokens:
                if spe not in self._vocab: vocab_size -= 1
                
            if len(self._vocab) > vocab_size:
                logger.warning(
                    'Truncating vocab to size {} (from {})'.format(len(self._vocab), vocab_size)
                )
                self._vocab = self._vocab[: vocab_size]
            elif len(self._vocab) < vocab_size:
                self._vocab += ['ukn_{}'.format(i) for i in range(vocab_size - len(self._vocab))]
        
        self._symbol_to_id  = {}
        if not add_special_tokens_at_end:
            # Add special tokens (if required)
            _add_symbols(special_tokens)
        # Build `symbol to id` pairs
        _add_symbols(self._vocab)

        if add_special_tokens_at_end:
            # Add special tokens (if required)
            _add_symbols(special_tokens)

        # Build ID --> symbol pairs
        self._id_to_symbol  = {idx : s for s, idx in self._symbol_to_id.items()}
        
        if self.bpe_pairs is not None and self.byte_encoder is None:
            self.byte_encoder = bytes_to_unicode()
        
    @property
    def vocab_size(self):
        return len(self._id_to_symbol)
    
    @property
    def vocab(self):
        return sorted(self._symbol_to_id.keys(), key = self._symbol_to_id.get)
    
    @property
    def word_split(self):
        return self.level != TextEncoderLevel.CHAR and self.splitter is None
    
    @cached_property
    def special_tokens(self):
        return set(v for k, v in self.get_config().items() if k.endswith('_token') and v is not None)
    
    @cached_property
    def tokens(self):
        tokens = {
            k : v for k, v in self.get_config().items()
            if k.endswith('_token') and v is not None
        }
        tokens.pop('pad_token', None)
        if not self.use_sos_and_eos:
            tokens.pop('sos_token', None)
            tokens.pop('eos_token', None)
        tokens.update(self.additional_tokens)
        return tokens
    
    @cached_property
    def token_indexes(self):
        return {v : self._symbol_to_id[v] for k, v in self.tokens.items()}
    
    @property
    def sos_token_idx(self):
        return self._symbol_to_id[self.sos_token] if self.sos_token else -1
    
    @property
    def eos_token_idx(self):
        return self._symbol_to_id[self.eos_token] if self.use_sos_and_eos else -1
    
    @property
    def sep_token_idx(self):
        return self._symbol_to_id[self.sep_token] if self.sep_token is not None else -1
    
    @property
    def blank_token_idx(self):
        default = 0 if not self.use_sos_and_eos else self.eos_token_idx
        return self._symbol_to_id.get(self.pad_token, default)
    
    @property
    def ukn_token_idx(self):
        return self._symbol_to_id[self.ukn_token] if self.ukn_token is not None else -1
    
    @property
    def mask_token_idx(self):
        return self._symbol_to_id[self.mask_token] if self.mask_token is not None else -1
        
    @property
    def blank_token(self):
        return self._id_to_symbol[self.blank_token_idx]
    
    def __str__(self):
        des = "========== {} ==========\n".format(self.name)
        des += "Vocab (size = {}) : {}\n".format(
            len(self), self.vocab if self.vocab_size <= 50 else '[{}, ...]'.format(str(self.vocab[:50])[1:-1])
        )
        config = self.get_config()
        if self.template:
            des += 'Template :\n{}\n'.format(pretty_print_template(config.pop('template')))
        
        for k in ['name', 'vocab', 'bpe_pairs', 'byte_encoder']:
            config.pop(k)
        des += 'Config : {}'.format(json.dumps(config, indent = 2))
        return des
        
    def __len__(self):
        return self.vocab_size
    
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._id_to_symbol[idx]
        else:
            if idx not in self._symbol_to_id:
                res = self.encode(
                    idx, add_sos_and_eos = False, cleaned = True, return_type = 'list'
                )
                return res[0] if len(res) == 1 else res
            return self._symbol_to_id[idx]
    
    def __contains__(self, token):
        return token in self._symbol_to_id
    
    def _should_add_sos_and_eos(self,
                                *,
                                add_sos = None,
                                add_eos = None,
                                add_sos_and_eos = None,
                                ** _
                               ):
        if add_sos_and_eos is None: add_sos_and_eos = self.use_sos_and_eos
        if add_sos is None: add_sos = add_sos_and_eos
        if add_eos is None: add_eos = add_sos_and_eos
        
        return (add_sos and self.sos_token, add_eos and self.eos_token)
    
    def is_end_of_word(self, tokens, index, tokens_to_not_split = []):
        token = tokens[index]
        prev_token = None if index == 0 else tokens[index - 1]
        next_token = None if index == len(tokens) - 1 else tokens[index + 1]
        
        if self.level == TextEncoderLevel.WORD:
            return True
        elif self.level == TextEncoderLevel.CHAR:
            return next_token is None or not self[next_token].isalnum()
        elif self.sub_word_prefix:
            return not self[next_token].startswith(self.sub_word_prefix)
        else:
            if next_token is None: return True
            if hasattr(self, 'space_replacement'):
                _space = self.space_replacement
            else:
                _space = ' ' if self.byte_encoder is None else self.byte_encoder.get(32, ' ')
            return self[next_token].startswith(_space)
    
    def clean_text(self, text, tokens = {}, ** kwargs):
        """ Apply all cleaners to 'text' """
        if not isinstance(tokens, dict):
            tokens = {self.clean_text(token, ** kwargs) : token for token in tokens}
        
        return cleaners_module.clean_text(text, self.cleaners_fn, tokens = tokens, ** kwargs)
    
    def split_text(self, text, tokens = None, strip = True):
        """ Splits `text` into a list of tokens """
        if tokens is None:
            if self.splitter is not None:
                if strip:
                    text = cleaners_module.strip(text, lstrip = self.lstrip, rstrip = self.rstrip)
                return list(re.findall(self.splitter, text))
            return text.split() if self.word_split else list(text)
        
        parts = [text]
        for token in tokens:
            new_parts = []
            for part in parts:
                new_parts.extend(split_and_join(part, token))
            parts = new_parts

        splitted = []
        for part in parts:
            splitted.extend(self.split_text(part, strip = strip) if part not in tokens else [part])
        
        return splitted
    
    def _char_tokenize(self, token):
        return token
    
    def _word_tokenize(self, token):
        return token
    
    def _sub_word_tokenize(self, token):
        if isinstance(token, (list, tuple)):
            return tree.flatten([self._sub_word_tokenize(tok) for tok in token])
        
        start, valid = 0, True
        sub_tokens = []
        while start < len(token):
            end = len(token)
            while start < end:
                sub_token = token[start : end]
                    
                sub = sub_token if start == 0 else self.sub_word_prefix + sub_token

                if sub in self._symbol_to_id:
                    sub_tokens.append(sub)
                    break
                end -= 1
                
            if end <= start:
                valid = False
                break
                
            start += len(sub_token)
            
        if valid:
            return sub_tokens
        return self.ukn_token
    
    def _bpe_tokenize(self, token):
        if isinstance(token, (list, tuple)):
            return tree.flatten([self._bpe_tokenize(tok) for tok in token])
        
        if token not in self._bpe_cache:
            bpe_token = ''.join([
                self.byte_encoder[b] for b in token.encode('utf-8') if b in self.byte_encoder
            ])
            bpe_token = bpe(bpe_token, self.bpe_ranks, end_of_word = self.bpe_end_of_word)
            self._bpe_cache[token] = bpe_token
        
        return self._bpe_cache[token]
    
    def _tokenize(self, token):
        if len(token) == 0: return None
        
        if self.level == TextEncoderLevel.CHAR:
            return self._char_tokenize(token)
        elif self.level == TextEncoderLevel.WORD:
            return self._word_tokenize(token)
        elif self.level == TextEncoderLevel.TOKEN and self.bpe_pairs is None:
            return self._sub_word_tokenize(token)
        elif self.level == TextEncoderLevel.TOKEN and self.bpe_pairs is not None:
            return self._bpe_tokenize(token)
    
    def tokenize(self, text, cleaned = False, ** kwargs):
        if not cleaned:
            text = self.clean_text(text, self._cleaned_tokens, ** kwargs)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Cleaned text : {}'.format(text))

        splitted = self.split_text(text, self._special_tokens, strip = not cleaned)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Splitted text : {}'.format(splitted))

        tokens = []
        for part in splitted:
            tok = part if part in self._special_tokens else self._tokenize(part)
            if isinstance(tok, (list, tuple)):
                tokens.extend(tok)
            else:
                tokens.append(tok)
        
        return tokens
    
    @execute_eagerly(signature = text_signature, default_key = ('text', 'label'), numpy = True)
    def encode(self, text, *, return_type = 'np', ** kwargs):
        """
            Encode text in np.ndarray
            Arguments :
                - text : text to encode
                    Can be a string, dict, list / tuple / np.ndarray / Tensor of dict or str
            Return : 
                - np.ndarray    : the encoded text (or list of np.ndarrays if multiple textwere given)
            
            1) Clean the text
            2) Split it either by word or by characters
            3) Convert all tokens to its corresponding id (with the `self.tokenize` method)
            4) If necessary, add [sos, eos] tokens 
        """
        if is_dataframe(text):          text = text['text'].values
        elif isinstance(text, dict):    text = text['text']
        text = convert_to_str(text)
        
        if isinstance(text, (list, tuple)):
            return [self.encode(t, return_type = return_type, ** kwargs) for t in text]
        
        add_sos, add_eos = self._should_add_sos_and_eos(** kwargs)
        
        tokens  = self.tokenize(text, ** kwargs)

        tokens = [
            self._symbol_to_id.get(token, self.ukn_token_idx) for token in tokens
        ]
        tokens = [token for token in tokens if token not in (None, -1)]

        if (add_sos) and (len(tokens) == 0 or tokens[0] != self.sos_token_idx):
            tokens.insert(0, self.sos_token_idx)
        if (add_eos) and (len(tokens) == 0 or tokens[-1] != self.eos_token_idx):
            tokens.append(self.eos_token_idx)
        
        if return_type == 'list': return tokens
        elif return_type == 'np': return np.array(tokens, dtype = np.int32)
        elif return_type == 'tf': return ops.convert_to_tf_tensor(tokens, dtype = 'int32')
        elif return_type == 'tensor':   return ops.convert_to_tensor(tokens, dtype = 'int32')
        else:   raise ValueError("Unknown `return_type` : {}".format(return_type))

    def decode(self,
               sequence,
               skip_padding = True,
               remove_tokens    = False,
               attach_punctuation   = True,
               ** _
              ):
        """ Decode a given np.ndarray by replacing each known id by its corresponding token """
        if hasattr(sequence, 'tokens'): sequence = process_model_output(sequence)
        
        if ops.is_tensor(sequence):     sequence = ops.convert_to_numpy(sequence)
        if isinstance(sequence, np.ndarray):
            if np.issubdtype(sequence.dtype, np.floating) and all(s > 0 for s in sequence.shape):
                sequence = np.argmax(sequence, axis = -1)
            
            if len(sequence.shape) > 1:
                return [self.decode(
                    s, skip_padding = skip_padding, attach_punctuation = attach_punctuation,
                    remove_tokens = remove_tokens
                ) for s in sequence]
        
        if isinstance(sequence, (list, tuple)) and not isinstance(sequence[0], (int, np.integer)):
            return [self.decode(
                s, skip_padding = skip_padding, attach_punctuation = attach_punctuation,
                remove_tokens = remove_tokens
            ) for s in sequence]
        
        sep = ' ' if self.word_split else ''

        tokens = [
            self._id_to_symbol[s] for s in sequence
            if s in self._id_to_symbol and s != self.blank_token_idx
        ]
        
        if skip_padding or remove_tokens:
            _special_tokens = self.special_tokens if remove_tokens else [self.blank_token]
            cleaned = []
            for token in tokens:
                if token not in _special_tokens: cleaned.append(token)
                if skip_padding and token == self.eos_token and len(cleaned) > 0: break
            tokens = cleaned
        
        text = sep.join(tokens)

        if self.byte_encoder is not None:
            text_bytes = [
                self.byte_encoder_inv.get(c, c) for c in text
            ]
            try:
                text = bytearray(text_bytes).decode('utf-8')
            except UnicodeDecodeError as e:
                pass
        
        if self.level == TextEncoderLevel.TOKEN and self.sub_word_prefix:
            text = text.replace(' ' + self.sub_word_prefix, '')
        if self.level == TextEncoderLevel.TOKEN and self.bpe_end_of_word:
            text = text.replace(self.bpe_end_of_word, ' ')

        if attach_punctuation:
            text = cleaners_module.attach_punctuation(text)
            
        return text
    
    def ctc_decode(self,
                   sequence,
                   lengths  = None,
                   method   = 'beam',
                   return_scores    = False,
                   ** kwargs
                  ):
        """ Decode a given np.ndarray by replacing each known id by its corresponding token """
        tokens, scores = ctc_decode(
            sequence,
            method  = method,
            lengths = lengths,
            blank_index = self.blank_token_idx
        )

        decoded = self.decode(tokens, ** kwargs)
        return decoded if not return_scores else (decoded, scores)

    def invert(self, text, ** kwargs):
        return self.decode(self.encode(text, ** kwargs))

    @execute_eagerly(signature = text_signature, numpy = True)
    def encode_template(self,
                        text = None,
                        *,
                        
                        template    = None,
                        
                        format  = None,
                        messages    = None,
                        system_prompt   = None,
                        
                        encode  = True,
                        add_eos = False,
                        add_generation_prompt   = True,
                        
                        ** kwargs
                       ):
        template = convert_to_str(template) if template else self.template
        kwargs   = convert_to_str(kwargs)

        if not messages:
            if not text and not format:
                raise RuntimeError('The `encode_template` requires either a list of `messages` ending by a `user` message, either a `text` or `format`, that will be used as user message')
                
            if format:
                text = apply_format(
                    format, text = text, ** kwargs, ** self.tokens
                )
        
            messages = {'role' : 'user', 'content' : text}
        
        if isinstance(messages, dict):  messages = [messages]
        elif isinstance(messages, str): messages = [{'role' : 'user', 'content' : messages}]
        elif not isinstance(messages, list) or any(not isinstance(msg, dict) for msg in messages):
            raise ValueError('Unsupported `messages` argument : {}'.format(messages))
        elif messages[-1]['role'] != 'user':
            assert text is not None, '`messages` must end with a "user" message'
            messages += [{'role' : 'user', 'content' : text}]
        
        if system_prompt and messages[0]['role'] != 'system':
            if '{' in system_prompt:
                system_prompt = apply_format(system_prompt, ** kwargs, ** self.tokens)
            messages = [{'role' : 'system', 'content' : system_prompt}] + messages
        
        kwargs['messages'] = messages
        
        if 'date_string' in template and 'date_string' not in kwargs:
            kwargs['date_string'] = datetime.now().strftime("%d %B %Y")
        
        text = apply_format(
            template, add_generation_prompt = add_generation_prompt, ** kwargs, ** self.tokens
        )
        return self.encode(text, add_eos = add_eos, ** kwargs) if encode else text

    @execute_eagerly(signature = [
        text_signature, TensorSpec(shape = (None, ), dtype = 'int32', name = 'types')
    ], numpy = True)
    def join(self, * sentences, sep_token = None, return_type = 'np', ** kwargs):
        kwargs.pop('text', None)
        if sep_token is None:   sep_token = self.sep_token
        add_sos, add_eos = self._should_add_sos_and_eos(** kwargs)

        encoded_parts = self.encode(
            sentences, add_sos_and_eos = False, return_type = 'list', ** kwargs
        )
        
        if sep_token is not None:
            sep_idx = self._symbol_to_id[sep_token]
            for part in encoded_parts[:-1]: part.append(sep_idx)

        encoded, ids = [], []
        for i, part in enumerate(encoded_parts):
            encoded.extend(part)
            ids.extend([i] * len(part))
        
        if (add_sos) and (len(encoded) == 0 or encoded[0] != self.sos_token_idx):
            encoded.insert(0, self.sos_token_idx)
            ids.insert(0, 0)
        if (add_eos) and (len(encoded) == 0 or encoded[-1] != self.eos_token_idx):
            encoded.append(self.eos_token_idx)
            ids.append(len(encoded_parts) - 1)
        
        if return_type == 'list': return encoded, ids
        elif return_type == 'np':
            return np.array(encoded, dtype = np.int32), np.array(ids, dtype = np.int32)
        elif return_type == 'tf':
            return (
                ops.convert_to_tf_tensor(encoded, dtype = 'int32'),
                ops.convert_to_tf_tensor(ids, dtype = 'int32')
            )
        elif return_type == 'tensor':
            return (
                ops.convert_to_tensor(encoded, dtype = 'int32'),
                ops.convert_to_tensor(ids, dtype = 'int32')
            )
        else:
            raise ValueError('Unknown `return_type` : {}'.format(return_type))

    @execute_eagerly(signature = text_signature, numpy = True)
    def format(self, pattern, ** kwargs):
        pattern = convert_to_str(pattern)
        kwargs  = convert_to_str(kwargs)

        if not isinstance(pattern, (list, tuple)): pattern = [pattern]
        
        formatted = [
            apply_format(pat, ** kwargs, ** self.tokens) for pat in pattern
        ]
        
        return self.join(* formatted, ** kwargs)[0]
    
    def split_encoded(self,
                      encoded,
                      max_length,
                      split_level       = 'word',
                      encoded_not_split = [],
                      ** kwargs
                     ):
        if split_level == 'token':  can_split   = lambda * a, ** kw: True
        elif split_level == 'word': can_split   = self.is_end_of_word
        
        start, parts, _iter = 0, [], 0
        while start < len(encoded) and _iter < 2 * len(encoded) / max_length:
            end = min(start + max_length, len(encoded))
            if not can_split(encoded, end - 1, encoded_not_split):
                new_end = _get_split_index(
                    encoded, start, end - 1, can_split
                ) + 1
                if new_end > start and new_end - start <= max_length:
                    end = new_end
            
            parts.append(encoded[start : end])
            start = end
            _iter += 1
        
        return parts
    
    @execute_eagerly(signature = [multi_text_signature, text_length_signature], numpy = True)
    def split(self,
              text,
              max_length,
              *,
              
              prefix    = None,
              suffix    = None,
              sep_token = None,
              
              return_type   = 'np',
              
              split_level   = 'sentence',
              tokens_to_not_split   = None,
              
              ** kwargs
             ):
        """
            Encodes then splits `text` such that each part has at most `max_length` tokens
            
            The `split_level` argument defines the unit to keep intact :
                - token : simple token-based split
                - word  : avoid splitting at the middle of a word
                - sentence  : avoid splitting a sentence (sentences are delimited by basic punctuation matching)
            
            **WARNING** : Only the "token" level ensures that each part is shorter than `max_length` ! If there is no possibility to split a part at the expected level, it may become longer than `max_length`
            
            Example :
            ```python
            text    = "Hello World !"
            max_length  = 5
            # In practice, an encoded string is a list of int (the token ids)
            # In the example, the encoded is simply the list of characters for simplicity of reading
            encoded = ['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', ' ', '!']
            
            # -> [['H', 'e', 'l', 'l', 'o'], [' ', 'W', 'o', 'r', 'l'], ['d', ' ', '!']]
            print(split(encoded, split_level = 'token', max_length = max_length))
            
            # -> [['H', 'e', 'l', 'l', 'o'], [' ', 'W', 'o', 'r', 'l', 'd'], [' ', '!']]
            print(split(encoded, split_level = 'word', max_length = max_length))

            # -> [['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', ' ', '!']]
            print(split(encoded, split_level = 'word', max_length = max_length))
            ```
        """
        assert split_level in ('token', 'word', 'sentence')

        if sep_token is None: sep_token = self.sep_token
        if tokens_to_not_split is None: tokens_to_not_split = []
        else:
            tokens_to_not_split = convert_to_str(tokens_to_not_split)
            if not isinstance(tokens_to_not_split, list):
                tokens_to_not_split = [tokens_to_not_split]
        
        add_sos, add_eos = self._should_add_sos_and_eos(** kwargs)

        encoded_not_split = [self.encode(
            ' ' + tok.trip() + ' ', add_sos_and_eos = False, return_type = 'list'
        ) for tok in tokens_to_not_split]

        sep_token_idx = self[sep_token] if sep_token else -1
        
        prefix = self.encode(
            prefix, add_sos_and_eos = False, return_type = 'list'
        ) if prefix is not None else []
        suffix = self.encode(
            suffix, add_sos_and_eos = False, return_type = 'list'
        ) if suffix is not None else []
        
        if len(prefix) > 0 and sep_token_idx != -1 and sep_token_idx != prefix[-1]:
            prefix.append(sep_token_idx)
        if len(suffix) > 0 and sep_token_idx != -1 and sep_token_idx != suffix[0]:
            suffix.insert(0, sep_token_idx)
        
        if add_sos and (not prefix or prefix[0] != self.sos_token_idx):
            prefix.insert(0, self.sos_token_idx)
        if add_eos and (not suffix or suffix[-1] != self.eos_token_idx):
            suffix.append(self.eos_token_idx)
        
        max_length  = max(1, max_length - len(prefix) - len(suffix))
        
        if split_level != 'sentence':
            parts = self.split_encoded(
                self.encode(text, add_sos_and_eos = False, return_type = 'list', ** kwargs),
                max_length  = max_length,
                split_level = split_level,
                encoded_not_split   = encoded_not_split,
                ** kwargs
            )
        else:
            text    = self.clean_text(text)
            sentences   = split_sentences(text)
            encoded_sentences   = [self.encode(
                sent, cleaned = True, add_sos_and_eos = False, return_type = 'list', ** kwargs
            ) for sent in sentences]
            
            parts, sents, length = [], [], 0
            for i, (sent, enc) in enumerate(zip(sentences, encoded_sentences)):
                if length + len(enc) >= (max_length - len(sents)) and length > 0:
                    parts.append(self.encode(
                        ' '.join(sents), cleaned = True, add_sos_and_eos = False,
                        return_type = 'list', ** kwargs
                    ) if length != len(encoded_sentences[i - 1]) else encoded_sentences[i - 1])
                    sents, length = [], 0
                
                if len(enc) >= max_length:
                    parts.extend(self.split_encoded(
                        enc,
                        max_length  = max_length,
                        split_level = 'word',
                        encoded_not_split   = encoded_not_split,
                        ** kwargs
                    ))
                else:
                    sents.append(sent)
                    length += len(enc)
            
            if length > 0:
                parts.append(self.encode(
                    ' '.join(sents), cleaned = True, add_sos_and_eos = False, return_type = 'list'
                ) if length != len(encoded_sentences[-1]) else encoded_sentences[-1])
        
        
        parts = [prefix + p + suffix for p in parts]
        lengths = [len(part) for part in parts]
        
        if return_type == 'list': return parts, lengths
        
        parts = pad_batch(parts, self.blank_token_idx, dtype = np.int32)
        if return_type == 'np':
            return (
                parts, np.array(lengths, dtype = np.int32)
            )
        elif return_type == 'tf':
            return (
                ops.convert_to_tf_tensor(parts, 'int32'),
                ops.convert_to_tf_tensor(lengths, dtype = 'int32')
            )
        elif return_type == 'tensor':
            return (
                ops.convert_to_tensor(parts, dtype = 'int32'),
                ops.convert_to_tensor(lengths, dtype = 'int32')
            )
        else:
            raise ValueError('Unknown `return_type` : {}'.format(return_type))
    
    @execute_eagerly(signature = [multi_text_signature, text_length_signature], numpy = True)
    def split_and_format(self, pattern, split_key, max_length, ** kwargs):
        pattern     = convert_to_str(pattern)
        split_key   = convert_to_str(split_key)
        kwargs      = convert_to_str(kwargs)

        if not isinstance(pattern, (list, tuple)): pattern = [pattern]
        pattern     = '{sep_token}'.join(pattern)
        splitted    = pattern.split('{' + split_key + '}')
        
        assert len(splitted) == 2, '`pattern` {} is invalid for `split_key` {} !'.format(
            pattern, split_key
        )
        
        prefix  = splitted[0].format(** {** self.tokens, ** kwargs})
        suffix  = splitted[1].format(** {** self.tokens, ** kwargs})
        
        return self.split(
            kwargs.pop(split_key), max_length, prefix = prefix, suffix = suffix, ** kwargs
        )
    
    def extract_sentence(self, tokens, idx, punct = '.?!', ** kwargs):
        def _is_start_of_sentence(tok):
            if tok == self.sos_token_idx: return True
            return any([final_punct in self._id_to_symbol.get(tok, None) for final_punct in punct])

        def _is_end_of_sentence(tok):
            if tok == self.eos_token_idx: return True
            return any([final_punct in self._id_to_symbol.get(tok, None) for final_punct in punct])

        if hasattr(tokens, 'numpy'): tokens = tokens.numpy()

        start, end = idx, idx + 1

        while start >= 0 and not _is_start_of_sentence(tokens[start]): start -= 1
        while end < len(tokens) and not _is_end_of_sentence(tokens[end]): end += 1
        start += 1
        end += 1

        return self.decode(tokens[start : end], ** kwargs).strip(), start, end

    def distance(self, hypothesis, truth, method = 'edit', ** kwargs):
        """ Compute the levenschtein distance between hypothesis and truth """
        from utils.distance import distance
        
        if isinstance(hypothesis, str) and not isinstance(truth, str):
            hypothesis = self.encode(hypothesis)
        if isinstance(truth, str) and not isinstance(hypothesis, str):
            truth = self.encode(truth)
        
        kwargs.setdefault('insertion_cost', {}).setdefault(self.blank_token, 0)
        kwargs.setdefault('deletion_cost', {}).setdefault(self.blank_token, 0)
        
        return distance(hypothesis, truth, method = method, ** kwargs)
    
    def save_to_file(self, filename):
        """ Saves `self.config` to `filename` (`json` format) """
        dump_json(filename, self.get_config(), indent = 4)

        return filename
    
    def get_config(self):
        return {
            'name'      : self.name,
            'vocab'     : self._vocab,
            'level'     : self.level,
            'template'  : self.template,
            
            'lstrip'    : self.lstrip,
            'rstrip'    : self.rstrip,
            'cleaners'  : self.cleaners,
            'split_pattern' : self.split_pattern,
            'bpe_pairs' : self.bpe_pairs,
            'byte_encoder'  : self.byte_encoder,
            'bpe_end_of_word'   : self.bpe_end_of_word,
            
            'pad_token' : self.pad_token,
            'sep_token' : self.sep_token,
            'ukn_token' : self.ukn_token,
            'sos_token' : self.sos_token,
            'eos_token' : self.eos_token,
            'mask_token'    : self.mask_token,
            'additional_tokens' : self.additional_tokens,
            
            'sub_word_prefix'   : self.sub_word_prefix,
            'use_sos_and_eos'   : self.use_sos_and_eos,
            'add_special_tokens_at_end' : self.add_special_tokens_at_end
        }
    
    @classmethod
    def load_from_file(cls, filename):
        config = load_json(filename)
        
        _update = False
        if 'word_level' in config:  # for retro-compatibility
            config['level'] = TextEncoderLevel.CHAR if not config.pop('word_level') else TextEncoderLevel.WORD
            _update = True
        
        if 'tokenizer' in config:
            from .sentencepiece_encoder import SentencePieceTextEncoder
            cls = SentencePieceTextEncoder
        
        instance = cls(** config)
        if _update: instance.save_to_file(filename) # re-save encoder with updated config
        
        return instance

    @classmethod
    def from_transformers_pretrained(cls, name, ** kwargs):
        def get_vocab_from_encoder(tokenizer):
            vocab = tokenizer.encoder if hasattr(tokenizer, 'encoder') else tokenizer.vocab
            specials = [w for w in tokenizer.all_special_tokens if w not in vocab]
            return list(sorted(vocab, key = vocab.get)) + specials
        
        from transformers import AutoTokenizer, BertTokenizer, GPT2Tokenizer, BartTokenizer, BarthezTokenizer, GPT2TokenizerFast, PreTrainedTokenizerFast, T5Tokenizer, LlamaTokenizer, WhisperTokenizer
        
        pretrained = name
        if isinstance(name, str):
            pretrained = AutoTokenizer.from_pretrained(name, use_fast = False)
        
        # /!\ WARNING /!\ The original transformers tokenizers have the `remove_control`
        # cleaner but it reduces performances by 30% for really rare occurences. 
        # I suggest you to directlyremove it in your dataset loading (cf SQUAD loading function)
        #_default_cleaners = ['remove_control', 'remove_accents']
        _default_cleaners = []
        if hasattr(pretrained, 'do_lower_case') and pretrained.do_lower_case:
            _default_cleaners.insert(0, 'lowercase')

        if isinstance(pretrained, BertTokenizer):
            if 'uncased' in pretrained.name_or_path: _default_cleaners.append('remove_accents')
            kwargs.update({
                'vocab' : list(sorted(pretrained.vocab.keys(), key = pretrained.vocab.get)),
                'sub_word_prefix'   : '##',
                'cleaners'          : _default_cleaners + ['detach_punctuation', 'collapse_whitespace'],
                'sos_token'         : pretrained.cls_token,
                'eos_token'         : pretrained.sep_token
            })
        elif isinstance(pretrained, (GPT2Tokenizer, BartTokenizer, WhisperTokenizer)):
            # Note that RoBERTa and BART Tokenizer are subclasses of GPT2Tokenizer
            kwargs.update({
                'vocab' : get_vocab_from_encoder(pretrained),
                'lstrip'    : False,
                'rstrip'    : True,
                'split_pattern'     : _gpt_pattern,
                'cleaners'          : _default_cleaners,
                'bpe_pairs'         : list(pretrained.bpe_ranks.keys()),
                'byte_encoder'      : pretrained.byte_encoder,
                'sos_token'         : pretrained.bos_token,
                'eos_token'         : pretrained.eos_token
            })
        elif isinstance(pretrained, PreTrainedTokenizerFast):
            path = glob.glob(os.path.expanduser(
                '~/.cache/huggingface/hub/models--{}/**/**/tokenizer.json'.format(
                    pretrained.name_or_path.replace('/', '--')
                )
            ))
            if not path: raise RuntimeError('Unable to find the tokenizer.json file !')

            data    = load_json(path[0])
            # Note that RoBERTa and BART Tokenizer are subclasses of GPT2Tokenizer
            vocab = data['model']['vocab']
            for token in data['added_tokens']:
                vocab[token['content']] = token['id']
            
            if isinstance(vocab, dict): vocab = list(sorted(vocab.keys(), key = vocab.get))
            kwargs.update({
                'vocab' : vocab,
                'lstrip'    : False,
                'rstrip'    : False,
                'cleaners'  : _default_cleaners,
                'sos_token' : pretrained.bos_token,
                'eos_token' : pretrained.eos_token,
                'split_pattern'     : _llama_pattern,
                'bpe_pairs'         : [pair.split(' ') for pair in data['model']['merges']],
                'byte_encoder'      : bytes_to_unicode(),
                'additional_tokens' : [tok['content'] for tok in data.get('added_tokens', [])]
            })
        elif hasattr(pretrained, 'sp_model'):
            from utils.text.sentencepiece_encoder import SentencePieceTextEncoder
            cls = SentencePieceTextEncoder
            
            kwargs.update({
                'vocab' : [
                    pretrained.sp_model.id_to_piece(i)
                    for i in range(pretrained.sp_model.get_piece_size())
                ],
                'offset'    : getattr(pretrained, 'fairseq_offset', 0),
                'tokenizer' : pretrained.sp_model,
                'sub_word_prefix'   : '',
                'sos_token'         : pretrained.bos_token,
                'eos_token'         : pretrained.eos_token
            })
            tokens  = pretrained.added_tokens_decoder.values()
            kwargs['vocab'] = [v for v in kwargs['vocab'] if v not in tokens]
            for idx, tok in getattr(pretrained, 'added_tokens_decoder', {}).items():
                kwargs['vocab'].insert(idx, tok.content)
        # Common config
        kwargs.update({
            'level'         : TextEncoderLevel.TOKEN,
            'use_sos_and_eos'   : kwargs.get('sos_token', None) is not None or kwargs.get('eos_token', None) is not None,
            'sep_token'     : pretrained.sep_token,
            'ukn_token'     : pretrained.unk_token,
            'mask_token'    : pretrained.mask_token,
            'template'  : getattr(pretrained, 'chat_template', None)
        })
        if 'additional_tokens' not in kwargs and getattr(pretrained, 'additional_special_tokens', []):
            kwargs['additional_tokens'] = pretrained.additional_special_tokens
        
        if pretrained.pad_token is not None:    kwargs['pad_token'] = pretrained.pad_token
        elif kwargs.get('eos_token', None):     kwargs['pad_token'] = kwargs['eos_token']
        else:                                   kwargs['pad_token'] = kwargs['vocab'][0]

        return cls(** kwargs)
    
    @classmethod
    def from_clip_pretrained(cls, ** kwargs):
        import gzip
        
        from models import _pretrained_models_folder
        
        filename    = download_file(url = _clip_bpe_url, directory = os.path.join(
            _pretrained_models_folder, 'pretrained_weights'
        ))
        
        with gzip.open(filename) as file:
            pairs = file.read().decode('utf-8').split('\n')
        
        pairs = [tuple(pair.split()) for pair in pairs[1:49152-256-2+1]]
        
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for pair in pairs:
            vocab.append(''.join(pair))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        kwargs.update({
            'vocab' : vocab,
            'level' : 'bpe',
            'cleaners'  : {'name' : 'english_cleaners', 'to_lowercase' : False},
            'bpe_pairs' : {pair : i for i, pair in enumerate(pairs)},
            'byte_encoder'  : bytes_to_unicode(),
            'split_pattern' : _clip_pattern,
            'use_sos_and_eos'   : True,
            'pad_token'     : vocab[0],
            'sos_token'     : '<|startoftext|>',
            'eos_token'     : '<|endoftext|>',
            'bpe_end_of_word'   : '</w>'
        })
        return cls(** kwargs)

    @classmethod
    def from_whisper_pretrained(cls, multilingual = True, ** kwargs):
        return cls.from_transformers_pretrained('openai/whisper-base')


def pretty_print_template(template):
    if '\n' in template: return template
    
    indent = 0
    str_template = ''
    for part in template.split('{'):
        if not part: continue
        p_indent = indent
        if part[0] != '%':
            part = '{' + part
        elif part[2:].strip().startswith(('if', 'for')):
            indent += 1
        elif part[2:].strip().startswith(('elif', 'else')):
            p_indent -= 1
        elif part[2:].strip().startswith('end'):
            indent -= 1
            p_indent -= 1

        str_template += ' ' * p_indent * 4 + '{' + part + '\n'
    return str_template

def _create_token_name(token):
    name = ''.join([c for c in token.lower() if c.isalnum() or c == '_'])
    for suffix in ('_id', '_tag'):
        if name.endswith(suffix): name = name.replace(suffix, '_token')
    if 'token' not in name: name += '_token'
    return name

def _can_split(encoded, idx, encoded_not_split):
    if idx >= len(encoded): return 0, 0
    
    n_prev, n_next = 0, 0
    for ns in encoded_not_split:
        if encoded[idx] in ns:
            pos = ns.index(encoded[idx])
            if encoded[idx - pos : idx + len(ns) - pos] == ns:
                n_prev, n_next = max(n_prev, pos), max(n_next, len(ns) - pos - 1)
    return n_prev, n_next

def _get_split_index(encoded, last_end, idx, _can_split_fn, _decrease = True):
    new_idx = idx
    while last_end < new_idx < len(encoded):
        valid = _can_split_fn(encoded, new_idx, ())
        if valid:       return new_idx
        elif _decrease: new_idx -= 1
        else:           new_idx += 1
    
    if not _decrease: return len(encoded) - 1
    return _get_split_index(encoded, last_end, idx, _can_split_fn, False)


def apply_format(format, ** kwargs):
    if '{%' not in format:
        return format.format(** kwargs)
    
    compiled_format = compile_jinja_template(format)

    return compiled_format.render(** kwargs)

@cache
def compile_jinja_template(template):
    import jinja2

    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
    env.globals['raise_exception'] = raise_exception
    return env.from_string(template)

def execute_and_concat(fn_name):
    fn = getattr(TextEncoder, fn_name)
    expanded_signature  = tree.map_structure(
        lambda sig: TensorSpec(
            shape = (None, ) + sig.shape, dtype = sig.dtype, name = sig.name
        ), fn.signature
    )
    if not isinstance(expanded_signature, (list, tuple)):
        expanded_signature = [expanded_signature]
    
    add_length = False
    if not any(s.name == text_length_signature.name for s in expanded_signature):
        expanded_signature = list(expanded_signature) + [text_length_signature]
        add_length  = True
    
    def inner(self, * args, return_type = 'np', ** kwargs):
        return_type = convert_to_str(return_type)
        max_len = max(
            [len(a) if isinstance(a, (list, np.ndarray)) else 1 for a in args] +
            [len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in kwargs.values()]
        )
        
        results = [getattr(self, fn_name)(* [
            a[i] if isinstance(a, (list, np.ndarray)) and len(a) == max_len else a
            for a in args
        ], ** {
            k : v[i] if isinstance(v, (list, np.ndarray)) and len(v) == max_len else v
            for k, v in kwargs.items()
        }, i = i, return_type = 'np') for i in range(max_len)]
        
        if isinstance(results[0], list):
            results = list(zip(* results))
        else:
            results = [results]
        
        if add_length:
            results.append([len(encoded) for encoded in results[0]])
        
        if len(results) == 2:
            results = filter_texts(* results, ** kwargs)
        else:
            texts, lengths, indices = filter_texts(
                results[0], results[-1], return_indices = True, ** kwargs
            )
            results = [
                texts, * [[res[idx] for idx in indices] for res in results[1:-1]], lengths
            ]
        
        if return_type == 'list': return results
        
        if not results[0]:
            results = [
                np.zeros(shape = [0] * len(sign.shape), dtype = np.int32)
                for sign in expanded_signature
            ]
        else:
            texts   = pad_batch(results[0], self.blank_token_idx, dtype = np.int32)
            others  = [pad_batch(res, dtype = np.int32) for res in results[1:]]
            results = [texts] + others
        
        if return_type == 'np':
            return results
        elif return_type == 'tf':
            return [ops.convert_to_tf_tensor(res, 'int32') for res in results]
        elif return_type == 'tensor':
            return [ops.convert_to_tensor(res, 'int32') for res in results]
        else:
            raise ValueError("Unknown `return_type` : {}".format(return_type))
    
    inner.__doc__   = fn.__doc__
    inner.__name__  = 'multi_' + fn_name
    return execute_eagerly(
        inner, signature = expanded_signature, numpy = True, default_key = fn.default_key
    )

for fn_name in ('encode', 'format', 'split', 'split_and_format'):
    setattr(TextEncoder, 'multi_' + fn_name, execute_and_concat(fn_name))
    