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

import os
import enum
import glob
import json
import logging
import numpy as np
import regex as re

from datetime import datetime
from functools import cached_property, cache

from loggers import Timer, timer
from .. import load_json, dump_json, pad_batch, get_enum_item, is_dataframe, convert_to_str
from ..keras import TensorSpec, ops, execute_eagerly
from .ctc_decoder import ctc_decode
from .cleaners import get_cleaners_fn, clean_text, strip
from .text_processing import bytes_to_unicode, bpe, format_text, split_and_join
from .tokens_processing import process_model_output

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

        
class TokenizerLevel(enum.IntEnum):
    CHAR    = 0
    BPE     = 1
    TOKEN   = 1
    SUBWORD = 1
    WORD    = 2

class Tokenizer:
    def __init__(self,
                 vocab,
                 level,
                 
                 *,
                 
                 template   = None,
                 vocab_size     = None,
                 
                 lstrip     = False,
                 rstrip     = False,
                 cleaners   = [],
                 
                 split_pattern  = None,
                 
                 bpe_pairs      = None,
                 byte_encoder   = None,
                 bpe_end_of_word    = None,
                 
                 pad_token      = '',       # blank token
                 sos_token      = None,     # Start Of Sequence
                 eos_token      = None,     # End Of Sequence
                 ukn_token      = None,     # for unknown toekn (if not provided, skip them)
                 sep_token      = None,
                 mask_token     = None,
                 additional_tokens  = None,
                 
                 sub_word_prefix    = '',   # Add for inner sub-word part
                 use_sos_and_eos    = False,
                 add_special_tokens_at_end  = True,
                 
                 name           = 'Tokenizer'
                ):
        """
            Constructor for the `Tokenizer` class
            
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
        self.name       = name
        self._vocab     = list(vocab)
        self.level      = get_enum_item(level, TokenizerLevel)
        self.template   = template
        
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.cleaners   = cleaners
        
        self.split_pattern  = split_pattern
        
        if bpe_pairs and not byte_encoder: byte_encoder = bytes_to_unicode()
        self.bpe_pairs      =  [tuple(pair) for pair in bpe_pairs] if bpe_pairs else None
        self.byte_encoder   =  {int(k) : v for k, v in byte_encoder.items()} if byte_encoder else None
        self.bpe_end_of_word    = bpe_end_of_word
        self.byte_encoder_inv   = {v : k for k, v in self.byte_encoder.items()} if byte_encoder else None
        
        self.pad_token  = pad_token
        self.sos_token  = sos_token
        self.eos_token  = eos_token
        self.ukn_token  = ukn_token
        self.sep_token  = sep_token
        self.mask_token = mask_token
        self.sub_word_prefix    = sub_word_prefix
        self.use_sos_and_eos    = use_sos_and_eos
        self.add_special_tokens_at_end  = add_special_tokens_at_end

        if additional_tokens is None:            additional_tokens = []
        elif isinstance(additional_tokens, str): additional_tokens = [additional_tokens]
        if isinstance(additional_tokens, (list, tuple)):
            additional_tokens = {_create_token_name(tok) : tok for tok in additional_tokens}
        self.additional_tokens  = additional_tokens
        
        
        self.splitter   = re.compile(split_pattern) if split_pattern else None
        self.bpe_ranks  = {pair : i for i, pair in enumerate(self.bpe_pairs)} if bpe_pairs else None
        self.cleaners_fn    = get_cleaners_fn(cleaners)
        
        self._special_tokens    = list(self.tokens.values())
        self._tokens_split_re   = re.compile(
            '({})'.format('|'.join([re.escape(tok) for tok in self._special_tokens]))
        )
        self._bpe_cache     = {}
        self._symbol_to_id  = {}
        self._id_to_symbol  = {}
        self.__build_indexes(vocab_size, add_special_tokens_at_end)
        self._cleaned_tokens    = {
            self.clean_text(token) : token for token in self._special_tokens
        }
        
        for name, token in self.additional_tokens.items():
            assert not hasattr(self, name), '{} is already defined'.format(name)
            
            setattr(self, name, token)
            setattr(self, name + '_idx', self[token])

    def __build_indexes(self, vocab_size, add_special_tokens_at_end):
        def _add_symbols(symbols):
            for s in symbols: self._symbol_to_id.setdefault(s, len(self._symbol_to_id))
        
        special_tokens = list(self.tokens.values())
        
        if vocab_size:
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
        
    @property
    def vocab_size(self):
        return len(self._id_to_symbol)
    
    @property
    def vocab(self):
        return sorted(self._symbol_to_id.keys(), key = self._symbol_to_id.get)
    
    @property
    def word_split(self):
        return self.level != TokenizerLevel.CHAR and self.splitter is None
    
    @cached_property
    def special_tokens(self):
        return set(v for k, v in self.get_config().items() if k.endswith('_token') and v is not None)
    
    @cached_property
    def tokens(self):
        _tokens = ('sep', 'ukn', 'sos', 'eos')
        
        tokens  = {
            k + '_token' : getattr(self, k + '_token') for k in _tokens
            if getattr(self, k + '_token', None) is not None
        }
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
        return self._symbol_to_id.get(self.sos_token, -1)
    
    @property
    def eos_token_idx(self):
        return self._symbol_to_id.get(self.eos_token, -1)
    
    @property
    def sep_token_idx(self):
        return self._symbol_to_id.get(self.sep_token, -1)
    
    @property
    def ukn_token_idx(self):
        return self._symbol_to_id.get(self.ukn_token, -1)
    
    @property
    def mask_token_idx(self):
        return self._symbol_to_id.get(self.mask_token, -1)
        
    @property
    def blank_token_idx(self):
        default = 0 if not self.use_sos_and_eos else self.eos_token_idx
        return self._symbol_to_id.get(self.pad_token, default)

    @property
    def blank_token(self):
        return self._id_to_symbol[self.blank_token_idx]
    
    def __str__(self):
        des = "========== {} ==========\n".format(self.name)
        des += "Vocab (size = {}) : {}\n".format(
            len(self), self.vocab if self.vocab_size <= 50 else '[{}, ...]'.format(
                str(self.vocab[:50])[1:-1]
            )
        )
        config = self.get_config()
        if self.template:
            des += 'Template :\n{}\n'.format(pretty_print_template(config.pop('template')))
        
        for k in ['name', 'vocab', 'bpe_pairs', 'byte_encoder']: config.pop(k)
        des += 'Config : {}'.format(json.dumps(config, indent = 2))
        return des
        
    def __len__(self):
        return self.vocab_size
    
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._id_to_symbol[idx]
        elif idx in self._symbol_to_id:
            return self._symbol_to_id[idx]
        elif isinstance(idx, str):
            res = self.encode(idx, add_sos_and_eos = False, cleaned = True, return_type = 'list')
            return res[0] if len(res) == 1 else res
        else:
            raise KeyError('Unknown index : {}'.format(idx))
    
    def __contains__(self, token):
        return token in self._symbol_to_id
    
    def _sub_word_tokenize(self, token):
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
        if token not in self._bpe_cache:
            token = ''.join([
                self.byte_encoder[b] for b in token.encode('utf-8') if b in self.byte_encoder
            ])
            self._bpe_cache[token] = bpe(token, self.bpe_ranks, end_of_word = self.bpe_end_of_word)
        
        return self._bpe_cache[token]
    
    @timer(log_if_root = False)
    def _tokenize(self, token):
        if isinstance(token, (list, tuple)):
            tokens = []
            for t in token:
                t = self._tokenize(t)
                tokens.extend(t if isinstance(t, (list, tuple)) else [t])
            return tokens
        elif self.level != TokenizerLevel.TOKEN or token in self._special_tokens:
            return token
        elif self.bpe_pairs is not None:
            return self._bpe_tokenize(token)
        else:
            return self._sub_word_tokenize(token)

    @timer(log_if_root = False)
    def clean_text(self, text, tokens = {}, ** kwargs):
        """ Apply all cleaners to 'text' """
        if not isinstance(tokens, dict):
            tokens = {self.clean_text(token, ** kwargs) : token for token in tokens}
        
        cleaned = clean_text(text, self.cleaners_fn, tokens = tokens, ** kwargs)
        return strip(cleaned, lstrip = self.lstrip, rstrip = self.rstrip)
    
    @timer(log_if_root = False)
    def split_text(self, text):
        """ Splits `text` into a list of tokens """
        parts = re.split(self._tokens_split_re, text)

        splitted = []
        for part in parts:
            if not part: continue
            elif part in self._special_tokens:
                splitted.append(part)
            elif self.splitter is not None:
                splitted.extend(re.findall(self.splitter, part))
            elif self.word_split:
                splitted.extend(part.split())
            else:
                splitted.extend(part)

        return splitted
    
    @timer(log_if_root = False)
    def tokenize(self, text, cleaned = False, ** kwargs):
        if not cleaned:
            text = self.clean_text(text, self._cleaned_tokens, ** kwargs)
        
        splitted = self.split_text(text)
        
        tokens = self._tokenize(splitted)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Cleaned  : {}\nSplitted : {}\nTokens  : {}'.format(text, splitted, tokens))
        
        return tokens
    
    @timer
    @execute_eagerly(signature = TensorSpec(shape = (None, ), dtype = 'int32'), numpy = True)
    def encode(self,
               text,
               
               *,
               
               cleaned  = False,
               
               add_sos  = None,
               add_eos  = None,
               add_sos_and_eos  = None,
               
               return_type  = 'np'
              ):
        """
            Encode text (str) into tokens (int)
            Arguments :
                - text : text to encode
                    Can be a string, dict, list / tuple / np.ndarray / Tensor of dict or str
            Return :
                - np.ndarray    : the encoded text (or list of np.ndarrays if multiple textwere given)
            
            1) Clean the text
            2) Split it into tokens (characters, words or sub-words) (cf `self.tokenize`)
            3) Convert all tokens to its corresponding id (with `self._symbol_to_id`)
            4) If necessary, add `sos` and `eos` tokens
        """
        return_type = convert_to_str(return_type)
        if add_sos_and_eos is None: add_sos_and_eos = self.use_sos_and_eos
        if add_sos is None: add_sos = add_sos_and_eos
        if add_eos is None: add_eos = add_sos_and_eos

        if is_dataframe(text):          text = text['text'].values.tolist()
        elif isinstance(text, dict):    text = text['text']
        text = convert_to_str(text)
        
        if isinstance(text, (list, tuple)):
            encoded = [self.encode(
                t, cleaned = cleaned, add_sos = add_sos, add_eos = add_eos, return_type = 'list'
            ) for t in text]
            
            if return_type == 'list': return encoded
            encoded = pad_batch(encoded, pad_value = self.blank_token_idx, dtype = 'int32')
            if return_type == 'np':     return encoded
            elif return_type == 'tf':   return ops.convert_to_tf_tensor(encoded, 'int32')
            elif return_type == 'tensor':   return ops.convert_to_tensor(encoded, 'int32')
            else:   raise ValueError("Unknown `return_type` : {}".format(return_type))

        tokens  = self.tokenize(text, cleaned = cleaned)
        tokens = [self._symbol_to_id.get(token, self.ukn_token_idx) for token in tokens]
        tokens = [token for token in tokens if token != -1]

        if (add_sos and self.sos_token) and (len(tokens) == 0 or tokens[0] != self.sos_token_idx):
            tokens.insert(0, self.sos_token_idx)
        if (add_eos and self.eos_token) and (len(tokens) == 0 or tokens[-1] != self.eos_token_idx):
            tokens.append(self.eos_token_idx)
        
        if return_type == 'list': return tokens
        elif return_type == 'np': return np.array(tokens, dtype = np.int32)
        elif return_type == 'tf': return ops.convert_to_tf_tensor(tokens, dtype = 'int32')
        elif return_type == 'tensor':   return ops.convert_to_tensor(tokens, dtype = 'int32')
        else:   raise ValueError("Unknown `return_type` : {}".format(return_type))

    __call__    = encode
    
    @timer
    @execute_eagerly(signature = TensorSpec(shape = (None, ), dtype = 'int32'), numpy = True)
    def encode_chat(self,
                    text = None,
                    *,
                    
                    system_prompt   = None,
                    answer_start    = None,

                    messages    = None,
                    message_format  = None,
                    last_message_format = None,
               
                    encode  = True,
                    add_eos = None,
                    max_length  = None,
                    add_generation_prompt   = True,
                    
                    return_text = False,
                    return_type = 'np',
                    
                    ** kwargs
                   ):
        assert text or messages
        
        if add_eos is None: add_eos = not add_generation_prompt
        
        kwargs   = convert_to_str(kwargs)
        kwargs.update(self.tokens)
        
        with Timer('messages preparation'):
            if messages is None:                messages = []
            elif isinstance(messages, dict):    messages = [messages]
            elif isinstance(messages, str):     messages = [{'role' : 'user', 'content' : messages}]
            elif not isinstance(messages, list):
                raise ValueError('Unsupported `messages` argument : {}'.format(messages))

            if text:
                messages += [{'role' : 'user', 'content' : text}]

            if message_format:
                messages    = [{
                    ** (msg if isinstance(msg, dict) else msg.get_config()),
                    'content' : format_text(
                        message_format, text = msg['content'], message = msg, ** kwargs
                    )
                } for msg in messages]

            if messages and last_message_format:
                messages[-1] = {** messages[-1], 'content' : format_text(
                    last_message_format, text = messages[-1]['content'], ** kwargs
                )}
            
            if system_prompt and messages[0]['role'] != 'system':
                messages = [
                    {'role' : 'system', 'content' : format_text(system_prompt, messages = messages, ** kwargs)}
                ] + messages

            if 'date_string' in self.template and 'date_string' not in kwargs:
                kwargs['date_string'] = datetime.now().strftime("%d %B %Y")

        for _ in range(max(1, len(messages) - 1)):
            with Timer('apply template'):
                text = format_text(
                    self.template,
                    messages    = messages,
                    add_generation_prompt = add_generation_prompt,
                    ** kwargs
                )
                if add_generation_prompt and answer_start: text += answer_start
            
            if not encode: return text

            encoded = self.encode(text, add_sos = False, add_eos = add_eos, return_type = return_type)
            if not max_length or len(encoded) <= max_length:
                return encoded if not return_text else (text, encoded)
            
            messages.pop(1)
        
        raise ValueError('The message length ({}) exceeded the maximum length ({})'.format(
            len(encoded), max_length
        ))

    def decode(self, tokens, *, skip_padding = True, remove_tokens = False, ** _):
        """ Decode the given list of token ids into their corresponding token (str) """
        if hasattr(tokens, 'tokens'): tokens = process_model_output(tokens)
        
        if ops.is_tensor(tokens): tokens = ops.convert_to_numpy(tokens)
        if isinstance(tokens, np.ndarray):
            if np.issubdtype(tokens.dtype, np.floating) and all(s > 0 for s in tokens.shape):
                tokens = np.argmax(tokens, axis = -1)
            
            if tokens.ndim > 1:
                return [self.decode(
                    t, skip_padding = skip_padding, remove_tokens = remove_tokens
                ) for t in tokens]
        
        elif isinstance(tokens, (list, tuple)) and not isinstance(tokens[0], int):
            return [self.decode(
                t, skip_padding = skip_padding, remove_tokens = remove_tokens
            ) for t in tokens]
        
        if remove_tokens: skip_padding = True
        
        if remove_tokens:   _skip = self.token_indexes
        elif skip_padding:  _skip = [self.blank_token_idx]
        else:               _skip = []
        
        symbols = []
        for i, token in enumerate(tokens):
            if token not in _skip: symbols.append(int(token))
            if (
                (skip_padding and token == self.blank_token_idx)
                and (i < len(tokens) - 1 and tokens[i + 1] == self.blank_token_idx)):
                
                break
        
        return self.decode_ids(symbols)
    
    def decode_ids(self, tokens):
        if isinstance(tokens, list):
            symbols = [self._id_to_symbol.get(t, '') for t in tokens]

            sep = ' ' if self.word_split else ''
            text = sep.join(symbols)
        else:
            text = self._id_to_symbol.get(t, '')
        
        if self.byte_encoder is not None:
            try:
                text = bytearray([self.byte_encoder_inv.get(c, c) for c in text]).decode('utf-8')
            except UnicodeDecodeError as e:
                pass
        
        if self.level == TokenizerLevel.TOKEN:
            if self.sub_word_prefix:
                text = text.replace(' ' + self.sub_word_prefix, '')
            elif self.bpe_end_of_word:
                text = text.replace(self.bpe_end_of_word, ' ')
        
        return text
    
    def ctc_decode(self, logits, lengths = None, method = 'beam', return_scores = False, ** kwargs):
        """ Decode a given np.ndarray by replacing each known id by its corresponding token """
        tokens, scores = ctc_decode(
            logits,
            method  = method,
            lengths = lengths,
            blank_index = self.blank_token_idx,
            ** kwargs
        )

        decoded = self.decode(tokens, ** kwargs)
        return decoded if not return_scores else (decoded, scores)

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
            'sos_token' : self.sos_token,
            'eos_token' : self.eos_token,
            'sep_token' : self.sep_token,
            'ukn_token' : self.ukn_token,
            'mask_token'    : self.mask_token,
            'additional_tokens' : self.additional_tokens,
            
            'sub_word_prefix'   : self.sub_word_prefix,
            'use_sos_and_eos'   : self.use_sos_and_eos,
            'add_special_tokens_at_end' : self.add_special_tokens_at_end
        }
    
    def save(self, filename):
        """ Saves `self.get_config()` to `filename` (`json` format) """
        return dump_json(filename, self.get_config(), indent = 4)

    save_to_file = save
    
    @classmethod
    def load_from_file(cls, filename):
        config = load_json(filename)
        
        if 'tokenizer' in config:
            from .sentencepiece_tokenizer import SentencePieceTokenizer
            cls = SentencePieceTokenizer
        
        return cls(** config)

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
            from .sentencepiece_tokenizer import SentencePieceTokenizer
            cls = SentencePieceTokenizer
            
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
            'level'         : TokenizerLevel.TOKEN,
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

def _create_token_name(token):
    name = ''.join([c for c in token.lower() if c.isalnum() or c == '_'])
    for suffix in ('_id', '_tag'):
        if name.endswith(suffix): name = name.replace(suffix, '_token')
    if 'token' not in name: name += '_token'
    return name

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

