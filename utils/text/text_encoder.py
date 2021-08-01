""" from https://github.com/keithito/tacotron """
import json
import regex as re
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.text import cleaners as cleaners_module
from utils.text.text_processing import bytes_to_unicode, bpe
from utils.distance.distance_method import distance
from utils.generic_utils import dump_json, load_json

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')
_gpt_pattern    = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

CHAR_LEVEL  = 0
TOKEN_LEVEL = 1
WORD_LEVEL  = 2

_str_level = {
    'char'  : CHAR_LEVEL,
    'token' : TOKEN_LEVEL,
    'word'  : WORD_LEVEL
}

def get_cleaners_fn(cleaners):
    cleaners_fn    = []
    for name in cleaners:
        kwargs = None
        if isinstance(name, dict): 
            name, kwargs = name['name'], {k : v for k, v in name.items() if k != 'name'}
        cleaner = getattr(cleaners_module, name)
        if not cleaner:
            raise ValueError("Cleaner inconnu : {}".format(name))
        elif not callable(cleaner):
            raise ValueError("Cleaner non callable : {}".format(name))
        cleaners_fn.append(cleaner if not kwargs else (cleaner, kwargs))
    return cleaners_fn

class TextEncoder(object):
    def __init__(self,
                 vocab,
                 level,
                 vocab_size     = None,
                 
                 cleaners       = [], 
                 split_pattern  = None,
                 bpe_pairs      = None,
                 byte_encoder   = None,
                 
                 pad_token      = '',       # blank token
                 ukn_token      = None,     # for unknown toekn (if not provided, skip them)
                 sos_token      = '[SOS]',  # Start Of Sequence
                 eos_token      = '[EOS]',  # End Of Sequence
                 sub_word_prefix    = '',   # Add for inner sub-word part
                 use_sos_and_eos    = False,
                 
                 name           = 'Text encoder'
                ):
        """
            Constructor for the TextEncoder class
            Arguments : 
                - vocab     : list of tokens (either words or characters). 
                - level     : tokenization level (char, token, word)
                - vocab_size    : special vocab_size (by default len(vocab)). 
                - cleaners      : cleaners to use for text-cleaning. 
                    - list  : list of names or callable
                    - dict  : keys are the clean_ername and values are dict of keywords to pass when calling the cleaner function
                - ukn_token     : specific token to put when the word is unknown
                - use_eos_and_eos   : whether to add <sos> / <eos> at the beginning / end of the sentence
                - name      : special name for this encoder
        """
        if isinstance(level, str): level = _str_level.get(level, None)
        assert level in (CHAR_LEVEL, TOKEN_LEVEL, WORD_LEVEL)
        if level != CHAR_LEVEL and 'detach_punctuation' not in cleaners:
            print("Note : when using token / word-level tokenizer, it can be useful to add 'detach_punctuation' in cleaners")
        
        self.name       = name
        self.vocab      = list(vocab)
        self.level      = level
        
        self.cleaners   = cleaners
        self.split_pattern  = split_pattern
        self.bpe_pairs      = bpe_pairs
        self.byte_encoder   = byte_encoder
        
        self.pad_token  = pad_token
        self.ukn_token  = ukn_token
        self.sos_token  = sos_token
        self.eos_token  = eos_token
        self.sub_word_prefix    = sub_word_prefix
        self.use_sos_and_eos    = use_sos_and_eos
        
        
        self.splitter   = re.compile(split_pattern) if split_pattern is not None else None
        self.bpe_ranks  = {pair : i for i, pair in enumerate(self.bpe_pairs)} if self.bpe_pairs else None
        self.cleaners_fn    = get_cleaners_fn(cleaners)
        
        self._bpe_cache     = {}
        self._symbol_to_id  = {}
        self._id_to_symbol  = {}
        self.__build_indexes(vocab_size)
        
        
    def __build_indexes(self, vocab_size = None):
        if vocab_size is not None:
            if self.use_sos_and_eos:
                if self.sos_token not in self.vocab: vocab_size -= 1
                if self.eos_token not in self.vocab: vocab_size -= 1
            if self.ukn_token is not None and self.ukn_token not in self.vocab: vocab_size -= 1
            
            if len(self.vocab) > vocab_size:
                self.vocab = self.vocab[: vocab_size]
            elif len(self.vocab) < vocab_size:
                self.vocab += ['ukn_{}'.format(i) for i in range(vocab_size - len(self.vocab))]
        
        # Build `symbol to id` pairs
        self._symbol_to_id  = {s: i for i, s in enumerate(self.vocab)}
        # Add UKN token (if required)
        if self.ukn_token is not None:
            self._symbol_to_id.setdefault(self.ukn_token, len(self.vocab))
        # Add EOS and SOS tokens (if required)
        if self.use_sos_and_eos:
            self._symbol_to_id.setdefault(self.sos_token, len(self._symbol_to_id))
            self._symbol_to_id.setdefault(self.eos_token, len(self._symbol_to_id))

        # Build ID --> symbol pairs
        self._id_to_symbol  = {idx : s for s, idx in self._symbol_to_id.items()}
        
        if self.bpe_pairs is not None and self.byte_encoder is None: self.byte_encoder = bytes_to_unicode()
        
    @property
    def vocab_size(self):
        return len(self._id_to_symbol)
    
    @property
    def word_split(self):
        return self.level != CHAR_LEVEL
    
    @property
    def sos_token_idx(self):
        return self._symbol_to_id[self.sos_token] if self.use_sos_and_eos else -1
    
    @property
    def eos_token_idx(self):
        return self._symbol_to_id[self.eos_token] if self.use_sos_and_eos else -1
    
    @property
    def blank_token_idx(self):
        return self._symbol_to_id.get(self.pad_token, 0)
    
    @property
    def ukn_token_idx(self):
        return self._symbol_to_id[self.ukn_token] if self.ukn_token is not None else -1
    
    @property
    def blank_token(self):
        return self._id_to_symbol[self.blank_token_idx]
    
    def __str__(self):
        des = "========== {} ==========\n".format(self.name)
        des += "Vocab (size = {}) : {}\n".format(len(self), self.vocab[:50])
        config = self.get_config()
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
            return self._symbol_to_id[idx]
    
    def __contains__(self, label):
        return label in self.vocab
    
    def clean_text(self, text):
        """ Apply all cleaners to 'text' """
        for cleaner in self.cleaners_fn:
            kwargs = {}
            if isinstance(cleaner, tuple): cleaner, kwargs = cleaner
            
            text = cleaner(text, ** kwargs)
        return text
    
    def split_text(self, text):
        if self.splitter is not None: return list(re.findall(self.splitter, text))
        return text.split() if self.word_split else list(text)
    
    def _char_tokenize(self, splitted_text):
        return splitted_text
    
    def _word_tokenize(self, splitted_text):
        return splitted_text
    
    def _sub_word_tokenize(self, splitted_text):
        tokens = []
        for part in splitted_text:
            start, valid = 0, True
            sub_tokens = []
            while start < len(part):
                end = len(part)
                while start < end:
                    sub_part = part[start : end]
                    
                    token = sub_part if start == 0 else self.sub_word_prefix + sub_part

                    if token in self._symbol_to_id:
                        sub_tokens.append(token)
                        break
                    end -= 1
                
                if end <= start:
                    valid = False
                    break
                
                start += len(sub_part)
            
            if valid:
                tokens.extend(sub_tokens)
            elif self.ukn_token is not None:
                tokens.append(self.ukn_token)
            
        return tokens
    
    def _bpe_tokenize(self, splitted_text):
        tokens = []
        for token in splitted_text:
            token = ''.join([self.byte_encoder.get(b, b) for b in token.encode('utf-8')])
            if token not in self._bpe_cache:
                bpe_token = bpe(token, self.bpe_ranks)
                self._bpe_cache[token] = bpe_token
                
            tokens.extend(self._bpe_cache[token])
            
        return tokens
    
    def tokenize(self, text):
        splitted = self.split_text(text)
        
        if self.level == CHAR_LEVEL:
            return self._char_tokenize(splitted)
        elif self.level == WORD_LEVEL:
            return self._word_tokenize(splitted)
        elif self.level == TOKEN_LEVEL and self.bpe_pairs is None:
            return self._sub_word_tokenize(splitted)
        elif self.level == TOKEN_LEVEL and self.bpe_pairs is not None:
            return self._bpe_tokenize(splitted)
    
    def encode(self, text):
        """
            Encode text in np.ndarray
            Arguments :
                - text : text to encode
                    Can be a string, dict, list / tuple / np.ndarray / tf.Tensor of dict or str
            Return : 
                - np.ndarray    : the encoded text (or list of np.ndarrays if multiple textwere given)
            
            1) Clean the text
            2) Split it either by word or by characters
            3) Convert all tokens to its corresponding id (with the `self.tokenize` method)
            4) If necessary, add [sos, eos] tokens 
        """
        if isinstance(text, pd.DataFrame): return [self.encode(row) for _, row in text.iterrows()]
        if isinstance(text, (dict, pd.Series)): text = text['text']
        if isinstance(text, tf.Tensor): text = text.numpy()
        if isinstance(text, bytes): text = text.decode('utf-8')
        if isinstance(text, (list, tuple, np.ndarray)): return [self.encode(t) for t in text]
        
        text = self.clean_text(text)
        
        tokens  = self.tokenize(text)
        
        if self.use_sos_and_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]
            
        return np.array([
            self._symbol_to_id.get(token, self.ukn_token_idx) for token in tokens
            if token in self._symbol_to_id or self.ukn_token is not None
        ])

    def decode(self, sequence, skip_padding = True, attach_punctuation = True):
        """ Decode a given np.ndarray by replacing each known id by its corresponding token """
        sep = ' ' if self.word_split else ''
        
        tokens = [self._id_to_symbol[s] for s in sequence if s in self._id_to_symbol]
        
        if skip_padding:
            tokens = [token for token in tokens if token != self.blank_token]
        
        if self.level == TOKEN_LEVEL:
            tokens = [
                token.replace(self.sub_word_prefix, '') for token in tokens
            ]
        
        text = sep.join(tokens)
        
        if attach_punctuation:
            text = cleaners_module.attach_punctuation(text)
            
        return text
    
    def distance(self, hypothesis, truth, method = 'edit', ** kwargs):
        """ Compute the levenschtein distance between hypothesis and truth """
        if isinstance(hypothesis, str) and not isinstance(truth, str):
            hypothesis = self.encode(hypothesis)
        if isinstance(truth, str) and not isinstance(hypothesis, str):
            truth = self.encode(truth)
        
        kwargs.setdefault('insertion_cost', {})
        kwargs.setdefault('deletion_cost', {})
        
        kwargs['insertion_cost'].setdefault(self.blank_token, 0)
        kwargs['deletion_cost'].setdefault(self.blank_token, 0)
        
        return distance(hypothesis, truth, method = method, ** kwargs)
    
    def save_to_file(self, filename):
        if '.json' not in filename:
            filename += '.json'
        
        dump_json(filename, self.get_config(), indent = 4)

        return filename
    
    def get_config(self):
        return {
            'name'      : self.name,
            'vocab'     : self.vocab,
            'level'     : self.level,
        
            'cleaners'  : self.cleaners,
            'split_pattern' : self.split_pattern,
            'bpe_pairs' : self.bpe_pairs,
            'byte_encoder'  : self.byte_encoder,
            
            'pad_token' : self.pad_token,
            'ukn_token' : self.ukn_token,
            'sos_token' : self.sos_token,
            'eos_token' : self.eos_token,
            'sub_word_prefix'   : self.sub_word_prefix,
            'use_sos_and_eos'   : self.use_sos_and_eos
        }
    
    @classmethod
    def load_from_file(cls, filename):
        config = load_json(filename)
        
        if 'word_level' in config:
            config['level'] = CHAR_LEVEL if not config.pop('word_level') else TextEncoder.WORD_LEVEL
        
        return cls(** config)

    @classmethod
    def from_transformers_pretrained(cls, name, ** kwargs):
        from transformers import AutoTokenizer, BertTokenizer, GPT2Tokenizer
        
        pretrained = AutoTokenizer.from_pretrained(name, use_fast = False)

        if isinstance(pretrained, BertTokenizer):
            kwargs.update({
                'vocab' : pretrained.vocab.keys(),
                'level'             : TOKEN_LEVEL,
                'sub_word_prefix'   : '##',
                'cleaners'          : ['detach_punctuation', 'collapse_whitespace'],
                'use_sos_and_eos'   : True,
                'sos_token'         : pretrained.cls_token,
                'eos_token'         : pretrained.sep_token,
                'pad_token'         : pretrained.pad_token,
                'ukn_token'         : pretrained.unk_token
            })
        elif isinstance(pretrained, GPT2Tokenizer):
            # Note that RoBERTa and BART Tokenizer are subclasses of GPT2Tokenizer
            kwargs.update({
                'vocab' : pretrained.encoder.keys(),
                'level'             : TOKEN_LEVEL,
                'split_pattern'     : _gpt_pattern,
                'cleaners'          : [],
                'bpe_pairs'         : list(pretrained.bpe_ranks.keys()),
                'byte_encoder'      : pretrained.byte_encoder,
                'use_sos_and_eos'   : True,
                'sos_token'         : pretrained.bos_token,
                'eos_token'         : pretrained.eos_token,
                'pad_token'         : pretrained.pad_token,
                'ukn_token'         : pretrained.unk_token
            })
        if pretrained.do_lower_case:
            kwargs['cleaners'].append('lowercase')
        return cls(** kwargs)
    
    @classmethod
    def build_from_corpus(cls, textes, word_level, max_vocab_size = -1, 
                          cleaners = [], tqdm = lambda x: x, **kwargs):
        """ In theory it should work but it has not been tested yet :3 """
        seen_vocab = {}
        
        cleaners_fn    = get_cleaners_fn(cleaners)
        for text in tqdm(texts):
            for cleaners_fn in cleaners_fn:
                text = cleaner(text)
            
            tokens = text.split() if word_level else list(text)
            for token in tokens:
                if token not in seen_vocab: seen_vocab[token] = 0
                seen_vocab[token] += 1
                
        if max_vocab_size > 0 and len(seen_vocab) > max_vocab_size:
            vocab_order = sorted(list(seen_vocab.items()), key = lambda x: x[1], reverse=True)
            vocab = [token for token, _ in vocab_order[:max_vocab_size]]
        else:
            vocab = [token for token, _ in seen_vocab]

        if sort: vocab.sort()

        return cls(vocab, word_level, cleaners = cleaners, **kwargs)
