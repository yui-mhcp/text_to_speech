""" from https://github.com/keithito/tacotron """
import re
import json
import numpy as np

from utils.text import cleaners as cleaners_module
from utils.distance.distance_method import distance
from utils.generic_utils import dump_json, load_json

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

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
    def __init__(self, vocab, word_level,
                 vocab_size     = None,
                 cleaners       = [], 
                 ukn_text       = None, 
                 use_sos_and_eos    = False, 
                 name           = 'Text encoder'
                ):
        """
            Constructor for the TextEncoder class
            Arguments : 
                - vocab     : list of tokens (either words or characters). 
                - word_level    : whether to encode text by words or by characters. 
                - vocab_size    : special vocab_size (by default len(vocab)). 
                - cleaners      : cleaners to use for text-cleaning. 
                    - list  : list of names or callable
                    - dict  : keys are the clean_ername and values are dict of keywords to pass when calling the cleaner function
                - ukn_token     : specific token to put when the word is unknown
                - use_eos_and_eos   : whether to add <sos> / <eos> at the beginning / end of the sentence
                - name      : special name for this encoder
        """
        self.name       = name
        self.vocab      = list(vocab)
        if vocab_size is not None and len(self.vocab) > vocab_size:
            self.vocab = self.vocab[:vocab_size]
        elif vocab_size is not None and len(self.vocab) < vocab_size:
            self.vocab += ['ukn_{}'.format(i) for i in range(vocab_size - len(self.vocab))]
        self.cleaners   = cleaners
        self.word_level = word_level
        self.ukn_text   = ukn_text
        self.use_sos_and_eos    = use_sos_and_eos
        
        self._symbol_to_id  = {s: i for i, s in enumerate(self.vocab)}
        self._id_to_symbol  = {i: s for i, s in enumerate(self.vocab)}
        if ukn_text is not None: self._id_to_symbol[len(self.vocab)] = ukn_text
        if self.use_sos_and_eos:
            self._id_to_symbol[len(self._id_to_symbol)] = '<start>'
            self._id_to_symbol[len(self._id_to_symbol)] = '<end>'
        self.cleaners_fn    = get_cleaners_fn(cleaners)
        
    @property
    def vocab_size(self):
        return len(self._id_to_symbol)
    
    @property
    def sos_token_idx(self):
        return self.vocab_size - 2 if self.use_sos_and_eos else -1
    
    @property
    def eos_token_idx(self):
        return self.vocab_size - 1 if self.use_sos_and_eos else -1
    
    @property
    def blank_token_idx(self):
        return self._symbol_to_id[''] if '' in self else 0
    
    @property
    def blank_token(self):
        return '' if '' in self else self._id_to_symbol[0]
    
    def __str__(self):
        des = "========== {} ==========\n".format(self.name)
        des += "Vocab (size = {}) : {}\n".format(len(self), self.vocab)
        config = self.get_config()
        config.pop('vocab')
        config.pop('name')
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
            if isinstance(cleaner, tuple):
                cleaner, kwargs = cleaner
                text = cleaner(text, ** kwargs)
            else:
                text = cleaner(text)
        return text
    
    def encode(self, text):
        """
            Encode text in np.ndarray
            Arguments :
                - text  (str or list of str) : text to encode
            Return : 
                - np.ndarray    : the encoded text (or list of np.ndarrays if multiple textwere given)
            
            1) Clean the text
            2) Split it either by word or by characters
            3) Convert all tokens to its corresponding id
            4) If necessary, add [sos, eos] tokens 
        """
        if isinstance(text, list): return [self.encode(t) for t in text]
        
        text = self.clean_text(text)
        text_part = text.split() if self.word_level else list(text)
        sequence = []
        for part in text_part:
            if part in self._symbol_to_id:
                sequence.append(self._symbol_to_id[part])
            elif self.ukn_text is not None:
                sequence.append(len(self._symbol_to_id))
            
        if self.use_sos_and_eos:
            sequence = [self.sos_token_idx] + sequence + [self.eos_token_idx]
        
        return np.array(sequence)
    
    def decode(self, sequence):
        """ Decode a given np.ndarray by replacing each known id by its corresponding token """
        sep = ' ' if self.word_level else '' 
        return sep.join([
            self._id_to_symbol[s] for s in sequence if s in self._id_to_symbol and s != self.blank_token_idx
        ])
    
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
            'cleaners'  : self.cleaners,
            'word_level'    : self.word_level,
            'ukn_text'  : self.ukn_text,
            'use_sos_and_eos'   : self.use_sos_and_eos
        }
    
    @classmethod
    def load_from_file(cls, filename):
        config = load_json(filename)

        return cls(** config)

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
