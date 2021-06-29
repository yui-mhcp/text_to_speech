import random
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from sklearn.utils import shuffle as sklearn_shuffle

from datasets.dataset_utils import prepare_dataset, build_siamese_dataset
from utils.audio.audio_io import load_audio, load_mel

class AudioSiameseGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 # General informations
                 dataset,
                 rate,
                 mel_fn = None,
                 # Cache parameters
                 min_apparition = 3,
                 cache_size     = 30000,
                 preload        = False,
                 # Column for merging / loading
                 id_column  = 'id',
                 processed_column   = None,
                 # additional informations
                 shuffle        = False,
                 suffixes       = ('_x', '_y'), 
                 random_state   = 10,
                 
                 ** kwargs
                ):
        assert isinstance(dataset, pd.DataFrame)
        self.dataset    = dataset
        self.rate       = rate
        self.mel_fn     = mel_fn
        
        self.shuffle        = shuffle
        self.suffixes       = suffixes
        self.random_state   = random_state
        
        self.cache      = {}
        self.cache_size = cache_size
        self.min_apparition = min_apparition
        
        self.id_column  = id_column
        self.id_col_x   = id_column + self.suffixes[0]
        self.id_col_y   = id_column + self.suffixes[1]
        
        self.file_column   = 'wavs_{}'.format(rate)
        if self.file_column not in dataset.columns: 
            self.file_column = 'filename' if 'filename' in dataset.columns else 'audio_filename'      
        self.file_col_x = self.file_column + self.suffixes[0]
        self.file_col_y = self.file_column + self.suffixes[1]
        
        if processed_column is None:
            processed_column = 'audio' if mel_fn is None else 'mel'
        self.processed_column   = processed_column
        self.processed_col_x = self.processed_column + self.suffixes[0]
        self.processed_col_y = self.processed_column + self.suffixes[1]
        
        self.build_datasets(** kwargs)
        self.build_cache(cache_size, min_apparition, preload)
    
    def build_datasets(self, ** kwargs):
        kwargs.setdefault('random_state', self.random_state)
        self.same, self.not_same = build_siamese_dataset(
            self.dataset,
            column      = self.id_column,
            suffixes    = self.suffixes,
            as_tf_dataset   = False,
            shuffle     = True,
            ** kwargs
        )
        
        self.unique_files, self.frequence = self.get_uniques()
    
    def get_uniques(self):
        uniques = {}
        for file in self.all_files:
            uniques.setdefault(file, 0)
            uniques[file] += 1
            
        return np.array(list(uniques.keys())), np.array(list(uniques.values()))
        
    def build_cache(self, size, min_apparition = 2, preload = False):
        # compute 'size' most present files (in 2 ds)
        cache_idx = np.flip(np.argsort(self.frequence))[:size]
        to_cache = self.unique_files[cache_idx]
        # get files with at least 'min_apparition' apparition
        freq = self.frequence[cache_idx]
        to_cache = to_cache[np.where(freq >= min_apparition)]
        
        self.files_to_cache = to_cache
        
        if preload:
            self.load_cache()
        
        return to_cache
    
    def load_cache(self, tqdm = tqdm, ** kwargs):
        self.cache = {f : self.cache[f] for f in self.unique_files if f in self.cache}
        
        not_cached = np.array([f for f in self.files_to_cache if f not in self.cache])
        ds = prepare_dataset(
            pd.DataFrame([{'filename' : f} for f in not_cached]),
            batch_size  = 0,                     
            map_fn      = self.load_file,
            prefetch    = True,
            cache       = False,
            ** kwargs
        )
        for filename, processed in tqdm(zip(not_cached, ds), total = len(not_cached)):
            self.cache[filename] = processed
    
    @property
    def ids(self):
        return self.same[self.id_column].unique()
    
    @property
    def all_ids(self):
        return np.unique(np.concatenate([
            self.same[self.id_column].values, 
            self.not_same[self.id_col_x].values, 
            self.not_same[self.id_col_y].values
        ]))
        
    @property
    def all_files(self):
        return np.concatenate([
            self.same[self.file_col_x].values, 
            self.same[self.file_col_y].values,
            self.not_same[self.file_col_x].values, 
            self.not_same[self.file_col_y].values
        ])
            
    @property
    def output_types(self):
        types = {
            self.processed_column   : tf.float32,
            self.file_column    : tf.string,
        }
        same_types = {self.id_column : tf.string}
        not_same_types = {
            self.id_column + suffix : tf.string for suffix in self.suffixes
        }
        for k, t in types.items():
            for suffix in self.suffixes:
                same_types[k + suffix] = t
                not_same_types[k + suffix] = t
        
        return (same_types, not_same_types)
    
    @property
    def output_shapes(self):
        shapes = {
            self.processed_column   : [None] if self.mel_fn is None else [None, self.mel_fn.n_mel_channels],
            self.file_column    : [],
        }
        same_shapes = {self.id_column : []}
        not_same_shapes = {
            self.id_column + suffix : [] for suffix in self.suffixes
        }
        for k, t in shapes.items():
            for suffix in self.suffixes:
                same_shapes[k + suffix] = t
                not_same_shapes[k + suffix] = t
        
        return (same_shapes, not_same_shapes)
    
    def __str__(self):
        des = "Audio Siamese Generator :\n"
        des += "- Unique ids : {}\n".format(len(self.ids))
        des += "- Same dataset length : {}\n".format(len(self.same))
        des += "- Not same dataset length : {}\n".format(len(self.not_same))
        des += "- Total files : {}\n".format(len(self.all_files))
        des += "- Unique files : {} ({:.2f} %)\n".format(len(self.unique_files), 100 * len(self.unique_files) / len(self.all_files))
        des += "- Cache size : {} (loaded : {:.2f} %)".format(len(self.files_to_cache), 100 * len(self.cache) / len(self.files_to_cache))
        return des
    
    def __len__(self):
        return max(len(self.same), len(self.not_same))
    
    def __getitem__(self, idx):
        if idx == 0 and self.shuffle:
            self.same = sklearn_shuffle(self.same)
            self.not_same = sklearn_shuffle(self.not_same)
        return self.get_same(idx), self.get_not_same(idx)
        
    def get_same(self, idx):
        if idx > len(self.same): idx = idx % len(self.same)
        line = self.same.loc[idx]
        
        return {
            self.id_column      : line[self.id_column],
            self.file_col_x     : line[self.file_col_x],
            self.file_col_y     : line[self.file_col_y],
            self.processed_col_x    : self.load(line[self.file_col_x]),
            self.processed_col_y    : self.load(line[self.file_col_y])
        }
    
    def get_not_same(self, idx):
        if idx > len(self.not_same): idx = idx % len(self.not_same)
        line = self.not_same.loc[idx]
        
        return {
            self.id_col_x       : line[self.id_col_x],
            self.id_col_y       : line[self.id_col_y],
            self.file_col_x     : line[self.file_col_x],
            self.file_col_y     : line[self.file_col_y],
            self.processed_col_x    : self.load(line[self.file_col_x]),
            self.processed_col_y    : self.load(line[self.file_col_y])
        }
    
    def load(self, filename):
        if filename not in self.cache:
            audio = self.load_file(filename)
            if filename not in self.files_to_cache: return audio
            
            self.cache[filename] = audio
        
        return self.cache[filename]
    
    def load_file(self, filename):
        if self.mel_fn is not None:
            return load_mel(filename, self.mel_fn)
        return load_audio(filename, self.rate)
    
    def sample(self, n = 1, ids = None, random_state = None, ** kwargs):
        if not random_state: random_state = self.random_state
        if ids is None: 
            return self.dataset.sample(n, random_state = random_state, ** kwargs)
        
        if not isinstance(ids, (tuple, list, np.ndarray)): ids = [ids]
            
        samples = []
        for speaker_id in ids:
            subset = self.dataset[self.dataset[self.id_column] == speaker_id]
            
            sample_id = subset.sample(min(len(subset), n), random_state = random_state, 
                                      ** kwargs)
            samples.append(sample_id)
        
        return pd.concat(samples)

