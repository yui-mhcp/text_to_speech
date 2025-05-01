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
import keras
import numpy as np

from tqdm import tqdm

from utils import load_data, dump_data, is_dataframe
from utils.keras_utils import ops, tree

class FileCacheGenerator(keras.utils.PyDataset):
    def __init__(self,
                 dataset,
                 load_fn,
                 
                 output_signature   = None,
                 
                 file_column    = 'filename',
                 
                 preload    = False,
                 cache_file = None,
                 cache_size = 50000,
                 cpu_cache_size = None,
                 gpu_cache_size = 0,
                 min_occurence  = 3,
                 
                 shuffle        = False,
                 random_state   = None,
                 
                 workers    = 4,
                 
                 ** kwargs
                ):
        super().__init__(workers = workers)
        
        assert is_dataframe(dataset), 'Dataset must be a DataFrame\n  Got : {}'.format(dataset)
        self.dataset    = dataset
        self.load_fn    = load_fn
        self._output_signature  = output_signature
        
        self.file_column    = file_column
        
        self.preload    = preload
        self.cache_file = cache_file
        self.cpu_cache_size = cpu_cache_size if cpu_cache_size else cache_size
        self.gpu_cache_size = gpu_cache_size
        self.min_occurence  = min_occurence
        
        self.shuffle        = shuffle
        self.random_state   = random_state
        
        self.build(** kwargs)
        
        self.unique_files, self.counts = self.get_uniques()

        self.cache  = {}
        self.cpu_files  = set()
        self.gpu_files  = set()
        self.files_to_cache = set()
        self.build_cache(
            cache_size = self.cache_size, min_occurence = min_occurence, preload = preload
        )
    
    def build(self, ** kwargs):
        raise NotImplementedError()
        
    @property
    def all_files(self):
        raise NotImplementedError()

    @property
    def cache_size(self):
        return self.cpu_cache_size + self.gpu_cache_size
    
    @property
    def output_signature(self):
        if self._output_signature is not None:
            return self._output_signature
        raise NotImplementedError()
    
    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        fn = ops.convert_to_tensor if key in self.gpu_files else ops.convert_to_numpy
        
        value = fn(value) if not tree.is_nested(value) else tree.map_structure(fn, value)
        self.cache[key] = value
    
    def __str__(self):
        des = "{} Generator :\n".format(self.__class__.__name__.replace('Generator', ''))
        des += "- # samples : {}\n".format(len(self))
        des += "- # unique ids  : {}\n".format(len(self.ids))
        des += "- # unique files    : {} ({:.2f} %)\n".format(
            len(self.unique_files), 100 * len(self.unique_files) / len(self.all_files)
        )
        if len(self.files_to_cache) > 0:
            des += "- Cache size   : {} (loaded : {:.2f} %)".format(
                len(self.files_to_cache), 100 * len(self.cache) / max(len(self.files_to_cache), 1)
            )
        return des

    def get_uniques(self):
        return np.unique(self.all_files, return_counts = True)
    
    def get(self, row_idx):
        data     = self.dataset.iloc[row_idx]
        filename = data if isinstance(data, str) else data[self.file_column]
        if filename not in self.cache:
            data = self.load_fn(data)
            if filename not in self.files_to_cache: return data
            
            self[filename] = data
        
        return self.cache[filename]

    def update(self, data):
        for k, v in data.items(): self[k] = v
    
    def build_cache(self, cache_size = 30000, min_occurence = 2, preload = False, ** kwargs):
        if not cache_size: return
        
        # compute 'cache_size' most present files (in 2 ds)
        indexes = np.flip(np.argsort(self.counts))[: cache_size]
        indexes = indexes[self.counts[indexes] >= min_occurence]
        
        files   = self.unique_files[indexes]
        self.gpu_files  = set(files[: self.gpu_cache_size])
        self.cpu_files  = set(files[self.gpu_cache_size :])
        self.files_to_cache = set(files)
        
        if self.cache:
            self.cache = {k : v for k, v in self.cache.items() if k in self.files_to_cache}
        if preload:
            if self.cache_file and os.path.exists(self.cache_file):
                self.load_cache(self.cache_file)
            self.preload_cache(** kwargs)
            if self.cache_file: self.save_cache(self.cache_file)
    
    def preload_cache(self, tqdm = tqdm, filename = None, ** kwargs):
        if filename: self.load_cache(filename)
        
        not_cached = self.files_to_cache.difference(set(self.cache.keys()))
        if len(not_cached) > 0:
            from utils.datasets import prepare_dataset

            ds = prepare_dataset(
                self.dataset[self.dataset[self.file_column].isin(not_cached)],
                batch_size  = 0,                     
                map_fn      = self.load_fn,
                prefetch    = True,
                cache       = False,
                ** kwargs
            )
            for filename, processed in tqdm(zip(not_cached, ds), total = len(not_cached)):
                self[filename] = processed

    def load_cache(self, filename, tqdm = tqdm):
        import h5py
        
        def get_data(v):
            return v.asstr()[:] if v.dtype == object else np.array(v)

        with h5py.File(filename, 'r') as file:
            for key in tqdm(self.files_to_cache):
                if key in self.cache or key not in file: continue
                self[key] = get_data(file.get(key))
        
    def save_cache(self, filename):
        if self.cache:
            dump_data(filename = filename, data = self.cache, mode = 'a')
    