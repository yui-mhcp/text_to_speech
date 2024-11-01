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

import warnings
import numpy as np

from sklearn.utils import shuffle as sklearn_shuffle

from utils.keras_utils import TensorSpec
from .file_cache_generator import FileCacheGenerator

class GE2EGenerator(FileCacheGenerator):
    def __init__(self, dataset, n_samples, id_column = 'id', batch_size = None, ** kwargs):
        self.id_column  = id_column
        self.batch_size = batch_size
        self.shuffle_groups = None

        super().__init__(dataset, n_samples = n_samples, ** kwargs)

    def build(self,
              n_samples = None,
              
              n_round   = 100,
              min_round_size    = None,
              
              max_ids   = None,
              max_length    = None,
              max_repeat    = 5,
              
              tqdm  = lambda x: x,

              ** kwargs
             ):
        if n_samples is None:    n_samples = self.n_samples
        self.n_samples = n_samples

        rnd = np.random.RandomState(kwargs.get('random_state', self.random_state))
        
        self.ids    = []
        self.groups = []
        self.group_ids  = []
        
        id_counts   = self.dataset[self.id_column].value_counts()
        id_counts   = id_counts[id_counts >= n_samples]
        if max_ids: id_counts = id_counts[: max_ids]
        
        if not min_round_size:
            min_round_size = self.batch_size // n_samples if self.batch_size else 1
        
        self.ids    = {id_name : i for i, id_name in enumerate(id_counts.index)}
        
        self.dataset.reset_index(inplace = True, drop = True)
        
        groups  = [
            (data_id, data.index, np.zeros((len(data),)))
            for data_id, data in self.dataset.groupby(self.id_column)
            if data_id in self.ids
        ]
        for i in tqdm(range(n_round)):
            round_groups, round_ids = [], []
            for data_id, files, n_repeat in groups:
                indexes = np.arange(len(files))[n_repeat < max_repeat]
                if len(indexes) < n_samples: continue
                
                indexes = rnd.choice(indexes, size = n_samples, replace = False)
                
                n_repeat[indexes] += 1
                
                round_groups.append(files[indexes])
                round_ids.append(self.ids[data_id])
            
            if len(round_ids) < min_round_size: break
            
            self.groups.extend(round_groups)
            self.group_ids.extend(round_ids)
        
        self.groups     = np.array(self.groups, dtype = np.int32)
        self.group_ids  = np.array(self.group_ids, dtype = np.int32)
        
        if self.batch_size: self.set_batch_size(self.batch_size)

    @property
    def output_signature(self):
        sign = super().output_signature
        if not isinstance(sign, tuple):
            sign = (sign, TensorSpec(shape = (1, ), dtype = 'int32'))
        return sign

    @property
    def all_files(self):
        return self.dataset[self.file_column].values[np.reshape(self.groups, [-1])]

    def __len__(self):
        return len(self.groups) * self.n_samples
    
    def __getitem__(self, idx):
        if idx == 0 and self.shuffle: self.shuffle_rounds()
        data_idx = self.groups[idx // self.n_samples, idx % self.n_samples]
        
        out = self.get(data_idx)
        if not isinstance(out, tuple):
            out = (out, self.group_ids[idx // self.n_samples, None])
        return out

    def set_batch_size(self, batch_size):
        if batch_size % self.n_samples != 0 or batch_size <= self.n_samples:
            raise ValueError('`batch_size = {}` is invalid for `n_samples = {}`'.format(
                batch_size, self.n_samples
            ))

        self.batch_size = batch_size
        
        group_size = batch_size // self.n_samples
        
        self.shuffle_groups = [0]
        
        group_ids = set()
        for start in range(0, len(self.group_ids), group_size):
            new_ids = set(self.group_ids[start : start + group_size])
            if len(new_ids) < group_size:
                self.groups = self.groups[: start]
                self.group_ids = self.group_ids[: start]
                break
            
            if len(new_ids) != len(self.group_ids[start : start + group_size]):
                warnings.warn('`batch_size = {}` is too high, duplicated ids encountered in a single batch : {}. Data will be truncated up to group {}'.format(batch_size, self.group_ids[start : start + group_size], start))
                
                self.groups = self.groups[: start]
                self.group_ids = self.group_ids[: start]
                break
            
            if new_ids.intersection(group_ids):
                group_ids = set()
                self.shuffle_groups.append(start)
            
            group_ids.update(new_ids)
        
        if len(group_ids) >= group_size: self.shuffle_groups.append(len(self.group_ids))
    
    def shuffle_rounds(self):
        assert self.shuffle_groups is not None, 'You must set `batch_size` with `set_batch_size()`'
        
        indexes = np.arange(len(self.group_ids))
        for i, start in enumerate(self.shuffle_groups[:-1]):
            end = self.shuffle_groups[i + 1]
            indexes[start : end] = sklearn_shuffle(indexes[start : end])
        
        self.groups     = self.groups[indexes]
        self.group_ids  = self.group_ids[indexes]
