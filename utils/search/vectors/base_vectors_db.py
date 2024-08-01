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

import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

from utils.distance import distance
from utils.keras_utils import ops

_default_keys = ('text', )

class BaseVectorsDB(metaclass = ABCMeta):
    def __init__(self,
                 vectors,
                 model = None,
                 *,
                 
                 query_format   = None,
                 
                 key    = _default_keys,
                 data   = None,
                 
                 ** _
                ):
        self.key    = key
        self._model = model
        self._vectors   = vectors
        self.query_format   = query_format
        
        self.data   = {}
        self.data_to_idx    = {}
        
        if data is not None: self.set_data(data, key = key)

    @property
    @abstractmethod
    def shape(self):
        pass
    
    @abstractmethod
    def append_vectors(self, vectors):
        pass
    
    @abstractmethod
    def top_k(self, query, k = 10, ** kwargs):
        pass

    
    def set_data(self, data, key = None):
        if not key: key = self.key
        
        if isinstance(data, list):
            if isinstance(data[0], dict):
                data = pd.DataFrame(data)
            else:
                self.key = _get_key([], key)
                self.data[self.key] = data

        if isinstance(data, pd.DataFrame):
            data = data.to_dict('list')
        
        if isinstance(data, dict):
            self.key = _get_key(list(data.keys()), key)
            self.data.update(data)

        self.data_to_idx = {k : i for i, k in enumerate(self.idx_to_data)}
        
        if not self.data:
            raise RuntimeError('Unsupported `data` type : {}\n{}'.format(type(data), data))

    @property
    def model(self):
        if isinstance(self._model, str):
            from models import get_pretrained
            return get_pretrained(self._model)
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
        
    @property
    def vectors(self):
        return self._vectors
    
    @vectors.setter
    def vectors(self, value):
        self._vectors = value
    
    @property
    def raw_vectors(self):
        return get_vectors(self.vectors)
    
    @property
    def vectors_dim(self):
        return self.shape[1]
    
    @property
    def idx_to_data(self):
        return self.data.get(self.key, [])
    
    def __iter__(self):
        for i in range(len(self)): yield self[i]
        
    def __str__(self):
        des = '========== {} ==========\n'.format(self.__class__.__name__)
        if self.data: des += '- # data    : {}\n'.format(len(self))
        des += '- # Vectors : {}\n'.format(len(self.raw_vectors))
        des += '- Dimension    : {}\n'.format(self.vectors_dim)
        if self.data: des += '- Data keys    : {}\n'.format(tuple(self.keys()))
        return des
    
    def __repr__(self):
        return '<{} shape={} keys={}>'.format(
            self.__class__.__name__, self.shape, tuple(self.keys())
        )
    
    def __len__(self):
        """ Returns the number of data """
        return self.shape[0]
    
    def __contains__(self, item):
        return item in self.data_to_idx
    
    def __setitem__(self, key, value):
        if len(value) != len(self):
            raise ValueError('Expected {} values, got {}'.format(len(self), len(value)))
        
        self.data[key] = value
    
    def __getitem__(self, item):
        if isinstance(item, str):
            if not self.data: raise RuntimeError('You must call `set_data`')
            item = self.data_to_idx[item]
        
        elif ops.rank(item) == 1:
            if ops.is_bool(item):
                item = ops.where(ops.convert_to_numpy(item))[0]
            
            return self.__class__(
                vectors = self.get_vectors(item),
                model   = self._model,
                key     = self.key,
                data    = {
                    k : [v[it] for it in item] if not ops.is_array(v) else ops.take(v, item, 0)
                    for k, v in self.data.items()
                }
            )
        
        return {
            'vectors'   : self.get_vectors(item),
            ** {k : v[item] for k, v in self.data.items()}
        }
    
    def clone(self, indexes = None, ** kwargs):
        if indexes is not None and ops.is_bool(indexes):
            indexes = ops.where(ops.convert_to_numpy(indexes))[0]
        
        struct = self[indexes] if indexes is not None else self.__class__(
            vectors = self.vectors,
            model   = self._model,
            key     = self.key,
            data    = self.data.copy()
        )
        for k, v in kwargs.items(): struct[k] = v
        if indexes is not None: struct['indexes'] = indexes
        return struct
    
    def keys(self):     return self.data.keys()
    def values(self):   return self.data.values()
    def items(self):    return self.data.items()
    def get(self, key): return self.data.get(key)
    
    def get_vectors(self, index):
        return get_vectors(self, index, raw = False)
    
    def get_raw_vectors(self, index):
        return get_vectors(self, index)
    
    def embed(self, query, ** kwargs):
        if self._model is None: raise RuntimeError('You must specify the `model` to use ')
        return self.model.embed(query, format = self.query_format, return_raw = True, ** kwargs)
    
    def extend_data(self, key, values):
        if key not in self.data: self.data[key] = [None] * len(self)
        self.data[key].extend(values)
        
        if key == self.key:
            self.data_to_idx = {k : i for i, k in enumerate(self.idx_to_data)}
    
    def append(self, vectors):
        if ops.is_array(vectors): vectors = {'vectors' : vectors}
        elif isinstance(vectors, pd.DataFrame): vectors = vectors.to_dict('list')
        
        if isinstance(vectors, (BaseVectorsDB, dict)):
            if isinstance(vectors, dict):
                _vectors_key = 'vectors' if 'vectors' in vectors else 'embedding'
                self.append_vectors(vectors[_vectors_key])
            else:
                _vectors_key = None
                self.append_vectors(vectors.vectors)
                vectors = vectors.data
            
            for k, v in self.items():
                if k in vectors:
                    self.extend_data(k, vectors[k])
                else:
                    self.extend_data(k, [None] * (len(self) - len(v)))
            
            for k, v in vectors.items():
                if k not in self.data and k != _vectors_key:
                    self.extend_data(v)
            
        else:
            raise ValueError('Unsupported vectors (type : {}) : {}'.format(
                type(vectors), vectors
            ))
    
    def compute_scores(self, query, method = 'cosine', ** kwargs):
        return distance(
            get_vectors(query, raw = True),
            get_vectors(self, raw = True),
            method,
            mode    = 'similarity',
            as_matrix   = True,
            ** kwargs
        )

    def search(self, query, n = 10, ** kwargs):
        if isinstance(query, (list, str)): query = self.embed(query, ** kwargs)
        ranking, scores = self.top_k(query, n = n, ** kwargs)
        ranking, scores = ops.convert_to_numpy(ranking), ops.convert_to_numpy(scores)

        if len(query.shape) == 1 or query.shape[0] == 1:
            return self.clone(ranking[0], scores = scores[0])
        
        return [
            self.clone(rank, scores = score)
            for rank, score in zip(ranking, scores)
        ]

def get_vectors(vectors, index = None, raw = True):
    if hasattr(vectors, 'vectors'): vectors = vectors.vectors
    
    if raw or ops.is_array(vectors): return get_raw_vectors(vectors, index)
    elif index is None: return vectors
    
    assert isinstance(vectors, dict), 'Unsupported vectors type : {}'.format(vectors)
    
    if 'lengths' not in vectors:
        if ops.rank(index) == 0:
            return {k : v[index] for k, v in vectors.items()}
        return {k : ops.take(v, index, axis = 0) for k, v in vectors.items()}
    
    
    cum_lengths = ops.cumsum(vectors['lengths'])
    if ops.rank(index) == 0:
        return {
            k : v[index] if len(v) == len(cum_lengths) else _get_vectors_slice(v, index, cum_lengths)
            for k, v in vectors.items()
        }
    
    return {
        k : ops.take(v, index, axis = 0) if len(v) == len(cum_lengths) else ops.concatenate([
            _get_vectors_slice(v, idx, cum_lengths) for idx in index
        ], axis = 0) for k, v in vectors.items()
    }
    
def get_raw_vectors(vectors, index = None, lengths = None):
    if ops.is_array(vectors):
        if index is None: return vectors
        if lengths is None:
            if ops.rank(index) == 0: return vectors[index]
            return ops.take(vectors, index, axis = 0)
        
        cum_lengths = ops.cumsum(lengths)
        if ops.rank(index) == 0:
            return _get_vectors_slice(vectors, index, cum_lengths)
        
        return ops.concatenate([
            _get_vectors_slice(vectors, idx, cum_lengths) for idx in index
        ], axis = 0)
        
    elif isinstance(vectors, dict):
        return get_raw_vectors(vectors['vectors'], index, vectors.get('lengths', None))
    elif hasattr(vectors, 'vectors'): return get_raw_vectors(vectors.vectors, index)
    else: raise ValueError('Invalid vectors : {}'.format(vectors))

def _get_vectors_slice(vectors, index, cum_lengths):
    end     = cum_lengths[index]
    start   = cum_lengths[index - 1] if index > 0 else 0
    return vectors[start : end]

def _get_key(data, candidates):
    if isinstance(candidates, str): return candidates
    
    if isinstance(candidates, (tuple, list)):
        for k in candidates:
            if k in data: return k
    
    if len(data) == 1: return data[0]
    raise ValueError('Key {} is not in data'.format(candidates))
