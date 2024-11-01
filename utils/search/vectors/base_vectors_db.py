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

from abc import ABCMeta, abstractmethod

from utils import is_dataframe
from utils.distance import distance
from utils.keras_utils import ops

_default_keys = ('text', )

class BaseVectorsDB(metaclass = ABCMeta):
    @property
    @abstractmethod
    def shape(self):
        """ Returns a tuple `(number_of_vectors, vector_size)` """
    
    @abstractmethod
    def append_vectors(self, vectors):
        """ Append `vectors` to `self.vectors` """
    
    @abstractmethod
    def top_k(self, query, k = 10, ** kwargs):
        """
            Returns a tuple `(top_k_indices, top_k_scores)`
            
            Arguments :
                - query : a 2-D `Tensor` with shape `(n_queries, vector_size)`
                - k     : the number of best items to retrieve
            Return :
                - indices   : a 2-D `Tensor` of shape `(n_queries, k)`, the indexes of nearest data
                - scores    : a 2-D `Tensor` of shape `(n_queries, k)`, the scores of the nearest data
        """

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
    
    def set_data(self, data, key = None):
        if not key: key = self.key
        
        if isinstance(data, list):
            if isinstance(data[0], dict):
                _data = {}
                for i, d in enumerate(data):
                    for k, v in d.items():
                        if k not in _data: _data[k] = [None] * i
                        _data[k].append(v)
                    for k in _data.keys():
                        if k not in d: _data[k].append(None)
                data = _data
                
            else:
                self.key = _get_key([], key)
                self.data[self.key] = data

        if is_dataframe(data):
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
            self.model = get_pretrained(self._model)
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
    
    @property
    def model_name(self):
        return getattr(self._model, 'name', self._model)
    
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
    
    @property
    def columns(self):
        return tuple(self.data.keys())
    
    def __len__(self):
        """ Returns the number of data """
        return self.shape[0]

    def __contains__(self, item):
        return item in self.data_to_idx

    def __iter__(self):
        for i in range(len(self)): yield self[i]
    
    def __getitem__(self, item):
        if isinstance(item, str):
            if not self.data: raise RuntimeError('You must call `set_data` for str-based indexing')
            if item in self.data: return self.data[item]
            item = self.data_to_idx[item]
        
        elif ops.rank(item) == 1:
            if ops.is_bool(item):
                item = np.where(ops.convert_to_numpy(item))[0]
            
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

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if len(value) != len(self):
                raise ValueError('Expected {} values, got {}'.format(len(self), len(value)))

            self.data[key] = value
            if key == self.key: self.data_to_idx = {v : i for i, v in enumerate(value)}
        elif isinstance(key, int):
            if not isinstance(value, dict): value = {self.key : value}
            for k, v in value.items():
                self[(key, k)] = v
        elif isinstance(key, tuple):
            idx, entry = key
            if isinstance(entry, int): idx, entry = entry, idx
            
            assert isinstance(idx, int) and isinstance(entry, str), 'Invalid entry idx = {}, entry = {}'.format(idx, entry)
            if entry not in self.data: raise ValueError('Entry {} is invalid'.format(entry))
            if idx >= len(self): raise ValueError('Index {} is too high'.format(idx))
            
            if key == self.key:
                if self.data_to_idx.get(entry, idx) != idx:
                    raise ValueError('Entry {} is already in `self.data` at index {}'.format(
                        entry, self.data_to_idx[entry]
                    ))
                self.data_to_idx.pop(self.data[entry])
                self.data_to_idx[entry] = idx
            
            self.data[entry][idx] = value

    def __repr__(self):
        return '<{} shape={} keys={}>'.format(
            self.__class__.__name__, self.shape, self.columns
        )

    def __str__(self):
        des = '========== {} ==========\n'.format(self.__class__.__name__)
        if self.data: des += '- # data    : {}\n'.format(len(self))
        else:         des += '- # Vectors : {}\n'.format(len(self.raw_vectors))
        des += '- Dimension    : {}\n'.format(self.vectors_dim)
        if self.data: des += '- Data keys    : {}\n'.format(tuple(self.keys()))
        return des
    
    def keys(self):     return self.data.keys()
    def values(self):   return self.data.values()
    def items(self):    return self.data.items()
    def get(self, key): return self.data.get(key)

    def copy(self, indexes = None, ** kwargs):
        if indexes is not None and ops.is_bool(indexes):
            indexes = np.where(ops.convert_to_numpy(indexes))[0]
        
        struct = self[indexes] if indexes is not None else self.__class__(
            vectors = self.vectors,
            model   = self._model,
            key     = self.key,
            data    = self.data.copy()
        )
        for k, v in kwargs.items(): struct[k] = v
        if indexes is not None: struct['index'] = indexes
        return struct
    
    clone = copy
    
    def get_vectors(self, index):
        return get_vectors(self, index, raw = False)
    
    def get_raw_vectors(self, index):
        return get_vectors(self, index)
    
    def embed(self, query, ** kwargs):
        if self._model is None: raise RuntimeError('You must specify the `model` to use ')
        return self.model.embed(query, format = self.query_format, return_raw = True, ** kwargs)
    
    def extend_data(self, key, values):
        self.data[key].extend(values)
        
        if key == self.key:
            self.data_to_idx = {k : i for i, k in enumerate(self.idx_to_data)}
    
    def append(self, vectors):
        if ops.is_array(vectors):   vectors = {'vectors' : vectors}
        elif is_dataframe(vectors): vectors = vectors.to_dict('list')
        
        if not isinstance(vectors, (BaseVectorsDB, dict)):
            raise ValueError('Unsupported vectors (type : {}) : {}'.format(
                type(vectors), vectors
            ))
        
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
                self.data[k] = [None] * (len(self) - len(v))
                self.data[k].extend(v)
    
    def compute_scores(self, query, method = 'cosine', ** kwargs):
        return distance(
            get_vectors(query, raw = True),
            get_vectors(self, raw = True),
            method,
            mode    = 'similarity',
            as_matrix   = True,
            ** kwargs
        )

    def search(self, query, n = 10, reverse = False, ** kwargs):
        if not ops.is_array(query): query = self.embed(query, ** kwargs)
        ranking, scores = self.top_k(query, n = n, ** kwargs)
        ranking, scores = ops.convert_to_numpy(ranking), ops.convert_to_numpy(scores)
        if reverse: ranking, scores = ranking[:, ::-1], scores[:, ::-1]
        
        if len(query.shape) == 1 or query.shape[0] == 1:
            return self.clone(ranking[0], score = scores[0])
        
        return [
            self.clone(rank, score = score)
            for rank, score in zip(ranking, scores)
        ]
    
    def get_config(self):
        return {
            'vectors'   : self.vectors,
            'data'      : self.data,
            'key'       : self.key,
            'model'     : self.model_name,
            'query_format'  : self.query_format
        }
    
    def save(self, filename, ** kwargs):
        return dump_data(filename, self.get_config(), ** kwargs)

    @classmethod
    def load(cls, filename, ** kwargs):
        config = load_data(filename) if os.path.exists(filename) else {}
        if not isinstance(config, dict): config = {'vectors' : config}
        config.update(kwargs)
        return cls(** config)
    
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
