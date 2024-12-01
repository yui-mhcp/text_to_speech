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

import logging
import numpy as np

from copy import deepcopy
from abc import ABCMeta, abstractmethod

from loggers import timer
from utils import is_dataframe, load_data, dump_data
from utils.distance import compiled_distance
from utils.keras_utils import ops, TensorSpec, graph_compile

logger = logging.getLogger(__name__)

class BaseVectorsDB(metaclass = ABCMeta):
    @property
    @abstractmethod
    def vectors_dim(self):
        """ Returns the dimension of vectors """
    
    @abstractmethod
    def append_vectors(self, vectors):
        """ Append `vectors` to `self.vectors` """
    
    @abstractmethod
    def update_vectors(self, indices, vectors):
        """ Updates `self.vectors` at the given `indices` to `vectors` """
    
    @abstractmethod
    def get_vectors(self, indices):
        """ Returns `self.vectors` at the given indices """
    
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
                 data,
                 vectors    = None,
                 primary_key    = None,
                 
                 *,
                 
                 model  = None,
                 query_format   = None,
                 
                 ** _
                ):
        if isinstance(primary_key, tuple) and len(primary_key) == 1: primary_key = primary_key[0]

        self._model = model
        self._vectors   = None
        self.query_format   = query_format
        self.primary_key    = primary_key
        
        self.data   = {}
        self._idx_to_data   = []
        self._data_to_idx   = {}
        
        self.update(data, vectors)

    @property
    def data_to_idx(self):
        if not self._data_to_idx:
            self._data_to_idx = {k : i for i, k in enumerate(self.idx_to_data)}
        return self._data_to_idx
    
    @property
    def idx_to_data(self):
        return self._idx_to_data
    
    @property
    def model(self):
        if isinstance(self._model, str):
            from models import get_pretrained
            self._model = get_pretrained(self._model)
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
    def length(self):
        return len(self._idx_to_data)
    
    @property
    def shape(self):
        return (self.length, self.vectors_dim)
    
    @property
    def columns(self):
        return tuple(self.data.keys())
    
    @property
    def compiled_top_k(self):
        if self.model.runtime not in ('keras', 'saved_model'): return self.top_k
        
        if getattr(self, '_compiled_top_k', None) is None:
            self._compiled_top_k = graph_compile(
                self.top_k,
                prefer_xla      = True,
                prepare_for_xla = getattr(self.model, 'prepare_for_xla', None),
                input_signature = [
                    TensorSpec(shape = (None, None), dtype = 'int32')
                ]
            )
        return self._compiled_top_k
    
    def __len__(self):
        """ Returns the number of data """
        return self.length

    def __repr__(self):
        return '<{} shape={} keys={}>'.format(
            self.__class__.__name__, self.shape, self.columns
        )

    def __str__(self):
        des = '========== {} ==========\n'.format(self.__class__.__name__)
        des += '- # data    : {}\n'.format(len(self))
        des += '- Dimension : {}\n'.format(self.vectors_dim)
        des += '- Columns (primary {}) : {}\n'.format(self.primary_key, tuple(self.keys()))
        return des

    def __contains__(self, item):
        """ Returns whether `item` is in `self.data_to_idx` (i.e., is an entry in the database) """
        if isinstance(item, dict): item = get_data_key(item, self.primary_key)
        return item in self.data_to_idx

    def __iter__(self):
        """ Iterates over the rows of the database """
        for i in range(len(self)): yield self[i]
    
    def __getitem__(self, item):
        """
            Returns the desired `item` depending on its type
            - int   : the row at the given index, a `dict` with `self.columns` + 'vector' entries
            - str (column name) : the list of all values for the given column
            - str (entry)       : the row at the given index (corresponding to the database netry)
            - str ('vector')    : equivalent to `self.vectors`
            - tuple (row, col)  : the value at the given index (index or entry) for the given column
            - range / list / array  : a copy of `self` containing only the list of indexes
        """
        if isinstance(item, str):
            if item == 'vector':        return self.vectors
            elif item in self.columns:  return self.data[item]
            elif item in self.data_to_idx:  item = self.data_to_idx[item]
            else:   raise KeyError('Entry `{}` not found'.format(item))
        
        elif isinstance(item, tuple):
            assert len(item) == 2
            index, col = item
            if col != 'vector' and col not in self.data:
                raise KeyError('The column {} is invalid'.format(col))
            
            if isinstance(index, str):
                index = self.data_to_idx[index]
                
            if col == 'vector': return self.get_vectors(index)
            return self.data[col][index]
        
        if isinstance(item, int):
            result = {'vector' : self.get_vectors(item)}
            for k, v in self.items(): result[k] = v[item]
            return result
        
        elif isinstance(item, range):
            indexes = list(item)
        elif isinstance(item, list):
            indexes = [it if isinstance(it, int) else self.index(it) for it in item]
        elif ops.is_array(item):
            if ops.is_bool(item):
                indexes = np.where(ops.convert_to_numpy(item))[0]
            else:
                indexes = ops.convert_to_numpy(item)
        else:
            raise ValueError('Unsupported index : {}'.format(item))
        
        config = self.get_config()
        config.update({
            'vectors'   : self.get_vectors(indexes),
            'data'      : {k : [v[idx] for idx in indexes] for k, v in self.data.items()}
        })
        return self.__class__(** config)
        
    def __setitem__(self, key, value):
        if isinstance(key, int):
            if not isinstance(value, dict):
                assert isinstance(self.primary_key, str)
                value = {self.primary_key : value}

            if key == -1:
                key = len(self)
                for k in self.keys(): self.data[k].append(None)
                self.idx_to_data.append(None)
            
            for k, v in value.items():
                if k not in self.data: self.data[k] = [None] * max(key + 1, len(self))
                self.data[k][key] = v

            self._idx_to_data[key] = get_data_key(value, self.primary_key)
            self._data_to_idx   = {}
        
            if logger.isEnabledFor(logging.DEBUG):
                if key == -1:
                    logger.debug('Adding new data with entry {}'.format(self._idx_to_data[key]))
                else:
                    logger.debug('Updating row {}'.format(key))

        elif isinstance(key, tuple):
            idx, col = key
            assert 0 <= idx < len(self), 'index {} is invalid'.format(idx)
            
            self.data[col][idx] = value
            if isinstance(self.primary_key, str) and col == self.primary_key:
                self._idx_to_data[idx] = value
                self._data_to_idx   = {}
            elif isinstance(self.primary_key, tuple) and col in self.primary_key:
                self._idx_to_data[idx] = get_data_key(
                    {k : v[idx] for k, v in self.items()}, self.primary_key
                )
                self._data_to_idx   = {}
        
        else:
            raise ValueError('Unsupported key : {}'.format(key))
    
    def index(self, key, value = None):
        """
            Returns the index of a given data
            If both `key` and `value` are provided, the function returns the index of `value` in `self.data[key]`
            If only `key` is provided, then the function returns the index of `key` based on `self.primary_key`
            
            Arguments :
                - key   : the data to retrieve (if `value is None`), the data entry otherwise
                - value : the value to retrieve
            Return :
                - index : the data index (-1 if not found)
        """
        if value is None:
            if isinstance(key, dict):
                key = get_data_key(key, self.primary_key)
            return self.data_to_idx.get(key, -1)
        
        try:
            return self.data[key].index(value)
        except:
            return -1
    
    def keys(self):     return self.data.keys()
    def values(self):   return self.data.values()
    def items(self):    return self.data.items()
    def get(self, key): return self.data.get(key)

    @timer
    def update(self, data, vectors = None, indices = None):
        if hasattr(data, 'to_dict'):
            data = data.to_dict('list')
        elif isinstance(data, BaseVectorsDB):
            data, vectors = data.data, data.vectors
        
        if vectors is None:
            data    = data.copy()
            vectors = get_vectors(data)
            if isinstance(vectors, list): vectors = ops.stack(vectors, axis = 0)
        
        if ops.rank(vectors) == 1:
            vectors, data = ops.expand_dims(vectors, 0), [data]
        
        if self.primary_key is None:
            d = data[0] if isinstance(data, list) else data
            if not isinstance(d, dict) or len(d.keys()) != 1:
                raise RuntimeError('Unable to infer `primary_key` from {}'.format(d))
            self.primary_key = list(d.keys())[0]
        
        # updates self.data
        indexes = []
        for i in range(len(vectors)):
            d   = data[i] if isinstance(data, list) else {k : v[i] for k, v in data.items()}
            idx = self.index(d) if indices is None else indices[i]
            
            self[idx] = d
            indexes.append(idx)
        
        # updates then append vectors
        if any(idx != -1 for idx in indexes):
            idx_update = [i for i, idx in enumerate(indexes) if idx != -1]
            
            if len(idx_update) == len(indexes):
                return self.update_vectors(indexes, vectors)
            
            mask = np.isin(np.arange(len(indexes)), idx_update)
            self.update_vectors(idx_update, vectors[mask])

            vectors = vectors[~mask]
        
        self.append_vectors(vectors)
    
    append = update
    
    def copy(self):
        config = self.get_config()
        config.update({
            'data'  : deepcopy(config['data']),
            'vectors'   : ops.copy(self.vectors)
        })
        
        return self.__class__(** config)
    
    def select_and_update(self, indices, ** kwargs):
        subset = self[indices]
        if kwargs:
            assert all(len(v) == len(subset) for v in kwargs.values())
            subset.data = {** subset.data, ** kwargs}
        return subset
    
    @timer
    def compute_scores(self, query, method = None, ** kwargs):
        if ops.is_int(query):
            query = self.model(query)
            if isinstance(query, dict): query = query['dense']
        
        if method is None:      method = self.model.distance_metric
        return compiled_distance(
            query, self.vectors, method, mode = 'similarity', as_matrix = True, ** kwargs
        )
    
    @timer
    def search(self, query, k = 10, reverse = False, ** kwargs):
        tokens = query
        if not ops.is_array(query):
            tokens = self.model.get_input(query, format = self.query_format)
        
        if ops.rank(tokens) == 1: tokens = ops.expand_dims(tokens, axis = 0)
        
        ranking, scores = self.compiled_top_k(tokens, k = min(k, len(self)), ** kwargs)
        
        ranking, scores = ops.convert_to_numpy(ranking), ops.convert_to_numpy(scores)
        if reverse: ranking, scores = ranking[:, ::-1], scores[:, ::-1]
        
        if len(tokens.shape) == 1 or tokens.shape[0] == 1:
            return self.select_and_update(ranking[0], score = scores[0])
        
        return [
            self.select_and_update(rank, score = score)
            for rank, score in zip(ranking, scores)
        ]
    
    def get_config(self):
        return {
            'data'  : self.data,
            'vectors'   : self.vectors,
            'primary_key'   : self.primary_key,
            
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

def get_data_key(data, key):
    return data[key] if isinstance(key, str) else tuple(data[k] for k in key)

def get_vectors(data):
    if isinstance(data, list):
        return [get_vectors(d) for d in data]
    elif not isinstance(data, dict):
        raise ValueError('Unsupported type for `data` : {}\n  {}'.format(type(data), data))
    
    for k in ('vectors', 'vector', 'embeddings', 'embedding'):
        if k in data: return data.pop(k)
    raise ValueError('No vectors entry found in {}'.format(tuple(data.keys())))
