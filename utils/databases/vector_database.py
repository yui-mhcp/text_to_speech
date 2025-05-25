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

from loggers import timer
from .vectors import VectorIndex, init_index
from .ordered_database_wrapper import OrderedDatabaseWrapper

class VectorDatabase(OrderedDatabaseWrapper):
    def __init__(self,
                 path,
                 primary_key,
                 *,
                 
                 index  = 'NumpyIndex',
                 database   = 'JSONDatabase',
                 
                 vector_key = 'embedding',
                 vector_primary_key = None,
                 
                 ** kwargs
                ):
        super().__init__(path, primary_key, database = database, ** kwargs)
        
        self.vector_key = vector_key
        self.vector_primary_key = vector_primary_key
        
        if not isinstance(index, VectorIndex):
            index = init_index(index, path = self.index_path, ** kwargs)
        
        self._index = index
    
    @property
    def index_path(self):
        if not os.path.exists(self.path):
            return None
        elif os.path.isdir(self.path):
            return os.path.join(self.path, 'vectors.index')
        else:
            return os.path.splitext(self.path)[0] + '.index'
    
    @property
    def vectors(self):
        return self._index

    @property
    def embedding_dim(self):
        return self._index.embedding_dim
    
    def insert(self, data):
        data    = data.copy()
        vector = data.pop(self.vector_key)
        super().insert(data)
        self.vectors.add(vector)
    
    def update(self, data):
        if self.vector_key in data:
            data = data.copy()
            data.pop(self.vector_key)
        
        return super().update(data)
    
    def pop(self, key):
        """ Remove and return the given entry from the database """
        entry = self._get_entry(key)
        index = self.index(entry)
        item  = super().pop(entry)
        self.vectors.remove(index)
        
        return item
    
    def multi_insert(self, iterable, /, vectors = None):
        if vectors is None:
            iterable = [data.copy() for data in iterable]
            vectors  = np.array([data.pop(self.vector_key) for data in iterable])
        
        super().multi_insert(iterable)
        self.vectors.add(vectors)
    
    def multi_pop(self, iterable, /):
        entries = [self._get_entry(data) for data in iterable]
        indexes = [self.index(entry) for entry in iterable]
        
        items   = super().multi_pop(entries)
        self.vectors.remove(indexes)
        
        return items
    
    @timer
    def search(self, query, reverse = False, ** kwargs):
        indexes, scores = self.vectors.top_k(query, ** kwargs)
        if reverse: indexes = indexes[:, ::-1]
        
        results = [self[idx] for idx in indexes.tolist()]
        for res_list, score_list in zip(results, scores):
            for res, score in zip(res_list, score_list): res['score'] = score
        
        return results
    
    def save_data(self, ** kwargs):
        """ Save the database """
        super().save_data(** kwargs)
        self._index.save(self.index_path, ** kwargs)
    
    def get_config(self):
        return {
            ** super().get_config(),
            
            'index' : self._index.get_config(),
            
            'vector_key'    : self.vector_key,
            'vector_primary_key'    : self.vector_primary_key
        }
    