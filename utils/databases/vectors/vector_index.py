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

from abc import ABC, abstractmethod

from utils import dump_json, load_json

class VectorIndex(ABC):
    @abstractmethod
    def __len__(self):
        """ Return the number of vectors in the index """
    
    @abstractmethod
    def __getitem__(self, index):
        """ Return the vectors at the given `index`(es) """

    @abstractmethod
    def add(self, vectors, ** kwargs):
        """ Add `vectors` to the index """
    
    @abstractmethod
    def remove(self, index):
        """ Remove the vectors at the given `index`(es) """

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

    @abstractmethod
    def load_vectors(self, filename):
        """ Load the index from `filename` """

    @abstractmethod
    def save_vectors(self, filename):
        """ Save the index to `filename` """

    
    def __init__(self, embedding_dim, * _, metric = 'cosine', vectors = None, ** kwargs):
        self.metric = metric
        self.embedding_dim  = embedding_dim
        
        self._vectors   = self.load_vectors(vectors) if isinstance(vectors, str) else None

    @property
    def shape(self):
        return (len(self), self.embedding_dim)
    
    @property
    def vectors(self):
        return self._vectors
    
    @vectors.setter
    def vectors(self, value):
        self._vectors = value

    def __repr__(self):
        return '<{} shape={}>'.format(self.__class__.__name__, self.shape)

    def __str__(self):
        des = '========== {} ==========\n'.format(self.__class__.__name__)
        des += '- # vectors : {}\n'.format(len(self))
        des += '- Dimension : {}\n'.format(self.embedding_dim)
        return des
    
    def get_config(self):
        return {
            'class_name'    : self.__class__.__name__,
            'metric'    : self.metric,
            'embedding_dim' : self.embedding_dim
        }
    
    def save(self, filename):
        dump_json(filename + '-config.json', self.get_config(), indent = 4)
        self.save_vectors(filename)
    
    @classmethod
    def load(cls, path, ** kwargs):
        """ Load the database from the given path """
        kwargs.update(VectorIndex.load_config(path))
        
        cls_name = kwargs.pop('class_name', None)
        if cls_name and cls_name != cls.__name__:
            raise ValueError('{} is restored but expected {}'.format(cls_name, cls.__name__))
        
        return cls(** kwargs)
    
    @staticmethod
    def load_config(path):
        config_file = path + '-config.json'
        
        config = load_json(config_file, default = {})
        config['vectors'] = path
        return config
