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
import numpy as np

from utils.keras import ops
from utils.distances import knn
from .vector_index import VectorIndex

class NumpyIndex(VectorIndex):
    def __len__(self):
        """ Return the number of vectors in the index """
        return len(self.vectors) if self.vectors is not None else 0
    
    def __getitem__(self, index):
        """ Return the vectors at the given `index`(es) """
        if self.vectors is None: raise IndexError('The index is empty')
        return self.vectors[index]

    def add(self, vectors, ** kwargs):
        """ Add `vectors` to the index """
        assert vectors.shape[-1] == self.embedding_dim, 'Expected dim {}, got {}'.format(self.embedding_dim, vectors.shape[-1])
        
        vectors = ops.convert_to_numpy(vectors)
        if len(vectors.shape) == 1: vectors = vectors[None]
        
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.concatenate([self.vectors, vectors], axis = 0)

    def remove(self, index):
        """ Remove the vectors at the given `index`(es) """
        if self.vectors is None: raise IndexError('The index is empty')
        if isinstance(index, int): index = [index]
        self.vectors = self.vectors[~np.isin(np.arange(len(self)), index)]

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
        indices, scores = knn(
            query, self.vectors, distance_metric = self.metric, k = k, return_scores = True, ** kwargs
        )
        return ops.convert_to_numpy(indices), ops.convert_to_numpy(scores)

    def load_vectors(self, filename):
        """ Load the index from `filename` """
        if not filename.endswith('.npy'): filename += '.npy'
        return np.load(filename) if os.path.exists(filename) else None

    def save_vectors(self, filename):
        """ Save the index to `filename` """
        if self.vectors is not None: np.save(filename, self.vectors)
