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
from utils.distances import distance
from .numpy_index import NumpyIndex

class KerasIndex(NumpyIndex):
    def add(self, vectors, ** kwargs):
        """ Add `vectors` to the index """
        assert vectors.shape[-1] == self.embedding_dim, 'Expected dim {}, got {}'.format(self.embedding_dim, vectors.shape[-1])
        
        vectors = ops.convert_to_tensor(vectors)
        if len(vectors.shape) == 1: vectors = vectors[None]
        
        if self.metric == 'cosine':
            vectors = ops.normalize(vectors)
        
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = ops.concatenate([self.vectors, vectors], axis = 0)

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
        query = ops.convert_to_tensor(query)
        if self.metric == 'cosine':
            metric = 'dp'
            query  = ops.normalize(query)
        else:
            metric = self.metric
            
        distance_matrix = distance(
            query, self.vectors, metric, as_matrix = True, mode = 'dimilarity'
        )
        dists, indices = ops.top_k(distance_matrix, k)
        return indices, dists

    def load_vectors(self, filename):
        """ Load the index from `filename` """
        if not filename.endswith('.npy'): filename += '.npy'
        if not os.path.exists(filename):
            return None
        else:
            return ops.convert_to_tensor(np.load(filename))

    def save_vectors(self, filename):
        """ Save the index to `filename` """
        if self.vectors is not None: np.save(filename, ops.convert_to_numpy(self.vectors))
