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

class TorchIndex(NumpyIndex):
    def add(self, vectors, ** kwargs):
        """ Add `vectors` to the index """
        assert vectors.shape[-1] == self.embedding_dim, 'Expected dim {}, got {}'.format(self.embedding_dim, vectors.shape[-1])
        
        import torch
        
        vectors = ops.convert_to_torch_tensor(vectors)
        if len(vectors.shape) == 1: vectors = vectors[None]
        
        if self.metric == 'cosine':
            vectors = vectors / torch.norm(vectors, dim = 1, keepdim = True)
        
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = torch.concat([self.vectors, vectors], axis = 0)

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
        import torch
        
        query = ops.convert_to_torch_tensor(query)
        
        if self.metric == 'dp':
            distance_matrix = torch.einsum('ik,jk->ij', query, self.vectors)
        elif self.metric == 'cosine':
            distance_matrix = torch.einsum(
                'ik,jk->ij', query / torch.norm(query, dim = 1, keepdim = True), self.vectors
            )
        elif self.metric == 'euclidian':
            xx = torch.einsum('...i, ...i -> ...', query, query)[:, None]
            yy = torch.einsum('...i, ...i -> ...', self.vectors, self.vectors)[None, :]
            xy = torch.einsum('ik,jk->ij', query, self.vectors)
            
            distance_matrix = - torch.sqrt(xx - 2 * xy + yy)

        dists, indices = torch.topk(distance_matrix, k)
        return indices, dists

    def load_vectors(self, filename):
        """ Load the index from `filename` """
        if not filename.endswith('.npy'): filename += '.npy'
        if not os.path.exists(filename):
            return None
        else:
            return ops.convert_to_torch_tensor(np.load(filename))

    def save_vectors(self, filename):
        """ Save the index to `filename` """
        if self.vectors is not None: np.save(filename, ops.convert_to_numpy(self.vectors))
