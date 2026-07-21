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

from ...keras import ops
from .vector_index import VectorIndex, mask_to_indices

class TorchIndex(VectorIndex):
    def concat_embeddings(self, embeddings):
        """ Concatenate a list of embeddings """
        import torch
        return torch.concat(embeddings, 0)

    def prepare_embeddings(self, embeddings):
        """ Converts `embeddings` to a 2D array of the correct format """
        embeddings = ops.convert_to_torch_tensor(embeddings)
        if len(embeddings.shape) == 1: embeddings = embeddings.unsqueeze(0)
        return embeddings

    def add(self, embeddings, ** kwargs):
        """ Add `vectors` to the index """
        import torch
        
        embeddings = self.prepare_embeddings(embeddings)
        if self.metric == 'cosine':
            embeddings = embeddings / torch.norm(embeddings, dim = 1, keepdim = True)
        
        if len(self._embeddings) == 0:
            self.embeddings = [embeddings]
        elif embeddings.shape[-1] != self.embedding_dim:
            raise ValueError('Expected dim {}, got {}'.format(
                self.embedding_dim, embeddings.shape[-1]
            ))
        elif isinstance(self._embeddings, list):
            self._embeddings.append(embeddings)
        else:
            self._embeddings = [self._embeddings, embeddings]

    def top_k(self, query, k = 10, *, mask = None, ** kwargs):
        """
            Returns a tuple `(top_k_indices, top_k_scores)`
            
            Arguments :
                - query : a 2-D `Tensor` with shape `(n_queries, vector_size)`
                - k     : the number of best items to retrieve
                - mask  : subset of candidate rows (see `VectorIndex.top_k`)
            Return :
                - indices   : a 2-D `Tensor` of shape `(n_queries, k)`, the indexes of nearest data
                              When `mask` is provided, this is a `np.ndarray` of **global** indices
                - scores    : a 2-D `Tensor` of shape `(n_queries, k)`, the scores of the nearest data
        """
        import torch

        query = self.prepare_embeddings(query)

        if mask is not None: mask = mask_to_indices(mask)

        embeddings = self.embeddings if mask is None else self[mask]

        if self.metric == 'dp':
            distance_matrix = torch.einsum('ik,jk->ij', query, embeddings)
        elif self.metric == 'cosine':
            distance_matrix = torch.einsum(
                'ik,jk->ij', query / torch.norm(query, dim = 1, keepdim = True), embeddings
            )
        elif self.metric == 'euclidian':
            xx = torch.einsum('...i, ...i -> ...', query, query)[:, None]
            yy = torch.einsum('...i, ...i -> ...', embeddings, embeddings)[None, :]
            xy = torch.einsum('ik,jk->ij', query, embeddings)
            
            distance_matrix = - torch.sqrt(xx - 2 * xy + yy)

        dists, indices = torch.topk(distance_matrix, min(k, distance_matrix.shape[1]))
        if mask is not None:
            indices = mask[ops.convert_to_numpy(indices)]
        return indices, dists