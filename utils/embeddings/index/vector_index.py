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

from abc import ABC, abstractmethod

from ...keras import ops
from ...distances import distance

class VectorIndex(ABC):
    @abstractmethod
    def concat_embeddings(self, embeddings):
        """ Concatenate a list of embeddings """
    
    @abstractmethod
    def prepare_embeddings(self, embeddings):
        """ Converts `embeddings` to a 2D array of the correct format """
        
    def __init__(self, *, metric = 'cosine', embeddings = None, ** _):
        self.metric = metric
        
        self._embeddings    = [] if embeddings is None else self.prepare_embeddings(embeddings)
        self.effective_metric   = 'dp' if metric == 'cosine' else metric
    
    @property
    def shape(self):
        return (len(self), self.embedding_dim)
    
    @property
    def dtype(self):
        return self._embeddings[0].dtype
    
    @property
    def embedding_dim(self):
        return self._embeddings[0].shape[-1]
    
    @property
    def embeddings(self):
        if isinstance(self._embeddings, list):
            if len(self._embeddings) == 1:
                self._embeddings = self._embeddings[0]
            else:
                self._embeddings = self.concat_embeddings(self._embeddings)
        return self._embeddings
    
    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value

    def __len__(self):
        if isinstance(self._embeddings, list):
            return sum(len(emb) for emb in self._embeddings)
        else:
            return len(self._embeddings)
    
    def __repr__(self):
        return '<{} shape={}>'.format(self.__class__.__name__, self.shape)

    def __str__(self):
        return self.__repr__()
    
    def __getitem__(self, index):
        """ Return the embeddings at the given `index`(es) """
        if isinstance(index, list): index = np.array(index)
        return self.embeddings[index]
    
    def add(self, embeddings, ** kwargs):
        """ Add `vectors` to the index """
        embeddings = self.prepare_embeddings(embeddings)
        if self.metric == 'cosine': embeddings = ops.normalize(embeddings)
        
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

    def remove(self, index):
        """ Remove the embeddings at the given `index`(es) """
        if isinstance(index, int):      index = np.array([index], np.int32)
        elif isinstance(index, list):   index = np.array(index, np.int32)
        
        if np.any(index >= len(self)):
            raise IndexError('Some indexes are higher or equal than {}: {}'.format(
                len(self), index
            ))
        
        mask = np.ones((len(self), ), dtype = bool)
        mask[index] = False
        
        self.embeddings = self[mask]

    def top_k(self, query, k = 10, *, mask = None, ** kwargs):
        """
            Returns a tuple `(top_k_indices, top_k_scores)`

            Arguments :
                - query : a 2-D `Tensor` with shape `(n_queries, embedding_dim)`
                - k     : the number of best items to retrieve
                - mask  : subset of candidate rows, either a list/array of indices, or a
                          boolean mask over the whole index
            Return :
                - indices   : a 2-D `Tensor` of shape `(n_queries, k)`, the indexes of nearest data
                              When `mask` is provided, this is a `np.ndarray` of **global** indices
                - scores    : a 2-D `Tensor` of shape `(n_queries, k)`, the scores of the nearest data
        """
        query = self.prepare_embeddings(query)
        if self.metric == 'cosine': query = ops.normalize(query)

        if mask is not None: mask = mask_to_indices(mask)

        embeddings = self.embeddings if mask is None else self[mask]
        distance_matrix = distance(
            query, embeddings, self.effective_metric, as_matrix = True, mode = 'similarity'
        )
        dists, indices = ops.top_k(distance_matrix, min(k, int(embeddings.shape[0])))
        if mask is not None:
            indices = mask[ops.convert_to_numpy(indices)]
        return indices, dists

    def save(self, filename):
        """ Save the index to `filename` """
        if not filename.endswith('.npy'): filename += '.npy'
        np.save(filename, ops.convert_to_numpy(self.embeddings))

    def get_config(self):
        return {'index' : self.__class__.__name__, 'metric' : self.metric}

    @classmethod
    def load(cls, filename, ** kwargs):
        if not filename.endswith('.npy'): filename += '.npy'
        return cls(embeddings = np.load(filename) if os.path.exists(filename) else None, ** kwargs)

def mask_to_indices(mask):
    """ Normalizes a (boolean or index) `mask` to a 1-D `np.ndarray` of indices """
    mask = ops.convert_to_numpy(mask)
    if mask.dtype == bool:  mask = np.where(mask)[0]
    elif not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int32)
    return mask