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

from loggers import timer
from ..keras import ops
from ..embeddings.index import VectorIndex, init_index
from .database import _match_filters
from .ordered_database_wrapper import OrderedDatabaseWrapper

class VectorDatabase(OrderedDatabaseWrapper):
    """
        Generic vector store : a regular `Database` (entries = `dict` with a primary key)
        associated to a `VectorIndex` (1 vector per entry, in insertion order).

        The index configuration (backend + metric) is stored in the database config, and is
        therefore automatically restored when re-loading the database from its `path`.
    """
    def __init__(self,
                 path,
                 primary_key,
                 *,

                 index  = 'NumpyIndex',
                 metric = 'cosine',
                 database   = 'JSONDatabase',

                 vector_key = 'embedding',

                 ** kwargs
                ):
        super().__init__(path, primary_key, database = database, ** kwargs)

        self.vector_key = vector_key

        if not isinstance(index, VectorIndex):
            index = init_index(index, filename = self.index_path, metric = metric, ** kwargs)

        self._index = index

    @property
    def index_path(self):
        if not self.path or not os.path.exists(self.path):
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

    def insert(self, data, ** kwargs):
        data    = data.copy()
        vector = data.pop(self.vector_key)
        entry  = super().insert(data, ** kwargs)
        self.vectors.add(vector)
        return entry

    def update(self, data):
        if self.vector_key in data:
            data = data.copy()
            data.pop(self.vector_key)

        return super().update(data)

    def pop(self, key, ** kwargs):
        """ Remove and return the given entry from the database """
        entry = self._get_entry(key)
        index = self.index(entry)
        item  = super().pop(entry, ** kwargs)
        self.vectors.remove(index)

        return item

    def multi_insert(self, iterable, /, vectors = None, ** kwargs):
        if vectors is None:
            iterable = [data.copy() for data in iterable]
            vectors  = np.array([data.pop(self.vector_key) for data in iterable])

        entries = super().multi_insert(iterable, ** kwargs)
        self.vectors.add(vectors)
        return entries

    def multi_pop(self, iterable, /, ** kwargs):
        entries = [self._get_entry(data) for data in iterable]
        indexes = [self.index(entry) for entry in entries]

        items   = super().multi_pop(entries)
        self.vectors.remove(indexes)

        return items

    @timer
    def search(self, query, k = 10, *, filters = None, reverse = False, ** kwargs):
        """
            Returns the `k` entries most similar to (each of) the `query` vector(s)

            Arguments :
                - query : a 1-D or 2-D array of query embedding(s)
                - k     : the number of entries to retrieve per query
                - filters   : `dict` of metadata filters (see `_match_filters`) restricting
                              the candidate entries (e.g., `{'source' : lambda s: s in sources}`)
                - reverse   : whether to return the results in ascending score order
                              (i.e., most relevant last)
            Return :
                - results   : a list (1 item per query) of lists of entries (`dict`),
                              each enriched with a `score` entry
        """
        if not len(self):
            query = self.vectors.prepare_embeddings(query)
            return [[] for _ in range(int(query.shape[0]))]

        mask = None
        if filters:
            mask = [
                idx for idx, entry in enumerate(self._idx_to_entry)
                if _match_filters(filters, self.get(entry))
            ]
            if not mask:
                query = self.vectors.prepare_embeddings(query)
                return [[] for _ in range(int(query.shape[0]))]

        indexes, scores = self.vectors.top_k(query, k = k, mask = mask, ** kwargs)
        indexes = ops.convert_to_numpy(indexes)
        scores  = ops.convert_to_numpy(scores)
        if reverse: indexes, scores = indexes[:, ::-1], scores[:, ::-1]

        results = [self[idx] for idx in indexes.tolist()]
        for res_list, score_list in zip(results, scores):
            for res, score in zip(res_list, score_list): res['score'] = float(score)

        return results

    def save_data(self, ** kwargs):
        """ Save the database """
        super().save_data(** kwargs)
        if self.index_path: self._index.save(self.index_path, ** kwargs)

    def get_config(self):
        return {
            ** super().get_config(),

            'vector_key'    : self.vector_key,
            'index' : self._index.get_config()
        }
