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

import os
import logging
import unittest
import numpy as np

from absl.testing import parameterized

from utils.search.vectors import *
from unitests import CustomTestCase, data_dir, reproductibility_dir

class FakeModel:
    @property
    def runtime(self):
        return 'keras'
    
    @property
    def distance_metric(self):
        return 'euclidian'
    
    def __call__(self, query):
        return query
    
class TestVectorsDatabase(CustomTestCase):
    def setUp(self):
        self.n, self.dim = 8, 4
        self.vectors    = np.arange(self.n * self.dim).reshape((self.n, self.dim))
        self.data   = [
            {'filename' : '{}.jpg'.format(i), 'text' : 'Hello World #{}'.format(i)}
            for i in range(1, self.n + 1)
        ]
        
        self.database   = build_vectors_db(
            data = self.data, vectors = self.vectors, primary_key = 'filename', model = FakeModel()
        )
    
    def test_attributes(self, database = None, indexes = None, copy = False):
        vectors, data = self.vectors, self.data
        if database is None: database = self.database
        if indexes:          vectors, data = vectors[indexes], [data[idx] for idx in indexes]
        
        self.assertEqual(database.length, len(data))
        self.assertEqual(len(database), len(data))
        self.assertEqual(database.vectors_dim, self.dim)
        self.assertEqual(database.shape, vectors.shape)
        
        self.assertEqual(database.vectors, vectors)
        if not indexes:
            if not copy:
                self.assertTrue(database.vectors is self.vectors)
            else:
                self.assertFalse(database.vectors is self.vectors)
        
        self.assertEqual(
            database.data_to_idx, {d['filename'] : i for i, d in enumerate(data)}
        )
        self.assertEqual(database.idx_to_data, [d['filename'] for d in data])
        
        self.assertEqual(set(database.keys()), {'filename', 'text'})
    
    def test_indexing(self):
        for i in range(self.n):
            self.assertEqual(self.database[i], {** self.data[i], 'vector' : self.vectors[i]})
            
        for i, item in enumerate(self.database):
            self.assertEqual(item, {** self.data[i], 'vector' : self.vectors[i]})

        for i, item in enumerate(self.data):
            self.assertEqual(self.database[item['filename']], {** item, 'vector' : self.vectors[i]})
        
        self.assertEqual(self.database['filename'], [d['filename'] for d in self.data])
        self.assertEqual(self.database['text'], [d['text'] for d in self.data])

        self.assertEqual(self.database['vector'], self.vectors)
        
        indexes = list(range(0, self.n, 2))
        self.test_attributes(self.database[indexes], indexes)
        
        file_indexes = [self.data[idx]['filename'] for idx in indexes]
        self.test_attributes(self.database[file_indexes], indexes)
    
    def test_update(self):
        database = self.database.copy()
        self.test_attributes(database, copy = True)
        
        database.update(build_vectors_db(
            vectors = np.zeros((2, self.dim), dtype = self.vectors.dtype),
            data    = [{'filename' : '1.jpg'}, {'filename' : '10.jpg'}],
            primary_key = 'filename'
        ))
        self.assertEqual(len(database), self.n + 1)
        
        self.assertEqual(database.vectors[0], np.zeros((self.dim, ), dtype = self.vectors.dtype))
        self.assertEqual(database.vectors[-1], np.zeros((self.dim, ), dtype = self.vectors.dtype))
        
        self.assertEqual(database[0, 'text'], 'Hello World #1')
        self.assertEqual(database[-1, 'text'], None)
        
        indexes = list(range(1, self.n))
        self.test_attributes(database[indexes], indexes)
    
    def test_dense_search(self):
        subset = self.database.search(
            np.zeros((self.dim, ), dtype = self.vectors.dtype), k = 3, run_eagerly = True
        )
        self.assertTrue('score' in subset.keys(), str(tuple(subset.keys())))
        subset.data.pop('score')
        self.test_attributes(subset, indexes = [0, 1, 2])
        
        subset = self.database.search(
            np.zeros((self.dim, ), dtype = self.vectors.dtype), k = 10, run_eagerly = True
        )
        subset.data.pop('score')
        self.test_attributes(subset, list(range(self.n)))
        
        subset = self.database.search(self.vectors[-1], k = 3, run_eagerly = True)
        subset.data.pop('score')
        self.test_attributes(subset, indexes = [self.n - 1, self.n - 2, self.n - 3])

