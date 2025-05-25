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
import unittest
import numpy as np

from functools import partial
from absl.testing import parameterized

from . import CustomTestCase, temp_dir
from utils.databases.vectors import *

class TestVectorIndex(CustomTestCase, parameterized.TestCase):
    path = None
    vector_class = None
    
    def clear(self):
        pass
    
    def testUp(self):
        self.clear()
    
    def testDown(self):
        self.clear()
    
    def create_index(self, embedding_dim = 128, metric = 'cosine', ** kwargs):
        if self.vector_class is None:
            self.skipTest("This is an abstract test class. Please define 'vector_class'")
        
        return self.vector_class(embedding_dim = embedding_dim, metric = metric, ** kwargs)
    
    def create_random_vectors(self, n_vectors = 100, embedding_dim = 128):
        return np.random.random(size = (n_vectors, embedding_dim)).astype(np.float32)
    
    @parameterized.parameters('cosine', 'euclidean', 'dot')
    def test_initial_state(self, metric):
        embedding_dim = 64
        index = self.create_index(embedding_dim = embedding_dim, metric = metric)
        
        self.assertEqual(len(index), 0)
        self.assertEqual(index.embedding_dim, embedding_dim)
        self.assertEqual(index.metric, metric)
        self.assertEqual(index.shape, (0, embedding_dim))
    
    def test_add_vectors(self):
        embedding_dim = 64
        index = self.create_index(embedding_dim = embedding_dim)
        
        n_vectors = 50
        vectors = self.create_random_vectors(n_vectors, embedding_dim)
        index.add(vectors)
        
        self.assertEqual(len(index), n_vectors)
        self.assertEqual(index.shape, (n_vectors, embedding_dim))
        
        more_vectors = self.create_random_vectors(30, embedding_dim)
        index.add(more_vectors)
        
        self.assertEqual(len(index), n_vectors + 30)
        self.assertEqual(index.shape, (n_vectors + 30, embedding_dim))
    
    def test_get_vectors(self):
        embedding_dim = 64
        index = self.create_index(embedding_dim=embedding_dim)
        
        n_vectors = 50
        vectors = self.create_random_vectors(n_vectors, embedding_dim)
        index.add(vectors)
        
        indices = [0, 10, 42]
        self.assertEqual(vectors[5], index[5])
        self.assertEqual(vectors[indices], index[indices])
        self.assertEqual(vectors[:5], index[:5])
    
    def test_remove_vectors(self):
        embedding_dim = 64
        index = self.create_index(embedding_dim=embedding_dim)
        
        n_vectors = 50
        vectors = self.create_random_vectors(n_vectors, embedding_dim)
        index.add(vectors)
        
        index.remove(5)
        self.assertEqual(len(index), n_vectors - 1)
        
        indices_to_remove = [10, 20, 30]
        index.remove(indices_to_remove)
        self.assertEqual(len(index), n_vectors - 4)
    
    def test_top_k(self):
        embedding_dim = 64
        index = self.create_index(embedding_dim=embedding_dim)
        
        n_vectors = 100
        vectors = self.create_random_vectors(n_vectors, embedding_dim)
        index.add(vectors)
        
        query = self.create_random_vectors(1, embedding_dim)
        
        k = 10
        indices, scores = index.top_k(query, k = k, run_eagerly = True)
        
        self.assertEqual(indices.shape, (1, k))
        self.assertEqual(scores.shape, (1, k))
        
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < n_vectors))
        
        if index.metric == 'euclidean':
            self.assertTrue(np.all(scores >= 0))
        
        queries = self.create_random_vectors(5, embedding_dim)
        indices, scores = index.top_k(queries, k = k, run_eagerly = True)
        
        self.assertEqual(indices.shape, (5, k))
        self.assertEqual(scores.shape, (5, k))
    
    def test_save_load(self):
        if self.path is None:
            self.skipTest("The `path` variable should be defined")
        
        embedding_dim = 64
        index = self.create_index(embedding_dim = embedding_dim)
        
        try:
            # Ajouter des vecteurs
            n_vectors = 50
            vectors = self.create_random_vectors(n_vectors, embedding_dim)
            index.add(vectors)
            
            # Sauvegarder l'index
            index.save(self.path)
            
            self.assertTrue(os.path.exists(self.path + '-config.json'),  "The configuration file does not exist")
            
            loaded_index = self.vector_class.load(self.path)
            
            self.assertEqual(loaded_index.embedding_dim, embedding_dim)
            self.assertEqual(loaded_index.metric, index.metric)
            self.assertEqual(len(loaded_index), n_vectors)
            
            self.assertEqual(vectors, loaded_index[np.arange(n_vectors)])
            self.assertEqual(index.get_config(), loaded_index.get_config())
        finally:
            self.clear()
    
    def test_error_handling(self):
        embedding_dim = 64
        index = self.create_index(embedding_dim = embedding_dim)
        
        with self.assertRaises(IndexError):
            _ = index[0]
        
        with self.assertRaises(IndexError):
            index.remove(0)
        
        wrong_dim_vectors = np.random.random((10, embedding_dim + 1)).astype(np.float32)
        with self.assertRaises(AssertionError):
            index.add(wrong_dim_vectors)
        
class TestNumpyVectorIndex(TestVectorIndex):
    path = os.path.join(temp_dir, 'test_vector_index.index')
    vector_class = NumpyIndex
    
    def clear(self):
        """Nettoie les fichiers temporaires créés pendant les tests."""
        if os.path.exists(self.path + '-config.json'):
            os.remove(self.path + '-config.json')
        if os.path.exists(self.path + '.npy'):
            os.remove(self.path + '.npy')

