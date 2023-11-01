
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf

from utils.distance import *
from unitests import CustomTestCase, data_dir

class TestDistance(CustomTestCase):
    def setUp(self):
        n, d = 256, 64
        self.queries    = np.random.normal(size = (n, d)).astype(np.float32)
        self.points     = np.random.normal(size = (n, d)).astype(np.float32)

    def test_single_distance(self):
        self.assertEqual(
            distance(self.queries[0], self.points, method = 'manhattan'),
            np.sum(np.abs(self.points - self.queries[0]), axis = -1)
        )
        self.assertEqual(
            distance(self.queries[0], self.points, method = 'euclidian'),
            np.linalg.norm(self.points - self.queries[0], axis = -1)
        )
        self.assertEqual(
            distance(self.queries[0], self.points, method = 'dp', force_distance = True),
            1. - np.sum(self.points * self.queries[0], axis = -1)
        )
        self.assertEqual(
            distance(self.queries[0], self.points, method = 'dp', force_distance = False),
            np.sum(self.points * self.queries[0], axis = -1)
        )

    def test_multi_distance(self):
        self.assertEqual(
            distance(self.queries, self.points, method = 'manhattan'),
            np.sum(np.abs(self.points - self.queries), axis = -1)
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'euclidian'),
            np.linalg.norm(self.points - self.queries, axis = -1)
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'dp', force_distance = True),
            1. - np.sum(self.points * self.queries, axis = -1)
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'dp', force_distance = False),
            np.sum(self.points * self.queries, axis = -1)
        )

    def test_matrix_distance(self):
        euclidian = distance(
            self.queries, self.points[: len(self.points) // 2], method = 'manhattan', as_matrix = True
        )
        self.assertEqual(euclidian.shape, (len(self.queries), len(self.points) // 2))
        self.assertEqual(
            distance(self.queries, self.points, method = 'manhattan', as_matrix = True),
            np.array([np.sum(np.abs(q - self.points), axis = -1) for q in self.queries])
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'euclidian', as_matrix = True),
            np.array([np.linalg.norm(q - self.points, axis = -1) for q in self.queries])
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'dp', force_distance = True, as_matrix = True),
            np.array([1. - np.sum(q * self.points, axis = -1) for q in self.queries])
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'dp', force_distance = False, as_matrix = True),
            np.array([np.sum(q * self.points, axis = -1) for q in self.queries])
        )
    
    def test_max_matrix_size(self):
        n, d = self.queries.shape
        config = {'x' : self.queries, 'y' : self.points, 'method' : 'euclidian'}
        
        self.assertEqual(
            distance(** config, max_matrix_size = n * d, as_matrix = False),
            distance(** config, max_matrix_size = -1, as_matrix = False)
        )
        self.assertEqual(
            distance(** config, max_matrix_size = n ** 2 * d, as_matrix = True),
            distance(** config, max_matrix_size = -1, as_matrix = True)
        )

