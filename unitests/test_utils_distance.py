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

import numpy as np
import keras.ops as K

from utils.distance import *
from unitests import CustomTestCase, data_dir

class TestNumpyDistance(CustomTestCase):
    def setUp(self):
        self.max_err    = 1e-6
        n, d = 256, 64
        self.queries    = np.random.normal(size = (n, d)).astype(np.float32)
        self.points     = np.random.normal(size = (n, d)).astype(np.float32)
        self.np_queries = self.queries
        self.np_points  = self.points

    def test_text_metrics(self):
        self.assertEqual([1, 1, 1, 1], text_f1("Hello World !", "Hello ! World"))
        self.assertEqual([0, 1, 1, 1], text_f1("Hello World !", "Hello ! World", normalize = False))
        self.assertEqual(
            [0, 2 / 3, 2 / 3, 2 / 3], text_f1("Hello World !", "Hello ! world", normalize = False)
        )
        self.assertEqual([1, 1, 1, 1], text_f1("Hello World !", "Hello world"))
        self.assertEqual([0, 1, 1, 1], text_f1([0, 1, 2], [0, 2, 1]))
        self.assertEqual([1, 1, 1, 1], text_f1([0, 1, 2], [0, 2], exclude = [1]))
        self.assertEqual([0, 0.8, 1, 2 / 3], text_f1([0, 1, 2], [0, 2]))

    def test_single_distance(self):
        self.assertEqual(
            distance(self.queries[0], self.points, method = 'manhattan'),
            np.sum(np.abs(self.np_points - self.np_queries[0]), axis = -1),
            max_err = self.max_err
        )
        self.assertEqual(
            distance(self.queries[0], self.points, method = 'euclidian'),
            np.linalg.norm(self.np_points - self.np_queries[0], axis = -1),
            max_err = self.max_err
        )
        self.assertEqual(
            distance(self.queries[0], self.points, method = 'dp', force_distance = True),
            1. - np.sum(self.np_points * self.np_queries[0], axis = -1),
            max_err = self.max_err
        )
        self.assertEqual(
            distance(self.queries[0], self.points, method = 'dp', force_distance = False),
            np.sum(self.np_points * self.np_queries[0], axis = -1),
            max_err = self.max_err
        )

    def test_multi_distance(self):
        self.assertEqual(
            distance(self.queries, self.points, method = 'manhattan'),
            np.sum(np.abs(self.np_points - self.np_queries), axis = -1),
            max_err = self.max_err
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'euclidian'),
            np.linalg.norm(self.np_points - self.np_queries, axis = -1),
            max_err = self.max_err
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'dp', force_distance = True),
            1. - np.sum(self.np_points * self.np_queries, axis = -1),
            max_err = self.max_err
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'dp', force_distance = False),
            np.sum(self.np_points * self.np_queries, axis = -1),
            max_err = self.max_err
        )

    def test_matrix_distance(self):
        euclidian = distance(
            self.queries, self.points[: len(self.points) // 2], method = 'manhattan', as_matrix = True
        )
        self.assertEqual(euclidian.shape, (len(self.queries), len(self.points) // 2))
        self.assertEqual(
            distance(self.queries, self.points, method = 'manhattan', as_matrix = True),
            np.array([np.sum(np.abs(q - self.np_points), axis = -1) for q in self.np_queries]),
            max_err = self.max_err
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'euclidian', as_matrix = True),
            np.array([np.linalg.norm(q - self.np_points, axis = -1) for q in self.np_queries]),
            max_err = self.max_err
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'dp', force_distance = True, as_matrix = True),
            np.array([1. - np.sum(q * self.np_points, axis = -1) for q in self.np_queries]),
            max_err = self.max_err
        )
        self.assertEqual(
            distance(self.queries, self.points, method = 'dp', force_distance = False, as_matrix = True),
            np.array([np.sum(q * self.np_points, axis = -1) for q in self.np_queries]),
            max_err = self.max_err
        )
    
    def test_max_slice(self):
        if ops.is_jax_backend():
            self.fail('The `max_slice` argument is not well supported in JAX')
            return
        
        n, d = self.queries.shape
        config = {'x' : self.queries, 'y' : self.points, 'method' : 'euclidian'}
        
        self.assertEqual(
            distance(** config, max_slice = n // 4, as_matrix = False),
            distance(** config, as_matrix = False)
        )
        self.assertEqual(
            distance(** config, max_slice = 200, as_matrix = True),
            distance(** config, as_matrix = True)
        )
        self.assertEqual(
            distance(** config, max_slice_x = 200, as_matrix = True),
            distance(** config, as_matrix = True)
        )
        self.assertEqual(
            distance(** config, max_slice_y = 200, as_matrix = True),
            distance(** config, as_matrix = True)
        )

class TestTensorDistance(TestNumpyDistance):
    def setUp(self):
        super().setUp()
        self.max_err    = 5e-6
        self.queries    = K.convert_to_tensor(self.np_queries, dtype = 'float32')
        self.points     = K.convert_to_tensor(self.np_points, dtype = 'float32')

class TestTensorAndArrayDistance(TestNumpyDistance):
    def setUp(self):
        super().setUp()
        self.max_err    = 5e-6
        self.queries    = self.np_queries
        self.points     = K.convert_to_tensor(self.np_points, dtype = 'float32')
