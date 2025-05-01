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

import keras
import numpy as np
import keras.ops as K

from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.datasets import make_blobs, make_moons

from . import CustomTestCase, data_dir
from utils.distances import distance, knn

class TestNumpyDistance(CustomTestCase):
    def setUp(self):
        self.max_err    = 1e-5
        n, d = 256, 32
        self.queries    = np.random.normal(size = (n, d)).astype(np.float32)
        self.points     = np.random.normal(size = (n, d)).astype(np.float32)
        self.np_queries = self.queries
        self.np_points  = self.points

    def test_simple_distance(self):
        with self.subTest('manhattan'):
            self.assertEqual(
                distance(self.queries[0], self.points, method = 'manhattan'),
                np.sum(np.abs(self.np_points - self.np_queries[:1]), axis = -1),
                max_err = self.max_err
            )
        with self.subTest('euclidian'):
            self.assertEqual(
                distance(self.queries[0], self.points, method = 'euclidian'),
                np.linalg.norm(self.np_points - self.np_queries[0], axis = -1),
                max_err = self.max_err
            )
        with self.subTest('dot_product'):
            self.assertEqual(
                distance(self.queries[0], self.points, method = 'dp'),
                np.sum(self.np_points * self.np_queries[0], axis = -1),
                max_err = self.max_err
            )
            self.assertEqual(
                distance(self.queries[0], self.points, method = 'dp', mode = 'distance'),
                - np.sum(self.np_points * self.np_queries[0], axis = -1),
                max_err = self.max_err
            )

    def test_multi_distance(self):
        with self.subTest('manhattan'):
            self.assertEqual(
                distance(self.queries, self.points, method = 'manhattan'),
                np.sum(np.abs(self.np_points - self.np_queries), axis = -1),
                max_err = self.max_err
            )
        with self.subTest('euclidian'):
            self.assertEqual(
                distance(self.queries, self.points, method = 'euclidian'),
                np.linalg.norm(self.np_points - self.np_queries, axis = -1),
                max_err = self.max_err
            )
        with self.subTest('dot_product'):
            self.assertEqual(
                distance(self.queries, self.points, method = 'dp'),
                np.sum(self.np_points * self.np_queries, axis = -1),
                max_err = self.max_err
            )
            self.assertEqual(
                distance(self.queries, self.points, method = 'dp', mode = 'distance'),
                - np.sum(self.np_points * self.np_queries, axis = -1),
                max_err = self.max_err
            )

    def test_matrix_distance(self):
        with keras.device('cpu'):
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
                distance(self.queries, self.points, method = 'dp', mode = 'distance', as_matrix = True),
                np.array([- np.sum(q * self.np_points, axis = -1) for q in self.np_queries]),
                max_err = self.max_err
            )
            self.assertEqual(
                distance(self.queries, self.points, method = 'dp', as_matrix = True),
                np.array([np.sum(q * self.np_points, axis = -1) for q in self.np_queries]),
                max_err = self.max_err
            )

class TestTensorDistance(TestNumpyDistance):
    def setUp(self):
        super().setUp()
        self.max_err    = 1e-5
        self.queries    = K.convert_to_tensor(self.np_queries, dtype = 'float32')
        self.points     = K.convert_to_tensor(self.np_points, dtype = 'float32')

class TestTensorAndArrayDistance(TestNumpyDistance):
    def setUp(self):
        super().setUp()
        self.max_err    = 1e-5
        self.queries    = self.np_queries
        self.points     = K.convert_to_tensor(self.np_points, dtype = 'float32')

class TestKNN(CustomTestCase):
    def setUp(self):
        self.points_x = np.array([
            [1., 1.], [2., 2.], [2., 1.], [1., 2.],
            [-1., -1.], [-2., -2.], [-2., -1.], [-1., -2.],
            [-1., 1.], [-2., 2.], [-2., 1.], [-1., 2.],
            [1., -1.], [2., -2.], [2., -1.], [1., -2.]
        ], dtype = np.float32)
        self.points_y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype = np.int32)

        self.blobs_x, self.blobs_y = make_blobs(
            n_samples = 250, n_features = 128, centers = 7, cluster_std = 1., random_state = 10
        )
        self.moons_x, self.moons_y = make_moons(
            n_samples = 250, random_state = 10
        )

    def test_knn(self):
        points_x, points_y = sklearn_shuffle(self.points_x, self.points_y, random_state = 10)
        for idx, (x, y) in enumerate(zip(points_x, points_y)):
            indices = [i for i in range(len(points_x)) if i != idx]
            sub_x, sub_y = points_x[indices], points_y[indices]

            self.assertEqual(
                knn(x, sub_x, distance_metric = 'euclidian', ids = sub_y), np.array([y])
            )
            self.assertEqual(
                knn(x, sub_x, distance_metric = 'euclidian', ids = sub_y, weighted = True),
                np.array([y])
            )
            
            many_x = x + np.random.uniform(-0.1, 0.1, size = (16, len(x)))
            self.assertEqual(
                knn(many_x, sub_x, distance_metric = 'euclidian', ids = sub_y), [y] * len(many_x)
            )
