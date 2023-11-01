
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

from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.datasets import make_blobs, make_moons

from utils.distance import knn, kmeans, evaluate_clustering, find_clusters
from utils import sample_df, compute_centroids, get_embeddings_with_ids
from unitests import CustomTestCase, data_dir

class TestDistance(CustomTestCase):
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

    def test_embeddings(self):
        points_x, points_y = sklearn_shuffle(self.points_x, self.points_y, random_state = 10)
        centroid_ids, centroids = compute_centroids(points_x, points_y)
        
        self.assertEqual(set(centroid_ids.numpy().tolist()), set(points_y.tolist()))
        self.assertEqual(
            centroids, [np.mean(points_x[points_y == i], axis = 0) for i in centroid_ids.numpy()]
        )

        points, ids = get_embeddings_with_ids(self.points_x, self.points_y, [0, 2])
        self.assertEqual(ids, self.points_y[[yi in [0, 2] for yi in self.points_y]])
        self.assertEqual(points, self.points_x[[yi in [0, 2] for yi in self.points_y]])
    
    def test_knn(self):
        points_x, points_y = sklearn_shuffle(self.points_x, self.points_y, random_state = 10)
        for idx, (x, y) in enumerate(zip(points_x, points_y)):
            indices = [i for i in range(len(points_x)) if i != idx]
            sub_x, sub_y = points_x[indices], points_y[indices]

            self.assertEqual(
                knn(x, sub_x, distance_metric = 'euclidian', ids = sub_y), [y]
            )
            self.assertEqual(
                knn(x, sub_x, distance_metric = 'euclidian', ids = sub_y, weighted = True), [y]
            )
            
            many_x = x + np.random.uniform(-0.1, 0.1, size = (16, len(x)))
            self.assertEqual(
                knn(many_x, sub_x, distance_metric = 'euclidian', ids = sub_y), [y] * len(many_x)
            )

    def test_clustering(self):
        for i in range(4):
            self.assertEqual(
                evaluate_clustering(self.points_y, np.roll(self.points_y, i * 4))[0], 1.
            )
        
        for method in ('kmeans', 'label_propagation'):
            with self.subTest(method = method):
                centroids, assignment = find_clusters(
                    self.blobs_x, method = method, k = range(5, 10)
                )
                self.assertEqual(len(centroids), len(np.unique(self.blobs_y)))
                self.assertEqual(evaluate_clustering(self.blobs_y, assignment)[0], 1.)
            
        for method in ('spectral_clustering', 'label_propagation'):
            with self.subTest(method = method):
                centroids, assignment = find_clusters(self.moons_x, method = method, k = 2)
                self.assertEqual(len(centroids), len(np.unique(self.moons_y)))
                self.assertEqual(evaluate_clustering(self.moons_y, assignment)[0], 1.)
