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

from absl.testing import parameterized
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.datasets import make_blobs

from unitests import CustomTestCase, data_dir
from utils.embeddings import *

class TestClustering(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.n_ids = 7
        self.embeddings, self.ids = sklearn_shuffle(* make_blobs(
            n_samples = 250, n_features = 64, centers = self.n_ids, cluster_std = 1., random_state = 10
        ), random_state = 10)
        self.ids = self.ids.astype(np.int32)
        self.embeddings = self.embeddings.astype(np.float32)
        
        self.ids_t  = K.convert_to_tensor(self.ids, 'int32')
        self.embeddings_t   = K.convert_to_tensor(self.embeddings, 'float32')

    def test_centroids(self):
        true_centroids = np.array([
            np.mean(self.embeddings[self.ids == i], axis = 0) for i in range(self.n_ids)
        ])
        centroid_ids, centroids = compute_centroids(self.embeddings, self.ids, 7, run_eagerly = True)
        self.assertArray(centroids)
        self.assertEqual(centroids, true_centroids)
        
        centroid_ids, centroids = compute_centroids(self.embeddings, self.ids, run_eagerly = True)
        self.assertArray(centroids)
        self.assertEqual(centroids, true_centroids)

        centroid_ids, centroids = compute_centroids(self.embeddings_t, self.ids_t, 7, run_eagerly = True)
        self.assertTensor(centroids)
        self.assertEqual(centroids, true_centroids)

        centroid_ids, centroids = compute_centroids(self.embeddings_t, self.ids_t, run_eagerly = True)
        self.assertTensor(centroids)
        self.assertEqual(centroids, true_centroids)

        self.assertGraphCompatible(
            compute_centroids, self.embeddings, self.ids, 7, target = (np.arange(7, dtype = 'int32'), true_centroids)
        )

    def get_embeddings_with_ids(self):
        mask = np.isin(self.ids, [0, 2])
        
        true_ids = self.ids[mask]
        true_emb = self.embeddings[mask]
        
        selected, ids = get_embeddings_with_ids(self.embeddings, self.ids, [0, 2])
        self.assertArray(ids)
        self.assertArray(selected)
        self.assertEqual(ids, true_ids)
        self.assertEqual(selected, true_emb)

        selected, ids = get_embeddings_with_ids(self.embeddings_t, self.ids_t, [0, 2])
        self.assertTensor(ids)
        self.assertTensor(selected)
        self.assertEqual(ids, true_ids)
        self.assertEqual(selected, true_emb)

        self.assertGraphCompatible(
            get_embeddings_with_ids, self.embeddings, self.ids, target = (true_emb, true_ids)
        )

    @parameterized.named_parameters(
        {
            'testcase_name' : 'numpy',
            'inputs'    : np.reshape(np.arange(64), [4, 16]),
            'target'    : np.reshape(np.arange(64), [4, 16])
        },
        {
            'testcase_name' : 'tensor',
            'inputs'    : K.reshape(K.arange(64), [4, 16]),
            'target'    : K.reshape(K.arange(64), [4, 16])
        },
        {
            'testcase_name' : 'dataframe',
            'inputs'    : pd.DataFrame([
                {'id' : i, 'embedding' : emb}
                for i, emb in enumerate(np.reshape(np.arange(64), [4, 16]))
            ]),
            'target'    : np.reshape(np.arange(64), [4, 16])
        },
        {
            'testcase_name' : 'string',
            'inputs'    : str(np.arange(64) / 3.),
            'target'    : (np.arange(64) / 3.).astype(np.float32)
        }

    )
    def test_convert_to_np(self, inputs, target):
        self.assertEqual(embeddings_to_np(inputs), target)