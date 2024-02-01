
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

import tensorflow as tf

from utils.tensorflow_utils import tf_compile
from utils.distance.k_means import _kmeans
from utils.distance.distance_method import tf_distance
from utils.distance.clustering import clustering_wrapper

@tf_compile(reduce_retracing = True, follow_type_hints = True, cast_defaults = True)
def _spectral_clustering(points : tf.TensorSpec(shape = (None, None), dtype = tf.float32),
                         k      : tf.TensorSpec(shape = (), dtype = tf.int32),
                         
                         sigma  : tf.TensorSpec(shape = (), dtype = tf.float32) = 0.1,
                         distance_metric = 'euclidian',
                         ** kwargs
                        ):
    W = tf_distance(
        points, points, distance_metric, force_distance = True, as_matrix = True
    )
    W = tf.exp(- W / (2. * sigma ** 2.))

    D = tf.eye(tf.shape(W)[0]) * tf.reduce_sum(W, axis = -1, keepdims = True)
    L = D - W
    
    (s, U, _) = tf.linalg.svd(L, full_matrices = True, compute_uv = True)
    #(U, s, _) = np.linalg.svd(L, full_matrices = True, compute_uv = True)
    
    U = U[:, -k:]
    #U.set_shape([points.shape[0], k])
    
    centroids, assignment = _kmeans(
        U, k = k, distance_metric = 'euclidian', init_method = 'kmeans_pp', ** kwargs
    )
    return assignment

spectral_clustering = clustering_wrapper(_spectral_clustering)