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

import keras

from utils.keras_utils import TensorSpec, graph_compile, ops
from .kmeans_method import _kmeans
from .distance_method import distance
from .clustering import clustering_wrapper

@graph_compile(support_xla = False)
def _spectral_clustering(points : TensorSpec(shape = (None, None), dtype = 'float'),
                         k      : TensorSpec(shape = (), dtype = 'int32'),
                         
                         sigma  : TensorSpec(shape = (), dtype = 'float') = 0.1,
                         distance_metric = 'euclidian',
                         ** kwargs
                        ):
    W = distance(
        points, points, distance_metric, force_distance = True, as_matrix = True, ** kwargs
    )
    W = ops.exp(- W / (2. * sigma ** 2.))

    D = ops.eye(len(points)) * ops.sum(W, axis = -1, keepdims = True)
    L = D - W
    
    (U, _, _) = ops.svd(L, full_matrices = True, compute_uv = True)
    #(U, s, _) = np.linalg.svd(L, full_matrices = True, compute_uv = True)
    
    U = U[:, -k:]
    #U.set_shape([points.shape[0], k])
    
    centroids, assignment = _kmeans(
        U,
        k   = k,
        n_init  = None,
        init_method = 'kmeans_pp',
        distance_metric = 'euclidian',
        ** kwargs
    )
    return assignment

spectral_clustering = clustering_wrapper(_spectral_clustering)