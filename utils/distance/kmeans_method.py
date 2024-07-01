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
from utils.embeddings import compute_centroids
from utils.distance.distance_method import distance
from utils.distance.clustering import clustering_wrapper, get_assignment, compute_score

@graph_compile
def _kmeans(points  : TensorSpec(shape = (None, None), dtype = 'float'),
            k       : TensorSpec(shape = (), dtype = 'int32'),

            n_init  : TensorSpec(shape = (), dtype = 'int32') = 5,
            max_iter    : TensorSpec(shape = (), dtype = 'int32') = 100,
            threshold   : TensorSpec(shape = (), dtype = 'float') = 1e-6,
            init_method = 'kmeans_pp',
            distance_metric = 'euclidian',
            normalize       = False,
            random_state    = None,
            ** kwargs
           ):
    """
        Computes the centroids following the KMeans procedure
        
        Special arguments :
            - n_init    : the number of times to perform random initialization (only the best result is returned)
            - max_iter  : the maximal number of KMeans iteration
            - threshold : a stop criterion, the minimal difference between 2 iterations
            - init_method   : the initialization method for initial centroids
                - normal    : computes the centroids based on a normal distribution
                - random    : randomly selects `k` points
                - kmeans_pp : uses the `KMeans++` selection procedure (see `help(kmeans_pp_init)`)
    """
    if n_init is not None:
        best_centroids  = ops.empty((k, ops.shape(points)[1]), dtype = 'float32')
        best_assignment = ops.empty((ops.shape(points)[0], ), dtype = 'int32')
        best_score      = ops.convert_to_tensor(-1., dtype = 'float32')
        for run in range(n_init):
            centroids, assignment = _kmeans(
                points,
                k,
                n_init  = None,
                init_method = init_method,
                max_iter    = max_iter,
                threshold   = threshold,
                distance_metric = distance_metric,
                ** kwargs
            )
            
            score = compute_score(
                points, assignment, centroids, ops.arange(k, dtype = 'int32'),
                distance_metric = distance_metric, normalize = normalize
            )

            if score < best_score or best_score == -1.:
                best_score, best_centroids, best_assignment = score, centroids, assignment
        
        return best_centroids, best_assignment
    
    if init_method == 'normal':
        centroids = keras.random.normal((k, ops.shape(points)[1]), seed = random_state)
    elif init_method == 'uniform':
        centroids = keras.random.uniform(
            shape   = (k, ops.shape(points)[1]),
            minval  = ops.min(points),
            maxval  = ops.max(points),
            seed    = random_state
        )
    elif init_method == 'random':
        indexes = keras.random.shuffle(ops.arange(ops.shape(points)[0]))[:k]
        
        centroids = ops.take(points, indexes, axis = 0)
    elif init_method == 'kmeans_pp':
        centroids = kmeans_pp_init(
            points, k, distance_metric = distance_metric, random_state = random_state, ** kwargs
        )
    else:
        raise ValueError("Initialization mode unknown : {}".format(init_method))
        
    def _kmeans_iter(i, centroids, assignment):
        new_centroids = _update_centroids(points, assignment, centroids, k)
        
        if ops.sum(ops.abs(new_centroids - centroids)) <= threshold:
            return max_iter, new_centroids, assignment
        
        return i + 1, new_centroids, get_assignment(
            points, new_centroids, distance_metric = distance_metric
        )

    assignment = get_assignment(points, centroids, distance_metric = distance_metric)
    
    return ops.while_loop(
        lambda i, c, a: i < max_iter,
        _kmeans_iter,
        (0, centroids, assignment),
        maximum_iterations = max_iter
    )[1:]

@graph_compile(reduce_retracing = True)
def kmeans_pp_init(points   : keras.KerasTensor(shape = (None, None), dtype = 'float32'),
                   k        : keras.KerasTensor(shape = (), dtype = 'int32'),
                   start    : keras.KerasTensor(shape = (), dtype = 'int32')    = None,
                   distance_metric  = 'euclidian',
                   random_state     = None
                  ):
    """
        Initializes the centroids with the `KMeans++` procedure :
            1) Selects a random point as first centroid
            2) Computes the distance between each point and each selected centroid
            3) Adds as new centroid, the point with the highest distance from each centroid
            4) Returns at step 2, until `k` centroids have been selected
    """
    if start is None:
        start = keras.random.randint((), minval = 0, maxval = len(points))
    
    centroids   = ops.empty((k, ops.shape(points)[1]), dtype = 'float32')
    centroids   = ops.slice_update(
        centroids,
        ops.array([0, 0], 'int32'),
        ops.expand_dims(ops.take(points, start, axis = 0), axis = 0)
    )
    for i in range(1, k):
        dist = ops.sum(distance(
            points, centroids[:i], distance_metric, as_matrix = True, force_distance = True
        ), axis = -1)
        centroids = ops.slice_update(
            centroids,
            ops.array([1, 0], 'int32') * i,
            ops.expand_dims(ops.take(points, ops.argmax(dist, axis = -1), axis = 0), axis = 0)
        )
    
    return centroids

def _update_centroids(points, assignments, centroids, k):
    mask    = ops.arange(k)[:, None] == assignments[None, :]
    count   = ops.count_nonzero(mask, axis = -1)[:, None]
    return ops.where(
        count == 0, centroids, ops.divide_no_nan(
            ops.sum(points[None, :, :] * ops.cast(mask, points.dtype)[:, :, None], axis = 1),
            ops.cast(count, points.dtype)
        )
    )

kmeans = clustering_wrapper(_kmeans)