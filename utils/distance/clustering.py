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
import numpy as np

from functools import wraps

from .distance_method import distance
from utils import get_fn_name, compute_centroids, dispatch_wrapper, add_doc
from utils.keras_utils import TensorSpec, ops, graph_compile

_clustering_methods = {}

@dispatch_wrapper(_clustering_methods, 'method', default = 'kmeans')
def find_clusters(* args, method = 'kmeans', ** kwargs):
    """ Computes the centroid with the clustering `method` """
    if method not in _clustering_methods:
        raise ValueError('Unknown clustering method !\n  Supported : {}\n  Got : {}'.format(
            tuple(_clustering_methods.keys()), method
        ))
    
    return _clustering_methods[method](* args, ** kwargs)

def clustering_wrapper(clustering_fn):
    @wraps(clustering_fn)
    @add_doc(clustering_fn, on_top = False)
    def wrapper(points, k, distance_metric = 'euclidian', normalize = False, ** kwargs):
        """
            Arguments :
                - points    : `Tensor` with shape `(n_embeddings, embedding_dim)`
                - k     : the number of centroids to compute
                    - int   : exact number of centroids
                    - list / range  : computes the assignment for each `k`, then returns the result for the `k` selected by `kneed.KNeeLocator`
                - distance_metric   : the distance / similarity metric to use to compute distance
                - normalize : whether to normalize the scores or not
                - kwargs    : forwarded to the original clustering function
            Return : `(centroids, assignment)`
                - centroids     : `Tensor` with shape `(k, embedding_dim)`
                - assignment    : `Tensor` with shape `(n_embedding, )`, the id for each point
        """
        points = ops.convert_to_tensor(points, 'float32')
        
        if isinstance(k, (list, tuple, range)):
            from kneed import KneeLocator

            scores = {}
            for ki in k:
                if ki <= 1: continue

                clusters = clustering_fn(
                    points, ki, distance_metric = distance_metric, normalize = normalize, ** kwargs
                )
                if isinstance(clusters, tuple):
                    centroids, assignment = clusters
                else:
                    centroids, assignment = compute_centroids(points, clusters)[1], clusters
                
                score = ops.convert_to_numpy(compute_score(
                    points,
                    assignment,
                    centroids,
                    ops.arange(len(centroids), dtype = 'int32'),
                    distance_metric = distance_metric,
                    normalize = normalize
                ))
                scores[ki] = {'score' : score, 'centroids' : centroids, 'assignment' : assignment}

            assert len(scores) > 0, 'k must be a range with at least 1 value > 1'

            valids_k    = sorted(scores.keys())
            list_scores = np.array([scores[ki]['score'] for ki in valids_k])
            convex      = all(
                list_scores[i] <= list_scores[i - 1] for i in range(1, len(list_scores))
            )
            if convex:
                kl = KneeLocator(
                    valids_k, list_scores, curve = "convex", direction = "decreasing"
                )

                best_k = kl.elbow if kl.elbow is not None else valids_k[np.argmin(list_scores)]
            else:
                best_k = valids_k[np.argmin(list_scores)]

            return scores[best_k]['centroids'], scores[best_k]['assignment']
        
        clusters = clustering_fn(
            points, k, distance_metric = distance_metric, normalize = normalize, ** kwargs
        )
        if isinstance(clusters, tuple):
            centroids, assignment = clusters
        else:
            centroids, assignment = compute_centroids(points, clusters)[1], clusters
        
        return centroids, assignment

    
    find_clusters.dispatch(wrapper, get_fn_name(clustering_fn).lstrip('_'))

    return wrapper

def evaluate_clustering(y_true, y_pred):
    y_true  = ops.convert_to_tensor(y_true, 'int32')
    y_pred  = ops.convert_to_tensor(y_pred, 'int32')
    
    uniques = ops.unique(y_true, return_inverse = False, return_counts = False)
    
    all_f1 = np.zeros((len(uniques), ))
    for i, cluster_id in enumerate(uniques):
        mask    = y_true == cluster_id
        
        pred_ids    = y_pred[mask]
        pred_ids    = pred_ids[pred_ids >= 0]
        
        if len(pred_ids) == 0: continue
        
        counts  = ops.cast(ops.bincount(pred_ids), 'int32')
        main_id = ops.argmax(counts)
        
        true_positive   = ops.cast(ops.max(counts), 'float32')
        total_pred      = ops.cast(ops.count_nonzero(y_pred == main_id), 'float32')
        total_true      = ops.cast(ops.count_nonzero(mask), 'float32')
        
        precision = true_positive / total_pred
        recall    = true_positive / total_true
        f1        = (2 * precision * recall) / (precision + recall)

        y_pred    = ops.where(y_pred == main_id, -1, y_pred)

        all_f1[i] = f1
    
    return np.mean(all_f1), all_f1

@graph_compile(reduce_retracing = True)
def get_assignment(points   : keras.KerasTensor(shape = (None, None), dtype = 'float32'),
                   centroids    : keras.KerasTensor(shape = (None, None), dtype = 'float32'),
                   distance_metric  = 'euclidian',
                   ** kwargs
                  ):
    """ Returns a vector of ids, the nearest centroid's index (according to `distance_metric`) """
    return ops.argmin(distance(
        points, centroids, distance_metric, as_matrix = True, force_distance = True, ** kwargs
    ), axis = -1)

@graph_compile(reduce_retracing = True)
def compute_score(points    : TensorSpec(shape = (None, None), dtype = 'float32'),
                  ids       : TensorSpec(shape = (None, )),
                  centroids : TensorSpec(shape = (None, None), dtype = 'float32'),
                  centroid_ids  : TensorSpec(shape = (None, )),
                  distance_metric = 'euclidian',
                  normalize = False
                 ):
    """
        Computes a *clustering score* based on an assignment and a set of centroids
        
        Arguments :
            - points    : 2-D matrix of shape [n_embeddings, embedding_dim]
            - ids       : the points' assignment (1-D vector of length [n_embeddings])
            - centroids : 2-D matrix of shape [n_clusters, embedding_dim]
            - centroid_ids  : 1-D vector, the centroids' ids
            - distance_metric   : the distance metric to compute the score
        Returns :
            - score : the scalar value representing the total distance between all points and their associated centroid
    """
    mask    = ops.cast(ids[:, np.newaxis] == centroid_ids[np.newaxis], 'float32')
    dist    = distance(points, centroids, distance_metric, as_matrix = True, force_distance = True)
    if normalize:
        return ops.sum(ops.divide_no_nan(
            ops.sum(dist * mask, axis = -1), ops.sum(mask, axis = -1)
        ))
    return ops.sum(dist * mask)

