
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

from utils.embeddings import compute_centroids
from utils.distance.distance_method import tf_distance

def find_clusters(* args, method = 'kmeans', ** kwargs):
    if method not in _clustering_methods:
        raise ValueError('Unknown clustering method !\n  Supported : {}\n  Got : {}'.format(
            tuple(_clustering_methods.keys()), method
        ))
    return _clustering_methods[method](* args, ** kwargs)

def clustering_wrapper(clustering_fn):
    def wrapper(points, k, distance_metric = 'euclidian', normalize = False, ** kwargs):
        points = tf.cast(points, tf.float32)
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
                
                score = compute_score(
                    points, assignment, centroids, tf.range(tf.shape(centroids)[0], dtype = tf.int32),
                    distance_metric = distance_metric, normalize = normalize
                ).numpy()
                scores[ki] = {'score' : score, 'centroids' : centroids, 'assignment' : assignment}

            assert len(scores) > 0, 'k must be a range with at least 1 value > 1'

            valids_k    = [ki for ki in k if ki > 1]
            list_scores = np.array([scores[ki]['score'] for ki in valids_k])
            convex      = all([list_scores[i] <= list_scores[i-1] for i in range(1, len(list_scores))])
            if convex:
                kl = KneeLocator(
                    valids_k, list_scores, curve = "convex", direction = "decreasing"
                )

                best_k = kl.elbow if kl.elbow is not None else valids_k[np.argmin(list_scores)]
            else:
                best_k = valids_k[np.argmin(list_scores)]

            return scores[best_k]['centroids'], scores[best_k]['assignment']
        
        return clustering_fn(points, k, distance_metric = distance_metric, ** kwargs)
    
    global _clustering_methods
    
    fn = wrapper
    fn.__name__ = clustering_fn.__name__
    fn.__doc__  = clustering_fn.__doc__

    _clustering_methods[clustering_fn.__name__.lstrip('_')] = fn
    return fn

def evaluate_clustering(y_true, y_pred):
    y_true  = tf.cast(y_true, tf.int32)
    y_pred  = tf.cast(y_pred, tf.int32)
    
    uniques = tf.unique(y_true)[0]
    
    all_f1 = np.zeros((len(uniques), ))
    for i, cluster_id in enumerate(uniques):
        mask    = y_true == cluster_id
        
        pred_ids    = tf.boolean_mask(y_pred, mask)
        pred_ids    = tf.boolean_mask(pred_ids, pred_ids >= 0)
        
        if tf.shape(pred_ids)[0] == 0: continue
        
        counts  = tf.math.bincount(pred_ids)
        main_id = tf.argmax(counts, output_type = tf.int32)
        
        true_positive   = tf.cast(tf.reduce_max(counts), tf.float32)
        total_pred      = tf.reduce_sum(tf.cast(y_pred == main_id, tf.float32))
        total_true      = tf.reduce_sum(tf.cast(mask, tf.float32))
        
        precision = true_positive / total_pred
        recall    = true_positive / total_true
        f1        = (2 * precision * recall) / (precision + recall)

        y_pred    = tf.where(y_pred == main_id, -1, y_pred)

        all_f1[i] = f1
    
    return np.mean(all_f1), all_f1

@tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
def get_assignment(points : tf.Tensor, centroids : tf.Tensor, distance_metric = 'euclidian'):
    """ Returns a vector of ids, the nearest centroid's index (according to `distance_metric`) """
    return tf.argmin(tf_distance(
        points, centroids, distance_metric, as_matrix = True, force_distance = True
    ), axis = -1, output_type = tf.int32)

@tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
def compute_score(points    : tf.Tensor,
                  ids       : tf.Tensor,
                  centroids : tf.Tensor,
                  centroid_ids  : tf.Tensor,
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
    mask    = tf.cast(
        tf.expand_dims(ids, axis = 1) == centroid_ids, tf.float32
    )
    dist    = tf_distance(
        points, centroids, distance_metric, as_matrix = True, force_distance = True
    )
    if normalize:
        return tf.reduce_sum(tf.math.divide_no_nan(
            tf.reduce_sum(dist * mask, axis = -1), tf.reduce_sum(mask, axis = -1)
        ))
    return tf.reduce_sum(dist * mask)

_clustering_methods = {}