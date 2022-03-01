
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

from kneed import KneeLocator

from utils.distance.clustering import Clustering, get_assignment, compute_score

class KMeans(Clustering):
    def _build_clusters(self, points, * args, ** kwargs):
        centroids, assignment = kmeans(points, * args, ** kwargs)
        
        return assignment
    
    @property
    def k(self):
        return self.n_cluster

def kmeans(points, k, max_iter = 100, init_method = 'normal', method = 'euclidian',
           threshold = 1e-6, n_init = 5, seed = None, plot = False):
    if not isinstance(points, tf.Tensor): points = tf.cast(points, tf.float32)
    if isinstance(k, (list, tuple)):
        scores = {}
        for ki in k:
            centroids, assignment = kmeans(
                points, ki, max_iter = max_iter, n_init = n_init, init_method = init_method, method = method,
                threshold = threshold, seed = None if not seed else seed * run
            )
            score = compute_score(
                points, assignment, centroids, tf.range(ki, dtype = tf.int32), method = method
            )
            scores[ki] = {'score' : score, 'centroids' : centroids, 'assignment' : assignment}
        
        list_scores = [scores[ki]['score'] for ki in k]
        convex = all([list_scores[i] <= list_scores[i-1] for i in range(1, len(list_scores))])
        if convex:
            kl = KneeLocator(
                k, list_scores, curve = "convex", direction = "decreasing"
            )
            
            if plot:
                kl.plot_knee_normalized()
            
            best_k = kl.elbow if kl.elbow is not None else k[np.argmin(list_scores)]
        else:
            best_k = k[np.argmin(list_scores)]

        return scores[best_k]['centroids'], scores[best_k]['assignment']
    
    if n_init > 0:
        best_score, best_centroids, best_assignment = tf.cast(float('inf'), tf.float32), None, None
        for run in range(n_init):
            centroids, assignment = kmeans(
                points, k, max_iter = max_iter, n_init = -1, init_method = init_method, method = method,
                threshold = threshold, seed = None if not seed else seed * run
            )
            score = compute_score(
                points, assignment, centroids, tf.range(k, dtype = tf.int32), method = method
            )
            if score < best_score:
                best_score, best_centroids, best_assignment = score, centroids, assignment
        
        return best_centroids, best_assignment
    
    if init_method == 'normal':
        centroids = tf.random.normal(
            (k, tf.shape(points)[1]), seed = seed
        )
    elif init_method == 'random':
        indexes = tf.random.shuffle(tf.range(tf.shape(points)[0]))[:k]
        
        centroids = tf.gather(points, indexes)
    elif init_method == 'kmeans_pp':
        centroids = kmeans_pp_init(points, k, method = method, seed = seed)
    else:
        raise ValueError("Initialization mode unknown : {}".format(self.init_method))
    
    assignment = tf.range(tf.shape(points)[0], dtype = tf.int32)
    for i in range(max_iter):
        assignment      = get_assignment(points, centroids, method = method)
        
        new_centroids = []
        for i in tf.range(k, dtype = tf.int32):
            cluster_i   = tf.gather(points, tf.cast(tf.reshape(tf.where(assignment == i), [-1]), tf.int32))
            centroid_i  = tf.reduce_mean(cluster_i, axis = 0) if len(cluster_i) > 0 else centroids[i]
            
            new_centroids.append(centroid_i)

        new_centroids = tf.stack(new_centroids, 0)
        
        diff = tf.reduce_sum(tf.abs(new_centroids - centroids))
        if tf.reduce_sum(tf.abs(new_centroids - centroids)) < threshold:
            break
        
        centroids = new_centroids
    
    return centroids, assignment

def kmeans_pp_init(points, k, method = 'euclidian', seed = None):
    n = tf.shape(points)[0]
    centroids = tf.expand_dims(
        points[tf.random.uniform((), minval = 0, maxval = n, dtype = tf.int32, seed = seed)], axis = 0
    )
    
    for i in range(1, k):
        dist = tf.reduce_min(
            distance(points, centroids, method = method, as_matrix = True), axis = -1
        )
        p = dist / tf.reduce_sum(dist)
        
        new_centroid = tf.expand_dims(points[np.random.choice(np.arange(0, n), p = p.numpy())], axis = 0)
        centroids = tf.concat([centroids, new_centroid], axis = 0)
    
    return centroids
