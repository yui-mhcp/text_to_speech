
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

from utils.distance.knn import knn
from utils.distance.distance_method import tf_distance
from utils.distance.clustering import clustering_wrapper

def _label_propagation(points,
                       k,
                       
                       k_nearest    = 11,
                       tolerance    = 0.5,
                       min_threshold    = 0.05,
                       
                       ids  = None,
                       distance_matrix    = None,
                       distance_metric      = 'euclidian',
                       plot = False,
                       ** kwargs
                      ):
    from utils import plot_embedding
    
    if distance_matrix is None:
        distance_matrix = tf_distance(
            points, points, distance_metric, as_matrix = True, force_distance = True
        )
    distance_matrix = distance_matrix / tf.reduce_max(distance_matrix)
    
    assignment = ids if ids is not None else tf.concat([
        tf.zeros((1, ), dtype = tf.int32),
        tf.fill((tf.shape(points)[0] - 1, ), -1)
    ], axis = 0)
    assignment = assignment.numpy()
    
    nb_unique = tf.size(tf.unique(assignment)[0]) - 1
    
    nearest_dist, nearest_idx = tf.nn.top_k(-distance_matrix, k_nearest + 1)
    max_dist    = - nearest_dist[:, -1]
    mean_dist   = - tf.reduce_mean(nearest_dist[:, 1:], axis = -1)
    threshold   = tf.maximum(mean_dist + (max_dist - mean_dist) * tolerance, min_threshold)
    
    visited = np.full((len(assignment), ), False)
    
    s = 0
    
    to_visit = []
    while not np.all(visited):
        if len(to_visit) == 0:
            remaining = np.where(visited == False)[0]
            if np.any(assignment[remaining] != -1):
                to_visit.append(np.where(np.logical_and(visited == False, assignment != -1))[0][0])
            else:
                to_visit.append(remaining[0])
        
        idx = to_visit.pop(0)
        if visited[idx]: continue
        
        visited[idx] = True
        if assignment[idx] == -1:
            assignment[idx] = knn(
                distance_matrix = tf.expand_dims(distance_matrix[idx], 0),
                ids = assignment + 1, k = k_nearest + 1
            ) - 1
            if assignment[idx] < 0:
                assignment[idx] = nb_unique
                nb_unique += 1
                s = 0
        
        nearest = np.where(
            np.logical_and(distance_matrix[idx] <= threshold[idx], assignment == -1)
        )[0]
        
        assignment[nearest] = assignment[idx]
        if s % 100 == 0 and plot:
            plot_embedding(
                points, assignment, c = ['w', 'r', 'g', 'cyan', 'b', 'black', 'yellow', 'violet', 'orange']
            )
        s += 1
        to_visit.extend(nearest)
    
    return assignment

label_propagation = clustering_wrapper(_label_propagation)