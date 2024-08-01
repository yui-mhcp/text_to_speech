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

from utils.keras_utils import ops
from utils.distance.knn_method import knn
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
    if distance_matrix is None:
        distance_matrix = tf_distance(
            points, points, distance_metric, as_matrix = True, mode = 'distance', ** kwargs
        )
    distance_matrix = distance_matrix / ops.max(distance_matrix)
    
    nearest_dist, nearest_idx = ops.top_k(-distance_matrix, k_nearest + 1)

    assignment = ids if ids is not None else np.concatenate([
        np.zeros((1, ), dtype = np.int32),
        np.full((len(points) - 1, ), -1)
    ], axis = 0)
    assignment = ops.convert_to_numpy(assignment)
    np_distance_matrix = ops.convert_to_numpy(distance_matrix)

    nb_unique = len(np.unique(assignment)) - 1
    
    max_dist    = - nearest_dist[:, -1]
    mean_dist   = - ops.mean(nearest_dist[:, 1:], axis = -1)
    threshold   = ops.maximum(mean_dist + (max_dist - mean_dist) * tolerance, min_threshold)
    threshold   = ops.convert_to_numpy(threshold)
    
    visited = np.zeros((len(assignment), ), dtype = bool)
    
    s = 0
    
    to_visit = []
    while not np.all(visited):
        if len(to_visit) == 0:
            remaining = np.where(~visited)[0]
            if np.any(assignment[remaining] != -1):
                to_visit.append(np.where(np.logical_and(visited == False, assignment != -1))[0][0])
            else:
                to_visit.append(remaining[0])
        
        idx = to_visit.pop(0)
        if visited[idx]: continue
        
        visited[idx] = True
        if assignment[idx] == -1:
            assignment[idx] = ops.convert_to_numpy(knn(
                distance_matrix = ops.expand_dims(ops.take(distance_matrix, idx, axis = 0), 0),
                ids = assignment + 1, k = k_nearest + 1
            ) - 1)
            if assignment[idx] < 0:
                assignment[idx] = nb_unique
                nb_unique += 1
                s = 0
        
        nearest = np.where(
            np.logical_and(np_distance_matrix[idx] <= threshold[idx], assignment == -1)
        )[0]
        
        assignment[nearest] = assignment[idx]
        if s % 100 == 0 and plot:
            from utils import plot_embedding
            plot_embedding(
                points, assignment, c = ['w', 'r', 'g', 'cyan', 'b', 'black', 'yellow', 'violet', 'orange']
            )
        s += 1
        to_visit.extend(nearest)
    
    return assignment

label_propagation = clustering_wrapper(_label_propagation)