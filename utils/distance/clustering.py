
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

import logging
import numpy as np
import tensorflow as tf

from utils.plot_utils import plot_embedding
from utils.distance.knn import KNN
from utils.distance.distance_method import distance

class Clustering:
    def __init__(self,
                 points,
                 min_cluster_size   = 5, 
                 distance_metric    = 'euclidian',
                 ** kwargs
                ):
        self.points     = points
        self.distance_metric    = distance_metric
        self.min_cluster_size   = min_cluster_size if isinstance(min_cluster_size, int) else int(min_cluster_size * len(points))
        
        self._labels    = self._build_clusters(points, ** kwargs)
    
    def _build_clusters(self, points):
        raise NotImplementedError()
    
    @property
    def ids(self):
        return np.unique(self.labels)
    
    @property
    def n_cluster(self):
        return len(self.ids)
    
    @property
    def clusters(self):
        return {cluster_id : self[cluster_id] for cluster_id in self.ids}
    
    @property
    def cluster_size(self):
        return {cluster_id : len(self[cluster_id]) for cluster_id in self.ids}
    
    @property
    def labels(self):
        return self._labels
    
    def __len__(self):
        return self.n_cluster
    
    def __str__(self):
        return "Found {} clusters for {} points".format(self.n_cluster, len(self.points))
    
    def __getitem__(self, idx):
        return self.points[self.labels == self.ids[idx]]
    
    def distance(self, a, b, ** kwargs):
        return distance(a, b, method = self.distance_metric, ** kwargs)
    
    def score(self, centroids = None, ids = None):
        if centroids is None or ids is None:
            centroids, ids = get_centroids(self.points, self.labels)

        return compute_score(self.points, self.labels, centroids, ids, method = self.distance_metric)
    
    def normalize(self, points):
        points = tf.cast(points, tf.float32)
        mean, std = tf.reduce_mean(points, axis = 0), tf.math.reduce_std(points, axis = 0)
        return (points - mean) / std
        
    def clean_clusters(self, points, cluster, tqdm = lambda x: x, ** kwargs):
        knn = KNN(points, cluster, ** kwargs)
        
        ids, _ = tf.unique(cluster)
        for i, cluster_id in enumerate(tqdm(ids)):
            cluster_indexes = tf.reshape(tf.where(cluster == cluster_id), [-1])
            if len(cluster_indexes) >= self.min_cluster_size: continue
            
            logging.debug("Clean cluster {}".format(cluster_id))
            other_ids = [id_i for id_i in ids if id_i != cluster_id]
            for idx in cluster_indexes:
                new_id = tf.squeeze(knn.predict(points[i], possible_ids = other_ids))
                if new_id == -2:
                    new_id = knn.predict(
                        points[i], possible_ids = other_ids, use_mean = True
                    )
                knn[idx] = new_id
            
        cluster = tf.cast(knn.ids, tf.int32)
        
        return cluster
        

    def evaluate(self, y_true):
        return evaluate_clustering(y_true, self.labels)
    
    def plot(self, ** kwargs):
        plot_embedding(self.points, self.labels, ** kwargs)

def evaluate_clustering(y_true, y_pred):
    if isinstance(y_pred, tf.Tensor): y_pred = y_pred.numpy()
    
    accs = []
    for i, cluster_id in enumerate(np.unique(y_true)):
        pred_ids = y_pred[y_true == cluster_id]
        ids = np.bincount(pred_ids[pred_ids != -1])
        if len(ids) == 0:
            accs.append(0.)
            continue

        y_pred[y_pred == np.argmax(ids)] = -1

        accs.append(np.max(ids) / len(pred_ids))
    return np.mean(accs), accs

def get_assignment(points, centroids, method = 'euclidian', ** kwargs):
    return tf.cast(tf.argmin(distance(
        points, centroids, method = method, as_matrix = True, ** kwargs
    ), axis = -1), tf.int32)
    
def get_centroids(points, cluster, k = None):
    """ Returns centroids, ids """
    ids = tf.unique(cluster)[0] if k is None else tf.range(k)
    return tf.concat([
        tf.reduce_mean(tf.gather(points, tf.where(cluster == cluster_id)), axis = 0)
        for cluster_id in ids
    ], axis = 0), ids

def compute_score(points, cluster_id, centroids, centroids_id, method = 'euclidian'):
    return tf.reduce_sum([
        tf.reduce_sum(distance(
            centroids[i],
            tf.gather(points, tf.reshape(tf.where(cluster_id == centroids_id[i]), [-1])),
            method = method
        )) for i in range(tf.shape(centroids)[0])
    ])
