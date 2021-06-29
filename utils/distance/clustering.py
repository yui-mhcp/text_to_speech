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
        self.min_cluster_size   = min_cluster_size if min_cluster_size > 1 else int(min_cluster_size * len(points))
        
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
        return [self[cluster_id] for cluster_id in self.ids]
    
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
    
    def distance(self, a, b):
        return distance(a, b, method = self.distance_metric)
    
    def normalize(self, points):
        points = tf.cast(points, tf.float32)
        mean, std = tf.reduce_mean(points, axis = 0), tf.math.reduce_std(points, axis = 0)
        return (points - mean) / std
        
    def build_centroids(self, points, cluster, k = None):
        ids = tf.unique(cluster)[0] if k is None else range(k)
        return tf.concat([
            tf.reduce_mean(tf.gather(points, tf.where(cluster == cluster_id)), axis = 0)
            for cluster_id in ids
        ], axis = 0), ids

    def clean_clusters(self, points, cluster, tqdm = lambda x: x, ** kwargs):
        knn = KNN(points, cluster, ** kwargs)
        
        ids, _ = tf.unique(cluster)
        for i, cluster_id in enumerate(tqdm(ids)):
            cluster_indexes = tf.reshape(tf.where(cluster == cluster_id), [-1])
            if len(cluster_indexes) >= self.min_cluster_size: continue
            
            print("Clean cluster {}".format(cluster_id))
            other_ids = [id_i for id_i in ids if id_i != cluster_id]
            for idx in cluster_indexes:
                new_id = knn.predict(points[i], possible_ids = other_ids)
                if new_id == -2:
                    new_id = knn.predict(
                        points[i], possible_ids = other_ids, use_mean = True
                    )
                knn[idx] = new_id
            
        cluster = tf.cast(knn.ids, tf.int32)
        
        return cluster
        
    def evaluate(self, y_true):
        y_pred = np.array(self.labels)
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

    def plot(self, ** kwargs):
        plot_embedding(self.points, self.labels, ** kwargs)
    
