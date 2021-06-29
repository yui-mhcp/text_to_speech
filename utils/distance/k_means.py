import numpy as np
import tensorflow as tf

from kneed import KneeLocator

from utils.plot_utils import plot
from utils.distance.knn import KNN
from utils.distance.clustering import Clustering 


class KMeans(Clustering):
    def __init__(self,
                 points,
                 k      = -1,
                 min_k  = 2,
                 max_k  = 10,
                 n_init     = 10,
                 max_iter   = 250,
                 threshold  = 1e-6,
                 random_state   = 10,
                 ** kwargs
                ):
        self.k  = k
        self.min_k  = min_k
        self.max_k  = max_k
        self.n_init = n_init
        self.max_iter   = max_iter
        self.threshold  = threshold
        self.random_state   = random_state
        
        self.centroids  = None
        self._score     = None
        
        super().__init__(points, ** kwargs)
    
    def _build_clusters(self, points, ** kwargs):
        if self.k == -1:
            best, _ = KMeans.fit_best_k(points, self.min_k, self.max_k, ** kwargs)
            self.k, self._score  = best.k, best.score
            return best.labels
        
        points = self.normalize(points)
        clusters = []
        for i in range(self.n_init):
            centroids = tf.random.normal(
                (self.k, tf.shape(points)[1]), seed = self.random_state + i
            )
            
            run     = 0
            changed = True
            while changed and run < self.max_iter:
                new_centroids, cluster = self.update_centroids(points, centroids)
                
                if tf.reduce_sum(tf.cast(tf.math.is_nan(new_centroids), tf.float32)) > 0:
                    centroids = tf.random.normal(
                        (self.k, tf.shape(points)[1]), seed = self.random_state
                    )
                    continue
                
                diff = tf.reduce_sum(tf.abs(new_centroids - centroids))
                changed = diff > self.threshold
                centroids = new_centroids
                run += 1
            
            cluster = self.clean_clusters(
                points, cluster, k = self.k, method = self.distance_metric
            )
            centroids, _ = self.build_centroids(points, cluster)
            
            clusters.append({
                'clusters'  : cluster,
                'centroids' : centroids,
                'total_distance' : self.compute_score(points, centroids, cluster)
            })
        best_cluster = sorted(clusters, key = lambda c: c['total_distance'])[0]
        
        self.centroids  = best_cluster['centroids']
        self._score     = best_cluster['total_distance']
        
        return best_cluster['clusters']

    @property
    def score(self):
        return self._score
    
    def update_centroids(self, points, centroids):
        distances = self.distance(tf.expand_dims(points, axis = 1), centroids)
        
        cluster = tf.argmin(distances, axis = -1)
        
        return self.build_centroids(points, cluster, k = self.k)[0], cluster
    
        
    def compute_score(self, points, centroids, cluster):
        return tf.reduce_sum([
            tf.reduce_sum(self.distance(
                tf.gather(points, tf.where(cluster == i)), centroids[i]
            )) for i in range(tf.shape(centroids)[0])
        ])
    
    @classmethod
    def fit_best_k(cls, points, min_k, max_k, debug = False, ** kwargs):
        kmeans = [
            cls(points, k = k, ** kwargs) for k in range(min_k, max_k)
        ]
        scores = [kmean.score for kmean in kmeans]
        
        convex = all([scores[i] <= scores[i-1] for i in range(1, len(scores))])
        if convex:
            kl = KneeLocator(
                range(min_k, max_k), scores, curve = "convex", direction = "decreasing"
            )
            
            if debug:
                kl.plot_knee_normalized()
            
            best_k = kl.elbow if kl.elbow is not None else np.argmin(scores) + min_k
        else:
            best_k = np.argmin(scores) + min_k
        
        return kmeans[best_k - min_k], kmeans            
    
