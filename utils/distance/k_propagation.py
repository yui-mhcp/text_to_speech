import numpy as np
import tensorflow as tf

from utils.distance.knn import KNN
from utils.distance.clustering import Clustering 


class KPropagation(Clustering):
    def __init__(self,
                 points,
                 similarity_matrix  = None,
                 
                 k  = 5,
                 use_mean   = True,
                 overwrite  = False,
                 
                 threshold  = 0.7,
                 min_threshold  = 0.5,
                 
                 fusion_threshold   = 0.99,
                 should_combine = False,
                 
                 ** kwargs
                ):
        """
            Clustering algorithm that will group `points` based on their similarity score. 
            The similarity score is defined by the `similarity_matrix` which is a (n x n) matrix where the element [i][j] is the probability [0, 1] for point i,j to be similar (the same class / cluster). 
            
            Arguments :
                - points            : [n, emb_size] matrix of points to clusterize. 
                - similarity_matrix : [n, n] matrix qhere element [i][j] is the probability (between [0, 1]) for points i and j to be similar (same cluster). 
                
                - k / use_mean  : K-NN parameters for predicting cluster from possible clusters
                - overwrite : whether to overwrite already given ids (by passing `ids` kwargs)
                
                - threshold     : threshold for the mean similarity to a cluster. A cluster will be marked as "possible" if the new point has a mean similarity score >= `threshold` (with all points from this cluster)
                - min_threshold : a cluster will be marked as "possible" if the minimal similarity score is >= `min_threshold`
                
                - fusion_threshold  : percentage of points from cluster A that must have a mean similarity score >= `threshold` (with all points of cluster B) to be considered as "sub-part" of cluster B and be combined with it.
                - should_combine    : whether to re-apply a K-NN algorithm on each point to reassign them to its best cluster
        """
        self.similarity_matrix  = similarity_matrix
        
        self.k  = k
        self.use_mean   = use_mean
        self.overwrite  = overwrite
        self.threshold  = threshold
        self.min_threshold  = min_threshold
        self.fusion_threshold   = fusion_threshold
        self.should_combine = should_combine
        
        super().__init__(points, ** kwargs)
    
    def _build_similarity_matrix(self):
        points = tf.cast(self.points, tf.float32)
        similarity_matrix = self.distance(tf.expand_dims(points, axis = 1), points)
        similarity_matrix = 1. - similarity_matrix / tf.reduce_max(similarity_matrix)
        self.similarity_matrix = similarity_matrix.numpy()

    def _build_clusters(self, points, ids = None, tqdm = lambda x: x,
                        debug = False, ** kwargs):
        if self.similarity_matrix is None: self._build_similarity_matrix()
        if hasattr(self.similarity_matrix, 'numpy'):
            self.similarity_matrix = self.similarity_matrix.numpy()
        
        if ids is None: ids = np.zeros((len(points),), dtype = np.int32) - 1
        
        if ids[0] == -1: ids[0] = 0
        nb_unique = len(np.unique(ids)) - 1
        
        sorted_indexes = np.flip(np.argsort(self.similarity_matrix[0]))
        for index in tqdm(range(1, len(ids))):
            if ids[index] != -1: continue
            
            similarities = self.similarity_matrix[index]
            
            possible_ids, means = [], {}
            for id_i in np.unique(ids):
                if id_i == -1: continue
                
                simi = similarities[np.where(ids == id_i)]
                
                means[id_i] = np.mean(simi)
                if np.mean(simi) >= self.threshold and np.min(simi) >= self.min_threshold:
                    possible_ids.append(id_i)

            if debug:
                print("Mean similarity / id (max = {}) : {}".format(
                    np.max(list(means.values())), means
                ))
            
            if len(possible_ids) == 0:
                if debug:
                    print("New id at idx {} ({})".format(index, nb_unique))
                    
                pred_id = nb_unique
                nb_unique += 1
            elif len(possible_ids) == 1:
                pred_id = possible_ids[0]
            else:
                possible_ids = np.array(possible_ids)

                knn = KNN(points, ids, k = self.k, use_mean = self.use_mean)

                pred_id = knn.predict(
                    points[index], possible_ids = possible_ids, ** kwargs
                )
            
            ids[index] = pred_id
        
        ids = self.englobe(points, ids, debug = debug, tqdm = tqdm)

        if self.should_combine:
            ids = self.combine(points, ids, debug = debug, tqdm = tqdm)
        
        return self.clean_clusters(points, ids, k = self.k, tqdm = tqdm)

    def combine(self, points, cluster, debug = False, tqdm = lambda x: x):
        knn = KNN(points, cluster, k = self.k + 1, use_mean = self.use_mean)
        
        for i, point in enumerate(tqdm(points)):
            new_id = knn.predict(point)
            if new_id != cluster[i] and new_id != -2:
                if debug: print("Clusterr {} get {}".format(new_id, i))
                knn[i] = new_id
        return knn.ids
        
    def englobe(self, points, cluster, debug = False, tqdm = lambda x: x):
        ids = np.unique(cluster)
        indexes = [np.where(cluster == cluster_id)[0] for cluster_id in ids]
        
        ids = ids[np.flip(np.argsort([len(idx) for idx in indexes]))]

        for i, cluster_i in enumerate(ids):
            indexes_i = np.where(cluster == cluster_i)[0]
            if len(indexes_i) == 0: continue
            
            for j, cluster_j in enumerate(ids):
                if cluster_i == cluster_j: continue
                indexes_j = np.where(cluster == cluster_j)[0]
                if len(indexes_j) == 0: continue
                
                mean_similarity = np.mean([
                    np.mean(self.similarity_matrix[idx][indexes_i]) > self.threshold
                    for idx in indexes_j
                ])
                
                min_similarity = self.fusion_threshold if len(indexes_j) > 10 else 0.66
                if mean_similarity >= min_similarity:
                    if debug: print("{} englobes {} !".format(cluster_i, cluster_j))
                    cluster[indexes_j] = cluster_i
                    
        return cluster
        
