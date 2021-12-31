import numpy as np
import pandas as pd
import tensorflow as tf

from utils.plot_utils import plot_embedding
from utils.embeddings import load_embedding, compute_mean_embeddings, embeddings_to_np
from utils.distance.distance_method import distance

class KNN(object):
    """
        Tensorflow implementation of the `K-Nearest Neighbors` algorithm
        
        It also has some additional features such as : 
            - Plotting embeddings / predictions
            - Use a `use_mean` version where the prediction is the nearest `centroid`*
            
        * A `centroid` is the mean point of all points belonging to a given label
    """
    def __init__(self, embeddings, ids = None, k = 5, use_mean = False, 
                 method = 'euclidian', ** kwargs):
        """
            Constructor for the KNN class
            
            Arguments : 
                - embeddings    : the embeddings to use as labelled points
                    If str  : call `load_embedding()` on it
                    If pd.DataFrame : use the 'id' column for `ids` and call `embeddings_to_np` on it
                    Else    : must be a np.ndarray / tf.Tensor 2D matrix
                - ids   : ids of the embeddings (if embeddings is a DataFrame, ids = embeddings['id'].values)
                - k / use_mean  : default configuration for the `predict` method
                - method        : distance method to use
        """
        if isinstance(embeddings, str):
            embeddings = load_embedding(embeddings)
        
        if isinstance(embeddings, pd.DataFrame):
            ids = embeddings['id'].values
            embeddings = embeddings_to_np(embeddings)
        
        assert ids is None or len(embeddings) == len(ids)
        
        self.ids        = np.array(ids)
        self.embeddings = tf.cast(embeddings, tf.float32)
        
        self.k          = tf.cast(k, dtype = tf.int32)
        self.use_mean   = use_mean
        self.method     = method
        
        self.__mean_ids         = None
        self.__mean_embeddings  = None
    
    @property
    def mean_ids(self):
        if self.__mean_ids is None:
            self.__mean_ids, self.__mean_embeddings = self.get_mean_embeddings()
        return self.__mean_ids
    
    @property
    def mean_embeddings(self):
        if self.__mean_embeddings is None:
            self.__mean_ids, self.__mean_embeddings = self.get_mean_embeddings()
        return self.__mean_embeddings
    
    def __setitem__(self, idx, val):
        if val != self.ids[idx]:
            self.ids[idx] = val
            self.__mean_ids, self.__mean_embeddings = None, None
    
    def get_mean_embeddings(self):
        """ Compute the mean embeddings for each id """
        return compute_mean_embeddings(self.embeddings, self.ids)
    
    def get_embeddings(self, ids = None, use_mean = False):
        """ Return all embeddings from specified ids """
        if ids is not None and not isinstance(ids, (list, tuple, np.ndarray, tf.Tensor)): ids = [ids]
        if use_mean:
            embeddings, res_ids = self.mean_embeddings, self.mean_ids
        else:
            embeddings, res_ids = self.embeddings, self.ids
        
        if ids is not None:
            indexes = tf.reshape(tf.concat([
                tf.where(res_ids == id_i) for id_i in ids
            ], axis = 0), [-1])

            embeddings  = tf.gather(embeddings, indexes)
            res_ids     = tf.gather(res_ids, indexes)

        return embeddings, res_ids
    
    def distance(self, x, ** kwargs):
        """ Compute distance between x and embeddings for given ids """
        embeddings, ids = self.get_embeddings(** kwargs)
        return distance(tf.cast(x, tf.float32), embeddings, method = self.method), ids
    
    def predict(self, query, possible_ids = None, k = None, use_mean = None,
                plot = False, tqdm = lambda x: x, ** kwargs):
        """
            Predict ids for each `x` vector based on the `k-nn` decision procedure
            
            Arguments :
                - x : the 1D / 2D matrix of embeddings vector(s) to predict label
                - possible_ids  : a list of `possible ids` (other ids are not taken into account for the k-nn)
                - k / use_mean  : k-nn metaparameter (if not provided use self.k / self.use_mean)
                - tqdm  : progress bar if `x` is a matrix
                - plot / kwargs : whether to plot the prediction result or not
            
            If x is a matrix, call `self.predict` for each vector in the matrix in a multi-threaded way
            It allows to achieve really good performances even for prediction on a large dataset
        """
        if use_mean is None: use_mean = self.use_mean
        if use_mean: k = 1
        elif k is None: k = self.k
        else: k = tf.cast(k, tf.int32)
        
        if possible_ids is not None and not isinstance(possible_ids, (list, tuple, np.ndarray, tf.Tensor)):
            possible_ids = [possible_ids]
        
        query = tf.cast(query, tf.float32)

        embeddings, ids = self.get_embeddings(possible_ids, use_mean)
        
        pred = knn(query, embeddings, ids, k, self.method)
        
        if plot:
            self.plot(query, pred, ** kwargs)
        
        return pred
        
    def plot(self, x = None, x_ids = None, marker_kwargs = None, ** kwargs):
        """
            Plot the labelled datasets + centroids + possible `x` to predict (with their predicted labels) 
        """
        if marker_kwargs is None: marker_kwargs = {}

        # Original points
        embeddings, ids = self.embeddings, self.ids
        marker = ['o'] * len(embeddings)
        
        # Means as big points
        embeddings = np.concatenate([embeddings, self.mean_embeddings], axis = 0)
        ids = np.concatenate([ids, self.mean_ids], axis = 0)
        
        marker += ['O'] * len(self.mean_ids)
        marker_kwargs.setdefault('O', {
            'marker'    : 'o',
            'linewidth' : kwargs.get('linewidth', 2.5) * 3
        })
        
        # New data points to plot
        if x is not None:
            if isinstance(x, pd.DataFrame):
                if 'id' in x and x_ids is None:
                    x_ids = x['id'].values
                x = embeddings_to_np(x)
            
            if len(x.shape) == 1: x = np.expand_dims(x, 0)
            if x_ids is not None:
                if not isinstance(x_ids, (list, tuple, np.ndarray, tf.Tensor)): x_ids = [x_ids]
                x_ids = np.reshape(x_ids, [-1])
            else:
                fake_id = 0
                while fake_id in ids: fake_id += 1
                x_ids = np.array([fake_id] * len(x))
                marker_kwargs.setdefault('x', {'c' : 'w'})
            
            assert len(x_ids) == len(x), "{} vs {}".format(x.shape, x_ids.shape)
            
            embeddings = np.concatenate([embeddings, x], axis = 0)
            ids = np.concatenate([ids, x_ids], axis = 0)
            marker += ['x'] * len(x)
        
        plot_embedding(
            embeddings, ids = ids, marker = np.array(marker), 
            marker_kwargs = marker_kwargs, ** kwargs
        )

def knn(query, embeddings, ids, k, distance_metric, return_index = False, ** kwargs):
    """
        Compute the k-nn decision procedure for a given x based on a list of labelled embeddings
        
        Return the majoritary id in the `k` nearest neigbors or `-2` if there is an equality.
        If no ids are provided (`ids = None`), return either indexes (`return_index = True`) or nearest embeddings.
    """
    distances = distance(query, embeddings, method = distance_metric, as_matrix = True, ** kwargs)

    _, k_nearest_idx = tf.nn.top_k(- distances, tf.minimum(tf.shape(distances)[1], k))
    
    if ids is None:
        if return_index:
            return k_nearest_idx
        return tf.gather(embeddings, k_nearest_idx, batch_dims = 1)
    
    nearest_ids = tf.cast(tf.gather(tf.reshape(ids, [-1]), k_nearest_idx), tf.int32)

    counts = tf.math.bincount(nearest_ids, axis = -1)

    max_counts = tf.reduce_max(counts, axis = -1, keepdims = True)
    
    max_idx = tf.cast(counts == max_counts, tf.int32)
    
    nb_nearest = tf.reduce_sum(max_idx, axis = -1)
    mask = tf.cast(nb_nearest == 1, tf.int32)
    
    nearest_ids = tf.cast(tf.argmax(max_idx, axis = -1), tf.int32) * mask - 2 * (1 - mask)

    return nearest_ids
