import numpy as np
import pandas as pd
import tensorflow as tf

from utils.thread_utils import ThreadPool
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
        
        assert len(embeddings) == len(ids)
        
        self.ids    = np.array(ids)
        self.embeddings = tf.cast(embeddings, tf.float32)
        
        self.k          = tf.cast(k, dtype = tf.int32)
        self.use_mean   = use_mean
        self.method     = method
        
        self._mean_ids   = None
        self._mean_embeddings    = None
        
    @property
    def mean_ids(self):
        if self._mean_ids is None:
            self._mean_ids, self._mean_embeddings = self.get_mean_embeddings()
        return self._mean_ids
    
    @property
    def mean_embeddings(self):
        if self._mean_embeddings is None:
            self._mean_ids, self._mean_embeddings = self.get_mean_embeddings()
        return self._mean_embeddings
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.ids[idx]
    
    def __setitem__(self, idx, value):
        self._mean_ids, self._mean_embeddings   = None, None
        self.ids[idx] = value
    
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
            indexes = tf.concat([
                tf.where(res_ids == id_i) for id_i in ids
            ], axis = 0)
            embeddings = tf.gather(embeddings, indexes)
            res_ids = tf.gather(res_ids, indexes)
        
        return embeddings, res_ids
    
    def distance(self, x, ids = None, use_mean = False):
        """ Compute distance between x and embeddings for given ids """
        embeddings, ids = self.get_embeddings(ids, use_mean)
        return distance(tf.cast(x, tf.float32), embeddings, method = self.method), ids
    
    def append(self, embeddings, ids):
        self._mean_ids, self._mean_embeddings   = None, None
        if tf.rank(embeddings) == 1: embeddings = tf.expand_dims(embeddings, axis = 0)
        self.ids    = np.concat([self.ids, ids])
        self.embeddings = tf.concat([self.embeddings, embeddings], axis = 0)
        
    def predict(self, x, possible_ids = None, k = None, use_mean = None,
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
        
        if possible_ids is not None and not isinstance(possible_ids, (list, tuple, np.ndarray)):
            possible_ids = [possible_ids]
        
        x = tf.cast(x, tf.float32)
        if tf.rank(x) == 2:
            if possible_ids is None: possible_ids = [None] * len(x)
            elif len(possible_ids) != len(x):
                possible_ids = [possible_ids] * len(x)

            assert len(possible_ids) == len(x)
            
            pool = ThreadPool(target = self.predict)
            for xi, ids_i in zip(x, possible_ids):
                pool.append(kwargs = {
                    'x' : xi, 'possible_ids' : ids_i, 'k' : k, 'use_mean' : use_mean
                })
            pool.start(tqdm = tqdm)
            
            pred = tf.concat(pool.result(), axis = 0)
        else:
            embeddings, ids = self.get_embeddings(possible_ids, use_mean)
            
            pred = knn(x, embeddings, ids, k, self.method)
        
        if plot:
            self.plot(x, pred, ** kwargs)
        
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
            
            if x.ndim == 1: x = np.expand_dims(x, 0)
            if x_ids is not None:
                if not isinstance(x_ids, (list, tuple, np.ndarray, tf.Tensor)):
                    x_ids = [x_ids]
                x_ids = np.array(x_ids)
            else:
                fake_id = 0
                while fake_id in ids: fake_id += 1
                x_ids = np.array([fake_id] * len(x))
                marker_kwargs.setdefault('x', {'c' : 'w'})
            
            assert len(x_ids) == len(x), "Got {} ids for {} vectors".format(len(x_ids), len(x))
            
            embeddings = np.concatenate([embeddings, x], axis = 0)
            ids = np.concatenate([ids, x_ids], axis = 0)
            marker += ['x'] * len(x)
        
        plot_embedding(
            embeddings, ids = ids, marker = np.array(marker), 
            marker_kwargs = marker_kwargs, ** kwargs
        )

@tf.function(experimental_relax_shapes = True)
def knn(x, embeddings, ids, k, distance_metric):
    """
        Compute the k-nn decision procedure for a given x based on a list of labelled embeddings
        
        Return the majoritary id in the `k` nearest neigbors or `-2` if there is an equality
    """
    distances = tf.squeeze(distance(x, embeddings, method = distance_metric))
    
    k_nearest_val, k_nearest_idx = tf.nn.top_k(-distances, k)
    
    nearest_ids = tf.cast(tf.gather(ids, k_nearest_idx), tf.int32)
    counts = tf.math.bincount(nearest_ids)

    nearest_ids = tf.squeeze(tf.where(counts == tf.reduce_max(counts)))

    return tf.cast(nearest_ids, tf.int32) if tf.rank(nearest_ids) == 0 else -2
