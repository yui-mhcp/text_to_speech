
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
import pandas as pd
import tensorflow as tf

from utils.embeddings import load_embedding, compute_centroids, embeddings_to_np, get_embeddings_with_ids
from utils.distance.distance_method import tf_distance

class KNN(object):
    """
        Tensorflow implementation of the `K-Nearest Neighbors (KNN)` algorithm
        
        It also has some additional features such as : 
            - Plotting embeddings / predictions
            - Use a `use_mean` version where the prediction is the nearest `centroid`*
            - A `weighted` version
            
        * A `centroid` is the mean point of all points belonging to a given label
    """
    def __init__(self,
                 embeddings,
                 ids    = None,
                 
                 k  = 5,
                 use_mean   = False, 
                 
                 method     = 'euclidian',
                 weighted   = False,
                 
                 ** kwargs
                ):
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
                - weighted      : whether to use a weighted knn or not
        """
        if isinstance(embeddings, str):
            embeddings = load_embedding(embeddings)
        
        if isinstance(embeddings, pd.DataFrame):
            ids         = embeddings['id'].values
            embeddings  = embeddings_to_np(embeddings)
        
        assert len(embeddings) == len(ids), '{} embeddings vs {} ids'.format(len(embeddings), len(ids))
        
        self.ids        = np.array(ids)
        self.embeddings = tf.cast(embeddings, tf.float32)
        
        self.k          = tf.cast(k, dtype = tf.int32)
        self.use_mean   = use_mean
        self.method     = method
        self.weighted   = weighted
        
        self.__centroids    = None
    
    @property
    def centroids(self):
        if self.__centroids is None: self.__centroids = self.compute_centroids()
        return self.__centroids
    
    def __setitem__(self, idx, val):
        if val != self.ids[idx]:
            self.ids[idx]   = val
            self.__centroids    = None
    
    def compute_centroids(self):
        """ Compute the mean embeddings for each id (namely the centroids) """
        return compute_centroids(self.embeddings, self.ids)
    
    def get_embeddings(self, ids = None, use_mean = False):
        """ Return all (ids, embeddings) for the expected `ids` """
        if ids is not None and not isinstance(ids, (list, tuple, np.ndarray, tf.Tensor)): ids = [ids]
        res_ids, embeddings = self.centroids if use_mean else (self.ids, self.embeddings)
        
        if ids is not None:
            embeddings, res_ids = get_embeddings_with_ids(embeddings, res_ids, ids)

        return res_ids, embeddings
    
    def distance(self, x, ** kwargs):
        """ Compute distance between x and embeddings for given ids """
        embeddings, ids = self.get_embeddings(** kwargs)
        return tf_distance(tf.cast(x, tf.float32), embeddings, method = self.method), ids
    
    def predict(self,
                query,
                possible_ids = None,
                k   = None,
                use_mean    = None,
                weighted    = None,

                tqdm    = lambda x: x,
                
                plot    = False,
                plot_kwargs = {},
                ** kwargs
               ):
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
        if weighted is None: weighted = self.weighted
        if use_mean: k = 1
        k = tf.cast(k if k is not None else self.k, tf.int32)
        
        if possible_ids is not None and not isinstance(possible_ids, (list, tuple, np.ndarray, tf.Tensor)):
            possible_ids = [possible_ids]
        
        query = tf.cast(query, tf.float32)

        ids, embeddings = self.get_embeddings(possible_ids, use_mean)
        
        pred = knn(query, embeddings, ids, k, self.method, weighted = weighted)
        
        if plot: self.plot(query, pred, ** plot_kwargs)
        
        return pred
    
    def plot(self, x = None, x_ids = None, marker_kwargs = None, ** kwargs):
        """
            Plot the labelled datasets + centroids + possible `x` to predict (with their predicted labels) 
        """
        from utils.plot_utils import plot_embedding

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

def knn(query   = None,
        embeddings  = None,
        distance_matrix = None,
        
        k   = 5,
        ids = None,
        distance_metric = None,
        
        return_index    = False,
        weighted    = False,
        ** kwargs
       ):
    """
        Compute the k-nn decision procedure for a given x based on a list of labelled embeddings
        
        Arguments :
            - query : the query point(s), 1-D or 2-D (tf.Tensor or np.ndarray)
            - embeddings    : the points to use
            - distance_matrix   : the already computed distance matrix (2-D) between `query` and `embeddings`
            
            - ids   : the ids for `embeddings` (**must be numeric values**)
            - k     : the `k` hyperparameter in the K-NN
            - distance_metric   : the metric to use to compute distance (irrelevant if passing `distance_matrix`)
            
            - return_index  : whether to return the nearest indexes
            - weighted      : whether to use the weighted KNN algorithm
            
            - kwargs    : passed to `distance`
        Return :
            If `ids` is not None    : 1-D tf.Tensor with the nearest ids
            elif `return_index`     : 1-D tf.Tensor, the nearest embeddings' indexes
            else                    : 2-D tf.Tensor, the k-nearest embeddings
    """
    assert distance_matrix is not None or (query is not None and embeddings is not None and distance_metric)
    
    if distance_matrix is None:
        distance_matrix = tf_distance(
            query, embeddings, distance_metric, as_matrix = True, force_distance = True
        )

    k_nearest_dists, k_nearest_idx = tf.nn.top_k(
        - distance_matrix, tf.minimum(tf.shape(distance_matrix)[1], k)
    )
    
    if ids is None:
        return k_nearest_idx if return_index else tf.gather(embeddings, k_nearest_idx, batch_dims = 1)
    
    nearest_ids = tf.cast(tf.gather(tf.reshape(ids, [-1]), k_nearest_idx), tf.int32)

    if not weighted:
        counts = tf.cast(tf.math.bincount(nearest_ids, axis = -1), tf.float32)
    else:
        indices = tf.reshape(tf.range(tf.reduce_max(nearest_ids + 1), dtype = tf.int32), [1, 1, -1])
        expanded_nearest = tf.expand_dims(nearest_ids, axis = -1)
        
        mask = tf.cast(indices == expanded_nearest, tf.float32)
        
        counts = tf.reduce_sum(
            mask * tf.expand_dims(1. / tf.maximum(-k_nearest_dists, 1e-9), axis = -1), axis = 1
        )
    
    max_counts = tf.reduce_max(counts, axis = -1, keepdims = True)
    
    max_idx = tf.cast(counts == max_counts, tf.int32)
    
    nb_nearest = tf.reduce_sum(max_idx, axis = -1)
    
    nearest_ids = tf.where(
        nb_nearest == 1, tf.cast(tf.argmax(max_idx, axis = -1), tf.int32), -2
    )

    return nearest_ids
