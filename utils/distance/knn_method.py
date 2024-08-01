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

import keras
import numpy as np
import pandas as pd

from utils.embeddings import load_embeddings, compute_centroids, embeddings_to_np, get_embeddings_with_ids
from utils.keras_utils import TensorSpec, graph_compile, ops
from utils.distance.distance_method import distance

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
                 
                 distance_metric    = 'euclidian',
                 weighted   = False,
                 
                 ** kwargs
                ):
        """
            Constructor for the KNN class
            
            Arguments : 
                - embeddings    : the embeddings to use as labelled points
                    If str  : call `load_embeddings()` on it
                    If pd.DataFrame : use the 'id' column for `ids` and call `embeddings_to_np` on it
                    Else    : must be a np.ndarray / `Tensor` 2D matrix
                - ids   : ids of the embeddings (if embeddings is a DataFrame, ids = embeddings['id'].values)
                - k / use_mean  : default configuration for the `predict` method
                - method        : distance method to use
                - weighted      : whether to use a weighted knn or not
        """
        if isinstance(embeddings, str):
            embeddings = load_embeddings(embeddings)
        
        if isinstance(embeddings, pd.DataFrame):
            if ids is None:
                ids     = embeddings['id'].values
            elif isinstance(ids, str):
                ids     = embeddings[ids].values
            embeddings  = embeddings_to_np(embeddings)
        elif isinstance(embeddings, dict):
            ids, embeddings = list(embeddings.keys()), list(embeddings.values())
        
        assert len(embeddings) == len(ids), '{} embeddings vs {} ids'.format(len(embeddings), len(ids))
        
        self.ids        = np.array(ids)
        self.embeddings = ops.cast(embeddings, 'float32')
        
        self.k          = ops.cast(k, dtype = 'int32')
        self.use_mean   = use_mean
        self.distance_metric    = distance_metric
        self.weighted   = weighted
        
        self._mapping   = None
        self.__centroids    = None
    
    @property
    def centroids(self):
        if self.__centroids is None: self.__centroids = self.compute_centroids()
        return self.__centroids
    
    @property
    def int_ids(self):
        if ops.is_int(self.ids): return self.ids
        self._mapping, ids = ops.unique(self.ids)
        return ids
    
    def __setitem__(self, idx, val):
        if val != self.ids[idx]:
            self.ids[idx]   = val
            self.__centroids    = None
    
    def compute_centroids(self):
        """ Compute the mean embeddings for each id (namely the centroids) """
        ids, centroids = compute_centroids(self.embeddings, self.int_ids)
        if self._mapping is not None: ids = ops.gather(self._mapping, ids)
        return ids, centroids
    
    def get_embeddings(self, ids = None, use_mean = False):
        """ Return all (ids, embeddings) for the expected `ids` """
        if ids is not None and not isinstance(ids, (list, tuple, np.ndarray)): ids = [ids]
        res_ids, embeddings = self.centroids if use_mean else (self.ids, self.embeddings)
        
        if ids is not None:
            embeddings, res_ids = get_embeddings_with_ids(embeddings, res_ids, ids)

        return res_ids, embeddings
    
    def distance(self, x, ** kwargs):
        """ Compute distance between x and embeddings for given ids """
        embeddings, ids = self.get_embeddings(** kwargs)
        return tf_distance(ops.cast(x, 'float32'), embeddings, method = self.distance_metric), ids
    
    def predict(self,
                query,
                possible_ids = None,
                k   = None,
                use_mean    = None,
                weighted    = None,
                max_matrix_size = -1,
                
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
        k = ops.cast(k if k is not None else self.k, 'int32')
        
        if possible_ids is not None and not isinstance(possible_ids, (list, tuple, np.ndarray)):
            possible_ids = [possible_ids]
        
        query = ops.cast(query, 'float32')

        ids, embeddings = self.get_embeddings(possible_ids, use_mean)
        
        pred = knn(
            query,
            embeddings,
            ids = ids,
            k   = k,
            max_matrix_size = max_matrix_size,
            distance_metric = self.distance_metric,
            weighted = weighted
        )
        
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
        centroid_ids, centroids = self.centroids
        embeddings = np.concatenate([embeddings, centroids], axis = 0)
        ids = np.concatenate([ids, centroid_ids], axis = 0)
        
        marker += ['O'] * len(centroid_ids)
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
                if not isinstance(x_ids, (list, tuple, np.ndarray)): x_ids = [x_ids]
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

@graph_compile()
def knn(query   : TensorSpec(shape = None, dtype = 'float') = None,
        embeddings  : TensorSpec(shape = (None, None), dtype = 'float') = None,
        distance_matrix : TensorSpec(shape = (None, None), dtype = 'float') = None,
        
        k   : TensorSpec(shape = (), dtype = 'int32') = 5,
        ids : TensorSpec(shape = (None, )) = None,
        distance_metric = None,
        max_matrix_size = -1,
        
        weighted    : TensorSpec(shape = (), dtype = 'bool') = False,

        return_scores   = False,
        return_index    = False,
        ** kwargs
       ):
    """
        Compute the k-nn decision procedure for a given x based on a list of labelled embeddings
        
        Arguments :
            - query : the query point(s), 1-D or 2-D `Tensor`
            - embeddings    : the points to use
            - distance_matrix   : the already computed distance matrix (2-D) between `query` and `embeddings`
            
            - ids   : the ids for `embeddings` (**must be numeric values**)
            - k     : the `k` hyperparameter in the K-NN
            - distance_metric   : the metric to use to compute distance (irrelevant if passing `distance_matrix`)
            
            - return_index  : whether to return the nearest indexes
            - weighted      : whether to use the weighted KNN algorithm
            
            - kwargs    : passed to `distance`
        Return :
            If `ids` is not None    : 1-D `Tensor` with the nearest ids
            elif `return_index`     : 1-D `Tensor`, the nearest embeddings' indexes
            else                    : 2-D `Tensor`, the k-nearest embeddings
    """
    assert distance_matrix is not None or (query is not None and embeddings is not None and distance_metric)
    
    if distance_matrix is None:
        distance_matrix = distance(
            query,
            embeddings,
            distance_metric,
            max_matrix_size = max_matrix_size,
            force_distance  = True,
            as_matrix   = True,
            
            ** kwargs
        )
    # the `- distance` is required as `top_k` takes the highest values
    # while we want the nearest points, i.e., those with the lowest distances
    k_nearest_dists, k_nearest_idx = ops.top_k(- distance_matrix, k)
    
    if ids is None:
        return k_nearest_idx if return_index else ops.take(
            embeddings, k_nearest_idx, axis = 0
        )
    
    unique_ids, pos_ids = ops.unique(ids, return_inverse = True)
    
    nearest_ids = ops.take(pos_ids, k_nearest_idx, axis = 0)

    # shape == [len(query), k]
    weights = ops.cond(
        weighted,
        lambda: 1. / ops.maximum(-k_nearest_dists, 1e-9),
        lambda: ops.ones_like(k_nearest_dists)
    )
    # shape == [len(points), len(unique_ids), k]
    mask    = ops.arange(ops.size(unique_ids))[None, :, None] == nearest_ids[:, None, :]
    # shape = [len(query), len(unique_ids)]
    scores  = ops.sum(weights[:, None, :] * ops.cast(mask, weights.dtype), axis = -1)
    #scores = K.bincount(
    #    nearest_ids, weights = weights, minlength = K.size(unique_ids)
    #)

    if return_scores: return unique_ids, scores
    return ops.take(unique_ids, ops.argmax(scores, axis = -1))
