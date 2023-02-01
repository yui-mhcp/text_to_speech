
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

import tensorflow as tf

from utils.distance import distance

class SimilarityLayer(tf.keras.layers.Layer):
    def __init__(self,
                 distance_metric,
                 pred_probability   = False,
                 probability_fn = None,
                 pred_matrix    = False,
                 ** kwargs
                ):
        super().__init__(** kwargs)
        
        self.distance_metric    = distance_metric
        self.pred_probability   = pred_probability
        self.probability_fn = probability_fn
        self.pred_matrix    = pred_matrix
        
        self._proba_fn  = None
        if probability_fn is not None:
            assert probability_fn in ('sigmoid', 'softmax')
            self._proba_fn = tf.sigmoid if probability_fn == 'sigmoid' else tf.softmax
        
        self.decision_layer = tf.keras.layers.Dense(1, activation = None, name = 'decision_layer')
    
    def call(self,
             inputs,
             training   = False,
             pred_matrix    = None,
             probability_fn = None,
             pred_probability   = None,
             max_matrix_size = -1
            ):
        """
            Perform distance computation between `inputs[0]` and `inputs[1]`
            
            Note : this layer can be called inside a `tf.keras.Model` or outside and can have some additional features
            
            Arguments : 
                - inputs    : 2-element tuple / list, the embeddings to compare
                    - In `tf.keras.Model` : both must have shape `[batch_size, embedding_dim]`
                    - Outsize : `inputs[0]` must have shape `[batch_size, embedding_dim]` but `inputs[1]` can have shape `[batch_size, n, embedding_dim]`
                - pred_matrix   : special argument available only when called outside the `tf.keras.Model` !
                    It will compute distance between all pairs of vectors
                    Note that if `pred_matrix` is True, shapes can be : 
                        - `inputs[0].shape` : `[batch_size_1, embedding_dim]`
                        - `inputs[1].shape` : `[batch_size_2, embedding_dim]`
                        and will compute distance matrix of shape `[batch_size_1, batch_size_2]`
            Return : distances between `inputs[0]` and `inputs[1]`
                Inside `tf.keras.Model` : `output.shape == [batch_size, 1]`
                Outside : 
                    If `pred_matrix == False` :
                        If `inputs[1].shape == [batch_size, embedding_dim]` :
                            `output.shape == [batch_size, 1]`
                        If `inputs[1].shape == [batch_size, n, embedding_dim]` :
                            `output.shape == [batch_size, n, 1]`
                    If `pred_matrix == True` :
                        If `inputs[1].shape == [batch_size_2, embedding_dim]` :
                            `output.shape == [batch_size, batch_size_2]`
                            
        """
        if pred_matrix is None: pred_matrix = self.pred_matrix
        if pred_probability is None: pred_probability = self.pred_probability
        
        if isinstance(inputs, (list, tuple)):
            embedded_1, embedded_2 = inputs
        else:
            pred_matrix = True
            embedded_1, embedded_2 = inputs, inputs

        distances = distance(
            embedded_1,
            embedded_2,
            method      = self.distance_metric,
            as_matrix   = pred_matrix,
            max_matrix_size = max_matrix_size,
            force_distance  = False
        )

        if len(tf.shape(distances)) < len(tf.shape(embedded_2)):
            distances = tf.expand_dims(distances, -1)

        if not pred_matrix:
            output = self.decision_layer(distances)
            return tf.sigmoid(output) if pred_probability else output

        b, n = tf.shape(distances)[0], tf.shape(distances)[1]
        last_dim = 1 if len(tf.shape(distances)) == 2 else tf.shape(distances)[-1]
        distances = tf.reshape(distances, [b * n, last_dim])
        
        output = self.decision_layer(distances)
        output = tf.reshape(output, [b, n])
        if pred_probability:
            if probability_fn is None:
                probability_fn = tf.nn.softmax if self._proba_fn is None else self._proba_fn
            output = probability_fn(output, axis = -1)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config['pred_matrix']   = self.pred_matrix
        config['distance_metric']   = self.distance_metric
        config['pred_probability']  = self.pred_probability
        config['probability_fn']    = self.probability_fn
        return config
