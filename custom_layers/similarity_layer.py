import tensorflow as tf

from utils.distance import distance

class SimilarityLayer(tf.keras.layers.Layer):
    def __init__(self, distance_metric, pred_probability = False, pred_matrix = False, ** kwargs):
        super().__init__(** kwargs)
        
        self.distance_metric    = distance_metric
        self.pred_probability   = pred_probability
        
        self.decision_layer = tf.keras.layers.Dense(1, activation = None, name = 'decision_layer')
    
    def call(self, inputs, training = False, pred_matrix = False, max_matrix_size = -1):
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
        embedded_1, embedded_2 = inputs

        distances = distance(
            embedded_1, embedded_2, method = self.distance_metric,
            as_matrix = pred_matrix, max_matrix_size = max_matrix_size
        )
        if len(tf.shape(distances)) < len(tf.shape(embedded_2)): distances = tf.expand_dims(distances, -1)

        if not pred_matrix:
            output = self.decision_layer(distances)
            if self.pred_probability: output = tf.sigmoid(output)
            return output

        b, n = tf.shape(distances)[0], tf.shape(distances)[1]
        last_dim = 1 if len(tf.shape(distances)) == 2 else tf.shape(distances)[-1]
        distances = tf.reshape(distances, [b * n, -1])
        
        output = self.decision_layer(distances)
        output = tf.reshape(output, [b, n])
        if self.pred_probability: output = tf.nn.softmax(output, axis = -1)
            
        return output
    
    def get_config(self):
        config = super().get_config()
        config['distance_metric']   = self.distance_metric
        config['pred_probability']  = self.pred_probability
        return config
