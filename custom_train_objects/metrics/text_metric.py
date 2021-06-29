import tensorflow as tf

class TextMetric(tf.keras.metrics.Metric):
    def __init__(self, pad_value = 0, name = 'TextMetrics', ** kwargs):
        super(TextMetric, self).__init__(name = name, ** kwargs)
        self.pad_value = pad_value
        
        self.samples    = self.add_weight("num_samples", initializer = "zeros", dtype = tf.int32)
        self.distance   = self.add_weight("edit_distance", initializer = "zeros")
    
    @property
    def metric_names(self):
        return ["edit_distance"]
    
    def update_state(self, y_true, y_pred):
        """
            Arguments : 
                - y_true : [codes, lengths]
                    codes   : expected values with shape (batch_size, max_length)
                    lengths : length for the ctc_decode (batch_size)
                - y_pred : predicted logits
                    shape : (batch_size, max_length, vocab_size)
        """
        codes, lengths = y_true
        
        predicted_codes, _ = tf.nn.ctc_beam_search_decoder(
            tf.transpose(y_pred, [1, 0, 2]),
            tf.zeros((tf.shape(y_pred)[0],), dtype = tf.int32) + tf.shape(y_pred)[1]
        )
        predicted_codes = tf.cast(predicted_codes[0], tf.int32)
        codes = tf.sparse.from_dense(codes)
        
        distance = tf.edit_distance(predicted_codes, codes, normalize = False)
        
        self.samples.assign_add(tf.shape(y_pred)[0])
        self.distance.assign_add(tf.reduce_sum(distance))
    
    def result(self):
        mean_dist = self.distance / tf.cast(self.samples, tf.float32)
        return mean_dist

    def get_config(self):
        config = super(TextMetric, self).get_config()
        config['pad_value'] = self.pad_value
        return config