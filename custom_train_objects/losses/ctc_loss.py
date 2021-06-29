import tensorflow as tf

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, pad_value = 0, name = 'CTCLoss', **kwargs):
        super(CTCLoss, self).__init__(name = name, **kwargs)
        self.pad_value      = pad_value
    
    def call(self, y_true, y_pred):
        if not isinstance(y_true, (list, tuple)):
            target_length = tf.reduce_sum(tf.cast(tf.math.not_equal(y_true, self.pad_value), tf.int32), axis = -1)
        else:
            y_true, target_length = y_true
        
        pred_length = tf.zeros_like(target_length) + tf.shape(y_pred)[1]
        
        loss = tf.nn.ctc_loss(
            y_true, y_pred, target_length, pred_length, logits_time_major = False,
            blank_index = self.pad_value
        )

        return tf.reduce_mean(loss / (tf.cast(target_length, tf.float32) + 1e-6))
    
    def get_config(self):
        config = super(CTCLoss, self).get_config()
        config['pad_value']     = self.pad_value
        return config
