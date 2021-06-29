import numpy as np
import tensorflow as tf

class TerminateOnNaN(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(TerminateOnNaN, self).__init__(**kwargs)
        
    def on_train_batch_end(self, batch, logs = None):
        logs = {} if logs is None else logs
        if 'loss' in logs:
            if np.isnan(logs['loss']) or np.isinf(logs['loss']):
                raise ValueError("NaN loss at batch : {}\n  Logs : {}".format(batch, logs))
                