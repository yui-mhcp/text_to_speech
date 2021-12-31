import logging
import tensorflow as tf

class PredictorCallback(tf.keras.callbacks.Callback):
    def __init__(self, method, generator, initial_step = 0, pred_every = 1000,
                 prefix = 'pred_step-{step:06d}_{batch}', 
                 *args, **kwargs):
        self.step   = initial_step
        self.method = method
        self.batch_generator    = generator
        self.prefix = prefix
        self.args   = args
        self.kwargs = kwargs
        self.pred_every = pred_every
        
    def on_train_batch_end(self, batch_num, logs = None):
        self.step += 1
        if self.pred_every > 0 and self.step % self.pred_every == 0:
            self._predict(logs)
        
    def on_epoch_end(self, epoch, logs = None):
        if self.pred_every < 0 and (epoch + 1) % abs(self.pred_every) == 0:
            self._predict(logs)
    
    def _predict(self, logs = None):
        logging.info("\nMaking prediction at step {}".format(self.step))
        if logs is None: logs = {}
        for i, batch in enumerate(self.batch_generator):
            prefix = self.prefix.format(batch = i, step = self.step, **logs)
            self.method(batch, *self.args, step = self.step, prefix = prefix, **self.kwargs)
        
