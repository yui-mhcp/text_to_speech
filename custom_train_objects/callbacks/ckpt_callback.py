import time
import logging
import tensorflow as tf

MIN_MODE    = 0
MAX_MODE    = 0

class CkptCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint, directory, save_best_only = False,
                 monitor = 'val_loss', mode = MIN_MODE, max_to_keep = 1, verbose = True,
                 save_every_hour = True, **kwargs):
        super(CkptCallback, self).__init__(**kwargs)
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, directory = directory, 
                                                       max_to_keep = max_to_keep)
        self.save_best_only = save_best_only
        self.monitor    = monitor
        self.prev_val   = None
        self.verbose    = verbose
        self.save_every_hour    = save_every_hour
        self.last_saving_time   = time.time()
        
        self.compare = lambda x_prev, x: x < x_prev if mode == MIN_MODE else lambda x_prev, x: x > x_prev
        
    def on_train_begin(self, *args):
        self.last_saving_time = time.time()
        
    def on_train_batch_end(self, *args, **kwargs):
        if self.save_every_hour and time.time() - self.last_saving_time > 3600:
            self.save()
        
    def on_epoch_end(self, epoch, logs):
        if self.save_best_only and self.monitor in logs:            
            new_val = logs[self.monitor]
            if self.prev_val is None or self.compare(self.prev_val, new_val):
                self.save(epoch + 1)
                self.prev_val = new_val
        else:
            self.save(epoch + 1)
        
    def save(self, epoch = None):
        if self.verbose:
            if epoch:
                logging.info("\nSaving at epoch {} !".format(epoch))
            else:
                logging.info("\nSaving after 1 hour training !")
        self.ckpt_manager.save()
        self.last_saving_time = time.time()
        