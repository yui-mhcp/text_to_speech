import numpy as np
import tensorflow as tf

from utils import plot

class CustomScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, **kwargs):
        super(CustomScheduler, self).__init__(**kwargs)
        
    def get_config(self):
        return {}
        
    def plot(self, n_step = 10000, **kwargs):
        x = tf.range(1, 1 + n_step)
        y = np.array(self(x))
        plot(x, y, xlabel = "step", ylabel = "learning_rate", 
             title = "Learning rate over epoch", **kwargs)

class DivideByStep(CustomScheduler):
    def __init__(self, factor = 1., maxval = 1e-2, minval = 1e-6, ** kwargs):
        super(DivideByStep, self).__init__(** kwargs)
        self.factor = tf.cast(factor, tf.float32)
        self.minval = tf.cast(minval, tf.float32)
        self.maxval = tf.cast(maxval, tf.float32)
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.clip_by_value(
            self.factor / step, clip_value_min = self.minval, clip_value_max = self.maxval
        )
    
    def get_config(self):
        config = super(DivideByStep, self).get_config()
        config['factor'] = self.factor
        config['minval'] = self.minval
        config['maxval'] = self.maxval
        return config

class WarmupScheduler(CustomScheduler):
    def __init__(self, factor = 8, warmup_steps = 2048, minval = 5e-4, maxval = 1e-2, 
                 **kwargs):
        super(WarmupScheduler, self).__init__(**kwargs)
        self.factor = float(factor)
        self.warmup_steps   = warmup_steps
        self.minval = minval
        self.maxval = maxval
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        x1 = tf.math.rsqrt(step)
        x2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.factor) * tf.math.minimum(x1, x2)
        return tf.clip_by_value(
            lr, clip_value_min = self.minval, clip_value_max = self.maxval
        )
    
    def get_config(self):
        config = super(WarmupScheduler, self).get_config()
        config['factor'] = self.factor
        config['warmup_steps']  = self.warmup_steps
        config['minval']    = self.minval
        config['maxval']    = self.maxval
        return config

class SinScheduler(CustomScheduler):
    def __init__(self, period = 1024, minval = 5e-4, maxval = 5e-3, with_decay = True, 
                 **kwargs):
        super(SinScheduler, self).__init__(**kwargs)
        self.period     = float(period)
        self.minval     = minval
        self.maxval     = maxval
        self.with_decay = with_decay
        self.range = maxval - minval
        self.decay_factor = tf.cast(1. / (np.pi * period), tf.float32)
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        t = step / self.period * np.pi * 2
        lr = (tf.math.sin(t) / 2. + 0.5) * self.range
        if self.with_decay: lr = lr / (step * self.decay_factor + 1)
        return lr + self.minval
    
    def get_config(self):
        config = super(SinScheduler, self).get_config()
        config['period']     = self.period
        config['minval']     = self.minval
        config['maxval']     = self.maxval
        config['with_decay']    = self.with_decay
        return config

class TanhDecayScheduler(CustomScheduler):
    def __init__(self, period = 2048, minval = 5e-4, maxval = 5e-3, 
                 **kwargs):
        super(TanhDecayScheduler, self).__init__(**kwargs)
        self.period     = float(period)
        self.minval     = minval
        self.maxval     = maxval
        self.range = maxval - minval
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        t = step / self.period
        lr = 1. - (tf.math.tanh(t - np.pi) / 2. + 0.5)

        return lr * self.range + self.minval
        
    def get_config(self):
        config = super(TanhDecayScheduler, self).get_config()
        config['period']     = self.period
        config['minval']     = self.minval
        config['maxval']     = self.maxval
        return config
