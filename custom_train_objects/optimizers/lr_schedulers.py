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
import keras.ops as K

class CustomScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, minval = 1e-5, maxval = 0.01, ** kwargs):
        self.minval = K.convert_to_tensor(minval, 'float32')
        self.maxval = K.convert_to_tensor(maxval, 'float32')
    
    def clip(self, lr):
        return K.clip(lr, self.minval, self.maxval)
    
    def get_config(self):
        return {
            'minval'    : float(K.convert_to_numpy(self.minval)),
            'maxval'    : float(K.convert_to_numpy(self.maxval))
        }
    
    def plot(self, n_step = 10000, xlabel = 'step', ylabel = 'Learning rate', ** kwargs):
        from utils import plot
        
        kwargs.setdefault('title', 'Learning rate over steps')
        
        x = K.arange(1, 1 + n_step)
        y = K.convert_to_numpy(self(x))
        return plot(K.convert_to_numpy(x), y, ** kwargs)
    
        plot(x, y, xlabel = xlabel, ylabel = ylabel, ** kwargs)

class DivideByStep(CustomScheduler):
    def __init__(self, factor = 1., maxval = 1e-2, minval = 1e-6, ** kwargs):
        super().__init__(** kwargs)
        self.factor = K.convert_to_tensor(factor, 'float32')
        
    def __call__(self, step):
        return self.clip(self.factor / K.cast(step, 'float32'))
    
    def get_config(self):
        config = super().get_config()
        config.update({'factor' : float(K.convert_to_numpy(self.factor))})
        return config

class ReduceEvery(CustomScheduler):
    def __init__(self, base = 1e-3, step = 10, factor = 0.1, ** kwargs):
        super().__init__(** kwargs)
        self.base   = K.convert_to_tensor(base, 'float32')
        self.step   = K.convert_to_tensor(step, 'int32')
        self.factor = K.convert_to_tensor(factor, 'float32')
    
    def __call__(self, step):
        return self.clip(self.base * (self.factor ** K.cast(self.step // step, 'float32')))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'factor'    : float(K.convert_to_numpy(self.factor)),
            'base'      : float(K.convert_to_numpy(self.base)),
            'step'      : int(K.convert_to_numpy(self.step))
        })
        return config

class WarmupScheduler(CustomScheduler):
    def __init__(self, factor = 8, warmup_steps = 2048, ** kwargs):
        super().__init__(** kwargs)
        self.factor = K.convert_to_tensor(factor, 'float32')
        self.warmup_steps   = K.convert_to_tensor(warmup_steps, 'float32')
        
    def __call__(self, step):
        step = K.cast(step, 'float32')
        x1 = K.rsqrt(step)
        x2 = step * (self.warmup_steps ** -1.5)
        lr = K.rsqrt(self.factor) * K.minimum(x1, x2)
        return self.clip(lr)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'factor'        : float(K.convert_to_numpy(self.factor)),
            'warmup_steps'  : int(K.convert_to_numpy(self.warmup_steps))
        })
        return config

class SinScheduler(CustomScheduler):
    def __init__(self, period = 1024, with_decay = True,  ** kwargs):
        super().__init__(** kwargs)
        self.period     = K.convert_to_tensor(period, 'float32')
        self.with_decay = with_decay
        
        self.range = maxval - minval
        self.decay_factor = K.convert_to_tensor(1. / (np.pi * period), 'float32')
        
    def __call__(self, step):
        step = K.cast(step, 'float32')
        t = step / self.period * K.cast(np.pi, 'float32') * 2
        lr = (K.sin(t) / 2. + 0.5) * self.range
        if self.with_decay: lr = lr / (step * self.decay_factor + 1)
        return lr + self.minval
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'period'    : int(K.convert_to_numpy(self.period)),
            'with_decay'    : self.with_decay
        })
        return config

class TanhDecayScheduler(CustomScheduler):
    def __init__(self, period = 2048, ** kwargs):
        super().__init__(** kwargs)
        self.period = K.convert_to_tensor(period, 'float32')
        self.range  = self.maxval - self.minval
        
    def __call__(self, step):
        step = K.cast(step, 'float32')
        t = step / self.period
        lr = 1. - (K.tanh(t - K.cast(np.pi, 'float32')) / 2. + 0.5)

        return lr * self.range + self.minval
        
    def get_config(self):
        config = super().get_config()
        config.update({'period' : int(K.convert_to_numpy(self.period))})
        return config
