
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

class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, pad_value = 0, name = 'MaskedLoss', **kwargs):
        super(MaskedLoss, self).__init__(name = name, **kwargs)
        self.pad_value      = float(pad_value)
    
    def call(self, y_true, y_pred):
        mask = tf.cast(tf.math.equal(y_true, self.pad_value), y_true.dtype)
        y_pred = y_pred * mask
        
        loss = self.loss_fn(y_true, y_pred)
        
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-6)
    
    def get_config(self):
        config = super(MaskedLoss, self).get_config()
        config['pad_value']     = self.pad_value
        return config

class MaskedMSE(MaskedLoss):
    def __init__(self, name = 'MaskedMSE', ** kwargs):
        super(MaskedMSE, self).__init__(name = name, ** kwargs)
    
    def loss_fn(self, y_pred, y_true):
        return tf.square(y_true - y_pred)
        
class MaskedMAE(MaskedLoss):
    def __init__(self, name = 'MaskedMAE', ** kwargs):
        super(MaskedMAE, self).__init__(name = name, ** kwargs)
    
    def loss_fn(self, y_pred, y_true):
        return tf.abs(y_true - y_pred)
