
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

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, pad_value = 0, name = 'CTCLoss', **kwargs):
        super().__init__(name = name, ** kwargs)
        self.pad_value      = pad_value
    
    def call(self, y_true, y_pred):
        if not isinstance(y_true, (list, tuple)):
            target_length = tf.reduce_sum(tf.cast(
                tf.math.not_equal(y_true, self.pad_value), tf.int32
            ), axis = -1)
        else:
            y_true, target_length = y_true
        
        pred_length = tf.fill(tf.shape(target_length), tf.shape(y_pred)[0])
        
        loss = tf.nn.ctc_loss(
            y_true, y_pred, target_length, pred_length, logits_time_major = False,
            blank_index = self.pad_value
        )

        return loss / tf.maximum(tf.cast(target_length, tf.float32), 1e-6)
    
    def get_config(self):
        config = super().get_config()
        config['pad_value'] = self.pad_value
        return config
