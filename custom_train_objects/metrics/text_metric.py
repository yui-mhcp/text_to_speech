
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

class TextMetric(tf.keras.metrics.Metric):
    def __init__(self, pad_value = 0, name = 'edit_distance', ** kwargs):
        super().__init__(name = name, ** kwargs)
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
        if isinstance(y_true, (list, tuple)): y_true = y_true[0]
        
        predicted_codes, _ = tf.nn.ctc_beam_search_decoder(
            tf.transpose(y_pred, [1, 0, 2]),
            tf.fill((tf.shape(y_pred)[0], ), tf.shape(y_pred)[1])
        )
        predicted_codes = tf.cast(predicted_codes[0], tf.int32)
        codes = tf.sparse.from_dense(y_true)
        
        distance = tf.edit_distance(predicted_codes, codes, normalize = False)
        
        self.samples.assign_add(tf.shape(y_pred)[0])
        self.distance.assign_add(tf.reduce_sum(distance))
    
    def result(self):
        mean_dist = self.distance / tf.cast(self.samples, tf.float32)
        return mean_dist

    def get_config(self):
        config = super().get_config()
        config['pad_value'] = self.pad_value
        return config