
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

class TextAccuracy(tf.keras.metrics.Metric):
    def __init__(self, mask_padding = True, pad_value = 0,
                 name = 'TextAccuracy', ** kwargs):
        super(TextAccuracy, self).__init__(name = name)
        self.mask_padding   = mask_padding
        self.pad_value      = tf.cast(pad_value, tf.int32)
        
        self.samples        = self.add_weight("num_batches", initializer = "zeros", dtype = tf.int32)
        self.true_sentences = self.add_weight("true_sentences", initializer = "zeros")
        self.true_symbols   = self.add_weight("true_symbols", initializer = "zeros")
    
    @property
    def metric_names(self):
        return ["accuracy", "sentence_accuracy"]
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        """
            Arguments : 
                - y_true : expected values with shape (batch_size, max_length)
                - y_pred : predicted values probabilities
                    shape : (batch_size, max_length, vocab_size)
        """
        skip_length = 0
        if not isinstance(y_true, (list, tuple)):
            target_length = tf.reduce_sum(tf.cast(
                tf.math.not_equal(y_true, self.pad_value), tf.int32
            ), axis = -1)
        else:
            if len(y_true) == 3: skip_length = y_true[2]
            y_true, target_length = y_true[:2]
        
        if len(tf.shape(y_true)) == 3:
            y_true, target_length = y_true[:, 0], target_length[:, 0]
        
        padding_mask    = tf.sequence_mask(
            skip_length + target_length, maxlen = tf.shape(y_pred)[1], dtype = tf.float32
        )
        if tf.reduce_any(skip_length > 0):
            padding_mask    = tf.minimum(
                padding_mask, 1 - tf.sequence_mask(
                    skip_length, maxlen = tf.shape(y_pred)[1], dtype = tf.float32
                )
            )
        
        
        n_symbols = tf.reduce_sum(padding_mask, axis = -1, keepdims = True)
        
        true_symbols = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred) * padding_mask
        
        true_symbols = tf.reduce_sum(true_symbols, axis = -1, keepdims = True) / n_symbols
        true_phrases = tf.cast(true_symbols == 1, dtype = tf.float32)
        
        score_symbols = tf.reduce_sum(true_symbols)
        score_phrases = tf.reduce_sum(true_phrases)
        
        self.samples.assign_add(tf.shape(y_true)[0])
        self.true_sentences.assign_add(score_phrases)
        self.true_symbols.assign_add(score_symbols)
    
    def result(self):
        score_symbols = self.true_symbols / tf.cast(self.samples, tf.float32)
        score_phrases = self.true_sentences / tf.cast(self.samples, tf.float32)
        return score_symbols, score_phrases

    def get_config(self):
        config = super().get_config()
        config.update({
            'mask_padding'  : self.mask_padding,
            'pad_value'     : self.pad_value.numpy()
        })
        return config