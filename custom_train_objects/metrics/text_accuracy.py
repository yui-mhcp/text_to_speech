
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
    def __init__(self, mask_padding = True, pad_value = 0, eos_value = -1,
                 name = 'TextAccuracy', ** kwargs):
        super(TextAccuracy, self).__init__(name = name)
        self.mask_padding   = mask_padding
        self.pad_value      = tf.cast(pad_value, tf.int32)
        self.eos_value      = tf.cast(eos_value, tf.int32)
        self.pad_is_eos     = pad_value == eos_value
        
        self.samples        = self.add_weight("num_batches", initializer = "zeros", dtype = tf.int32)
        self.true_sentences = self.add_weight("true_sentences", initializer = "zeros")
        self.true_symbols   = self.add_weight("true_symbols", initializer = "zeros")
    
    @property
    def metric_names(self):
        return ["accuracy", "exact_match"]
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        """
            Arguments : 
                - y_true : expected values with shape (batch_size, max_length)
                - y_pred : predicted values probabilities
                    shape : (batch_size, max_length, vocab_size)
        """
        if not isinstance(y_pred, tf.Tensor): y_pred = y_pred[0]
        if isinstance(y_true, tuple) and len(y_true) != 2: y_true = y_true[0]
        
        if not isinstance(y_true, (list, tuple)):
            target_length = tf.reduce_sum(tf.cast(
                tf.math.not_equal(y_true, self.pad_value), tf.int32
            ), axis = -1)
            if self.pad_is_eos: target_length += 1
        else:
            y_true, target_length = y_true[:2]
        
        if len(tf.shape(y_true)) == 3:
            y_true, target_length = y_true[:, 0], target_length[:, 0]
        
        accuracy        = tf.keras.metrics.sparse_categorical_accuracy(
            y_true, y_pred[:, - tf.shape(y_true)[1] :]
        )
        padding_mask    = tf.sequence_mask(
            target_length, maxlen = tf.shape(y_true)[1], dtype = self.true_symbols.dtype
        )
        
        true_symbols    = tf.reduce_sum(
            tf.cast(accuracy, self.true_symbols.dtype) * padding_mask, axis = -1
        ) / tf.cast(target_length, self.true_symbols.dtype)
        exact_match     = tf.cast(true_symbols == 1, dtype = self.true_symbols.dtype)
        
        self.samples.assign_add(tf.shape(y_true)[0])
        self.true_symbols.assign_add(tf.reduce_sum(true_symbols))
        self.true_sentences.assign_add(tf.reduce_sum(exact_match))
    
    def result(self):
        score_symbols = self.true_symbols / tf.cast(self.samples, self.true_symbols.dtype)
        score_phrases = self.true_sentences / tf.cast(self.samples, self.true_sentences.dtype)
        return score_symbols, score_phrases

    def get_config(self):
        config = super().get_config()
        config.update({
            'mask_padding'  : self.mask_padding,
            'pad_value'     : self.pad_value.numpy(),
            'eos_value'     : self.eos_value.numpy()
        })
        return config