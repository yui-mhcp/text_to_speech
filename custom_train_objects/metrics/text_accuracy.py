# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras
import keras.ops as K

@keras.saving.register_keras_serializable('custom_loss')
class TextAccuracy(keras.metrics.Metric):
    def __init__(self, pad_value = 0, eos_value = -1, name = 'text_accuracy', ** kwargs):
        super().__init__(name = name)
        self.pad_value  = pad_value
        self.eos_value  = eos_value
        
        self.samples    = self.add_weight(name = 'batches', initializer = 'zeros')
        self.accuracy   = self.add_weight(name = 'accuracy', initializer = 'zeros')
        self.exact_match    = self.add_weight(name = 'exact_match', initializer = 'zeros')
    
    def build_padding_mask(self, y_true):
        if self.eos_value != self.pad_value: return K.not_equal(y_true, self.pad_value)
        lengths = K.count_nonzero(K.not_equal(y_true, self.pad_value), axis = 1) + 1
        return K.arange(K.shape(y_true)[1])[None] < lengths

    def update_state(self, y_true, y_pred, sample_weight = None):
        if isinstance(y_pred, (list, tuple)):   y_pred = y_pred[0]
        if len(K.shape(y_true)) == 3:           y_true = y_true[:, 0]
        mask = self.build_padding_mask(y_true)
        
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.divide_no_nan(
            K.sum(accuracy * K.cast(mask, accuracy.dtype), axis = 1),
            K.cast(K.count_nonzero(mask, axis = 1), accuracy.dtype)
        )
        
        self.samples.assign_add(K.cast(K.shape(y_true)[0], 'float32'))
        self.accuracy.assign_add(K.sum(accuracy))
        self.exact_match.assign_add(K.cast(K.count_nonzero(accuracy == 1), 'float32'))
    
    def result(self):
        return {
            'accuracy'      : self.accuracy / self.samples,
            'exact_match'   : self.exact_match / self.samples
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            'pad_value' : self.pad_value,
            'eos_value' : self.eos_value
        })
        return config