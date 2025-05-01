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
class CTCLoss(keras.losses.Loss):
    def __init__(self, pad_value = 0, name = 'ctc_loss', ** kwargs):
        for k in ('eos_value', 'from_logits'): kwargs.pop(k, None)
        super().__init__(name = name, ** kwargs)
        self.pad_value  = pad_value
    
    def call(self, y_true, y_pred):
        true_length = K.count_nonzero(y_true == self.pad_value, axis = 1)
        pred_length = K.full(K.shape(target_length), K.shape(y_pred)[1], dtype = 'int32')
        
        return K.ctc_loss(
            y_true, y_pred, true_length, pred_length, mask_value = self.pad_value
        )
    
    def get_config(self):
        return {** super().get_config(), 'pad_value' : self.pad_value}
