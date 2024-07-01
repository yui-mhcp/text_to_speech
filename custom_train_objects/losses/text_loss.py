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
import keras.ops as K

@keras.saving.register_keras_serializable('custom_loss')
class TextLoss(keras.losses.Loss):
    def __init__(self,
                 pad_value  = 0,
                 eos_value  = -1,
                 warmup_tokens  = 0,
                 from_logits    = False,
                 ** kwargs
                ):
        super().__init__(** kwargs)
        
        self.pad_value  = pad_value
        self.eos_value  = eos_value
        self.from_logits    = from_logits
        self.warmup_tokens  = warmup_tokens
    
    def build_padding_mask(self, y_true):
        if self.eos_value != self.pad_value: return K.not_equal(y_true, self.pad_value)
        lengths = K.count_nonzero(K.not_equal(y_true, self.pad_value), axis = 1) + 1
        return K.arange(K.shape(y_true)[1])[None] < lengths
    
    def build_warmup_mask(self, length, dtype):
        warmup_idx  = K.minimum(length, K.convert_to_tensor(self.warmup_tokens + 1, length.dtype))
        warmups = K.arange(1, warmup_idx, dtype = dtype) / K.cast(warmup_idx, dtype)
        
        return K.concatenate([
            warmups, K.ones((K.maximum(0, length - self.warmup_tokens), ), dtype = dtype)
        ], axis = -1)[None]
    
    def call(self, y_true, y_pred, sample_weight = None, padding_mask = None):
        if isinstance(y_pred, (list, tuple)):   y_pred = y_pred[0]
        if len(K.shape(y_true)) == 3:           y_true = y_true[:, 0]
        mask = self.build_padding_mask(y_true)

        loss    = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits = self.from_logits
        )
        loss    = loss * K.cast(mask, loss.dtype)
        
        if self.warmup_tokens > 0:
            loss = loss * self.build_warmup_mask(K.shape(loss)[1], loss.dtype)
        
        return K.divide_no_nan(
            K.sum(loss, axis = 1), K.cast(K.count_nonzero(mask, axis = 1), loss.dtype)
        )
    
    def get_config(self):
        return {
            ** super().get_config(),
            'pad_value' : self.pad_value,
            'eos_value' : self.eos_value,
            'from_logits'   : self.from_logits,
            'warmup_tokens' : self.warmup_tokens
        }
