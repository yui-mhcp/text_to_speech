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

@keras.saving.register_keras_serializable('custom_layers')
class CustomEmbedding(keras.layers.Embedding):
    def __init__(self, * args, mask_value = None, ** kwargs):
        super().__init__(* args, ** kwargs)
        
        if self.mask_zero: mask_value = 0
        self.mask_value         = None if mask_value is None else K.array(mask_value, 'int32')
        self.supports_masking   = mask_value is not None

    def change_vocabulary(self, new_vocab, old_vocab = None, ** kwargs):
        old_embeddings = self.embeddings.numpy()
        
        self.input_dim = len(new_vocab)
        self.build((None, None))
        
        if old_vocab:
            new_embeddings = old_embeddings[[
                old_vocab.index(v) if v in old_vocab else i
                for i, v in enumerate(new_vocab)
            ]]
            self.set_weights([new_embeddings])

    def compute_mask(self, inputs, mask = None):
        if mask is not None or self.mask_value is None: return mask
        
        return K.not_equal(inputs, self.mask_value)

    def call(self, inputs):
        return K.take(self.embeddings, K.cast(inputs, 'int32'), axis = 0)
    
    def get_config(self):
        return {** super().get_config(), 'mask_value' : self.mask_value}
