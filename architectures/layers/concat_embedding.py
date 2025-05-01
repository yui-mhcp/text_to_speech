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

import enum
import keras
import keras.ops as K

class ConcatMode(enum.IntEnum):
    CONCAT  = 0
    ADD     = 1
    SUB     = 2
    MUL     = 3
    DIV     = 4

@keras.saving.register_keras_serializable('custom_layers')
class ConcatEmbedding(keras.layers.Layer):
    """ Concat (a batch of) embedding vector to a sequence of embeddings """

    def __init__(self, concat_mode = 'concat', ** kwargs):
        super().__init__(** kwargs)
        
        if isinstance(concat_mode, str): concat_mode = ConcatMode[concat_mode.upper()]
        self.concat_mode    = concat_mode

    def _concat(self, sequence, embeddings):
        if len(K.shape(embeddings)) == 2: embeddings = K.expand_dims(embeddings, axis = 1)
        if self.concat_mode == ConcatMode.CONCAT:
            embeddings = K.tile(embeddings, [1, K.shape(sequence)[1], 1])

            return K.concatenate([sequence, embeddings], axis = -1)
        elif self.concat_mode == ConcatMode.ADD:
            return sequence + embeddings
        elif self.concat_mode == ConcatMode.SUB:
            return sequence - embeddings
        elif self.concat_mode == ConcatMode.MUL:
            return sequence * embeddings
        elif self.concat_mode == ConcatMode.DIV:
            return K.divide_no_nan(sequence, embeddings)

    def build(self, input_shape):
        super().build(input_shape)
        
    def compute_mask(self, inputs, mask = None):
        if mask is None: return None
        return mask if not isinstance(mask, (list, tuple)) else mask[0]
    
    def call(self, inputs, mask = None):
        sequence, embeddings = inputs
        
        out = self._concat(sequence, embeddings)
        
        if isinstance(mask, (list, tuple)): mask = mask[0]
        if mask is not None:
            out = K.where(K.expand_dims(mask, axis = -1), out, 0.)
            try:
                out._keras_mask = mask
            except AttributeError:
                pass
        
        return out

    def compute_output_shape(self, input_shape):
        if self.concat_mode != ConcatMode.CONCAT: return input_shape[0]
        seq_shape, emb_shape = input_shape
        
        return seq_shape[:-1] + (seq_shape[-1] + emb_shape[-1], )
    
    def get_config(self):
        return {** super().get_config(), 'concat_mode' : self.concat_mode}
    
