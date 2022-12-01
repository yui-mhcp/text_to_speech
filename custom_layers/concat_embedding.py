
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

import enum
import tensorflow as tf

class ConcatMode(enum.IntEnum):
    CONCAT  = 0
    ADD     = 1
    SUB     = 2
    MUL     = 3
    DIV     = 4

class ConcatEmbedding(tf.keras.layers.Layer):
    """ Concat (a batch of) embedding vector to a sequence of embeddings """

    def __init__(self, concat_mode = 'concat', ** kwargs):
        super().__init__(** kwargs)
        
        from utils.generic_utils import get_enum_item
        
        self.concat_mode        = get_enum_item(concat_mode, ConcatMode)
        self.supports_masking   = True

    def _concat(self, sequence, embeddings):
        if len(tf.shape(embeddings)) == 2: embeddings = tf.expand_dims(embeddings, axis = 1)
        if self.concat_mode == ConcatMode.CONCAT:
            embeddings = tf.tile(embeddings, [1, tf.shape(sequence)[1], 1])

            return tf.concat([sequence, embeddings], axis = -1)
        elif self.concat_mode == ConcatMode.ADD:
            return sequence + embeddings
        elif self.concat_mode == ConcatMode.SUB:
            return sequence - embeddings
        elif self.concat_mode == ConcatMode.MUL:
            return sequence * embeddings
        elif self.concat_mode == ConcatMode.DIV:
            return sequence / embeddings

    def compute_mask(self, inputs, mask = None):
        if mask is None: return None
        return mask if not isinstance(mask, (list, tuple)) else mask[0]
    
    def call(self, inputs, mask = None):
        sequence, embeddings = inputs
        
        out = self._concat(sequence, embeddings)
        
        if isinstance(mask, (list, tuple)): mask = mask[0]
        if mask is not None:
            out = tf.where(tf.expand_dims(mask, axis = -1), out, 0.)
        
        return out

    def get_output_shape(self, input_shape):
        if self.concat_mode != ConcatMode.CONCAT: return input_shape[0]
        seq_shape, emb_shape = input_shape
        
        return seq_shape[:-1] + (seq_shape[-1] + emb_shape[-1], )
    
    def get_config(self):
        config = super().get_config()
        config['concat_mode'] = self.concat_mode
        return config