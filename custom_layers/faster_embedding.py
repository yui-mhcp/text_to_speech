
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

class FasterEmbedding(tf.keras.layers.Embedding):
    """Faster version of embedding."""

    def __init__(self, * args, mask_value = None, ** kwargs):
        super().__init__(* args, ** kwargs)
        
        if self.mask_zero: mask_value = 0
        self.mask_value         = mask_value
        self.supports_masking   = mask_value is not None

    def change_vocabulary(self, new_vocab, ** kwargs):
        self.input_dim = len(new_vocab)
        self.build((None, None))

    def compute_mask(self, inputs, mask = None):
        if mask is not None or self.mask_value is None: return mask
        
        return tf.math.not_equal(inputs, self.mask_value)

    def call(self, inputs):
        inputs  = tf.cast(tf.expand_dims(inputs, -1), tf.int32)
        outputs = tf.gather_nd(self.embeddings, inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config['mask_value'] = self.mask_value
        return config