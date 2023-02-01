
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

def build_mask_1d(inputs, mask, kernel_size, strides, padding, dilation = 1):
    if mask is None: return None

    #tf.print(tf.reduce_sum(tf.cast(mask, tf.int32), -1), kernel_size, strides, padding)
    if strides == 1:
        if padding == 'same': return mask
        return mask[:, (kernel_size - 1) * dilation :]
    
    seq_len = tf.reduce_sum(tf.cast(mask, tf.int32), axis = -1)
    if padding != 'same': seq_len = seq_len - kernel_size + 1
    new_len = tf.maximum(1, tf.cast(tf.math.ceil(seq_len / strides), tf.int32))
    
    #tf.print('New seq length : {} - inputs : {}'.format(new_len, tf.shape(inputs)))
    
    return tf.sequence_mask(new_len, tf.shape(inputs)[1], dtype = tf.bool)

class MaskedMaxPooling1D(tf.keras.layers.MaxPooling1D):
    def compute_mask(self, inputs, mask = None):
        return build_mask_1d(inputs, mask, self.pool_size[0], self.strides[0], self.padding)

    def call(self, inputs, mask = None):
        out = super().call(inputs)

        if mask is not None:
            out = out * tf.expand_dims(tf.cast(self.compute_mask(out, mask), out.dtype), axis = -1)
        
        return out

class MaskedAveragePooling1D(tf.keras.layers.AveragePooling1D):
    def compute_mask(self, inputs, mask = None):
        return build_mask_1d(inputs, mask, self.pool_size[0], self.strides[0], self.padding)

    def call(self, inputs, mask = None):
        out = super().call(inputs)

        if mask is not None:
            out = out * tf.expand_dims(tf.cast(self.compute_mask(out, mask), out.dtype), axis = -1)
        
        return out

class MaskedZeroPadding1D(tf.keras.layers.ZeroPadding1D):
    def compute_mask(self, inputs, mask = None):
        if mask is None: return None
        
        return tf.pad(mask, [(0, 0), (sum(self.padding), 0)], constant_values = True)
    
    def call(self, inputs, mask = None):
        out = super().call(inputs)
        
        if mask is not None:
            out_mask = tf.pad(mask, [(0, 0), self.padding], constant_values = True)
            out      = tf.where(tf.expand_dims(out_mask, axis = -1), out, 0.)
        
        return out
    
class MaskedConv1D(tf.keras.layers.Conv1D):
    def compute_mask(self, inputs, mask = None):
        return build_mask_1d(
            inputs, mask, self.kernel_size[0], self.strides[0], self.padding, self.dilation_rate[0]
        )
    
    def call(self, inputs, mask = None):
        if mask is not None:
            inputs = tf.where(tf.expand_dims(mask, axis = -1), inputs, 0.)
        
        out = super().call(inputs)

        if mask is not None:
            out_mask = self.compute_mask(out, mask)
            out = tf.where(tf.expand_dims(out_mask, axis = -1), out, 0.)
            out._keras_mask = out_mask
        
        return out

