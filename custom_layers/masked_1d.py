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

def _compute_new_len(seq_len, kernel_size, strides, padding, dilation_rate):
    if padding != 'same': seq_len = seq_len - kernel_size + 1
    new_len = K.maximum(1, K.cast(K.ceil(seq_len / strides), 'int32'))
    return new_len

def build_mask_1d(inputs, mask, kernel_size, strides, padding, dilation = 1):
    if mask is None: return None

    if strides == 1:
        if padding == 'same': return mask
        return mask[:, (kernel_size - 1) * dilation :]
    
    seq_len = K.count_nonzero(mask, axis = 1)
    new_len = _compute_new_len(seq_len, kernel_size, strides, padding, dilation)
    
    return K.arange(K.shape(inputs)[1])[None, :] < new_len[:, None]

@keras.saving.register_keras_serializable('masked_layers')
class MaskedMaxPooling1D(keras.layers.MaxPooling1D):
    def compute_mask(self, inputs, mask = None):
        return build_mask_1d(inputs, mask, self.pool_size[0], self.strides[0], self.padding)

    def call(self, inputs, mask = None):
        out = super().call(inputs)

        if mask is not None:
            mask = self.compute_mask(out, mask)
            out  = out * K.cast(mask, out.dtype)[:, :, None]
            try:
                out._keras_mask = mask
            except AttributeError:
                pass
        
        return out

@keras.saving.register_keras_serializable('masked_layers')
class MaskedAveragePooling1D(keras.layers.AveragePooling1D):
    def compute_mask(self, inputs, mask = None):
        return build_mask_1d(inputs, mask, self.pool_size[0], self.strides[0], self.padding)

    def call(self, inputs, mask = None):
        out = super().call(inputs)

        if mask is not None:
            mask = self.compute_mask(out, mask)
            out  = out * K.cast(mask, out.dtype)[:, :, None]
            try:
                out._keras_mask = mask
            except AttributeError:
                pass
        
        return out

@keras.saving.register_keras_serializable('masked_layers')
class MaskedZeroPadding1D(keras.layers.ZeroPadding1D):
    def compute_mask(self, inputs, mask = None):
        if mask is None: return None
        
        if len(mask.shape) == 1: mask = K.expand_dims(mask, axis = 0)
        return K.pad(mask, [(0, 0), (sum(self.padding), 0)], constant_values = True)
    
    def call(self, inputs, mask = None):
        out = super().call(inputs)
        
        if mask is not None:
            if len(mask.shape) == 1: mask = K.expand_dims(mask, axis = 0)
            out_mask = K.pad(mask, [(0, 0), self.padding], constant_values = True)
            out      = out * K.cast(out_mask[:, :, None], out.dtype)
        
        return out
    
@keras.saving.register_keras_serializable('masked_layers')
class MaskedConv1D(keras.layers.Conv1D):
    def compute_mask(self, inputs, mask = None, *, use_cached = True):
        if use_cached:
            mask = getattr(self, '_cached_mask', mask)
            if mask is not None: self._cached_mask = None
            return mask
        
        return build_mask_1d(
            inputs, mask, self.kernel_size[0], self.strides[0], self.padding, self.dilation_rate[0]
        )
    
    def call(self, inputs, mask = None):
        if mask is not None:
            inputs = inputs * K.cast(mask[:, :, None], inputs.dtype)
        
        out = super().call(inputs)

        if mask is not None:
            mask = self.compute_mask(out, mask, use_cached = False)
            out  = out * K.cast(mask[:, :, None], out.dtype)
            self._cached_mask = mask
            try:
                out._keras_mask = mask
            except AttributeError:
                pass
        
        return out

