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
class RMSLayerNormalization(keras.layers.Layer):
    """ Normalization layer used in recent LLM architectures, such as `LLaMA`, `Mistral`, etc. """
    def __init__(self, epsilon = 1e-6, ** kwargs):
        super().__init__(** kwargs)
        self.supports_masking   = True
        self.epsilon    = epsilon
    
    def build(self, input_shape):
        super().build(input_shape)
        self.weight = self.add_weight(
            shape = (input_shape[-1], ), initializer = 'ones', name = 'weight'
        )
    
    def call(self, inputs):
        dtype = inputs.dtype
        
        inputs      = K.cast(inputs, 'float32')
        variances   = K.rsqrt(K.mean(K.square(inputs), axis = -1, keepdims = True) + self.epsilon)
        output      = K.cast(inputs * variances, dtype)
        return output * self.weight 
    
    def get_config(self):
        return {** super().get_config(), 'epsilon' : self.epsilon}
