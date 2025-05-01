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
class Invertible1x1Conv(keras.layers.Layer):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c, ** kwargs):
        super().__init__(** kwargs)
        self.c  = c
        
        self.conv = keras.layers.Conv1D(
            filters     = c,
            kernel_size = 1,
            strides     = 1,
            padding     = 'same',
            use_bias    = False,
            name    = 'conv'
        )
    
    def build(self, input_shape):
        super().build(input_shape)
        self.conv.build(input_shape)
        self.conv._load_own_variables = self.conv.load_own_variables
        self.conv.load_own_variables = lambda store: self.build_inverse(store)

    def build_inverse(self, store = None):
        if store is not None: self.conv._load_own_variables(store)
        kernel = self.conv.kernel
        W = K.transpose(K.squeeze(kernel))

        W_inverse = K.transpose(K.inv(W))
        self.W_inverse = K.expand_dims(W_inverse, axis = 0)

    def call(self, inputs, reverse = False):
        if reverse:
            return K.conv(inputs, self.W_inverse, padding = 'same')
        else:
            batch_size  = K.cast(K.shape(inputs)[0], 'float32')
            group_size  = K.cast(K.shape(inputs)[2], 'float32')
            n_of_groups = K.cast(K.shape(inputs)[1], 'float32')
            
            W = K.transpose(K.squeeze(self.conv.weights))
            # Forward computation
            log_det_W = batch_size * n_of_groups * K.log(K.det(W))
            output = self.conv(inputs)
            return output, log_det_W

    def get_config(self):
        return {** super().get_config(), 'c' : self.c}
