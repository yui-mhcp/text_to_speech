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

import math
import keras
import keras.ops as K

@keras.saving.register_keras_serializable('custom_activations')
class CustomActivation(keras.layers.Layer):
    def __init__(self, activation, ** kwargs):
        super().__init__(** kwargs)
        self.activation = activation
        
        self.activation_fn = get_activation(activation, return_layer = False)
        if not isinstance(activation, (list, tuple)):
            self.activation_fn = [self.activation_fn]
        
        self.supports_masking = True

    def call(self, inputs):
        out = inputs
        for fn in self.activation_fn: out = fn(out)
        return out
    
    def get_config(self):
        return {** super().get_config(), 'activation' : self.activation}

def l2_normalization(x, axis = -1):
    return K.divide_no_nan(x, K.norm(x, axis = axis, keepdims = True))

l2 = l2_norm = l2_normalize = l2_normalization

def glu(x, axis = -1):
    """ Gated Linear Unit activation function (equivalent to torch.nn.functional.glu) """
    a, b = K.split(x, 2, axis = axis)
    return a * K.sigmoid(b)

def gelu_new(x):
    """
    Gaussian Error Linear Unit. This is a smoother version of the GELU. Original paper: https://arxiv.org/abs/1606.0841
    Args:
        x: float Tensor to perform activation
    Returns:
        `x` with the GELU activation applied.
    """
    pi      = K.cast(math.pi, x.dtype)
    coeff   = K.cast(0.044715, x.dtype)
    cdf     = 0.5 * (1.0 + K.tanh(K.sqrt(2.0 / pi) * (x + coeff * x ** 3)))

    return x * cdf

def quick_gelu(x):
    return x * K.sigmoid(1.702 * x)

soft_gelu = gelu_new

_activations    = {k : v for k, v in globals().items() if callable(v)}

def get_activation(activation, *, return_layer = True, ** kwargs):
    if isinstance(activation, keras.layers.Layer): return activation

    if activation is None: activation = 'linear'
    if isinstance(activation, str):
        if activation == 'leaky': return keras.layers.LeakyReLU(** kwargs)
        if not return_layer:
            return keras.activations.deserialize(activation.lower(), custom_objects = _activations)
        return CustomActivation(activation)

    return [get_activation(act) for act in activation]
