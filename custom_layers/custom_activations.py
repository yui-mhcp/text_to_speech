
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

import math
import tensorflow as tf

class CustomActivation(tf.keras.layers.Layer):
    def __init__(self, activation, ** kwargs):
        super().__init__(** kwargs)
        self.activation = activation
        
        self.activation_fn  = get_activation(activation, return_layer = False)
    
    def call(self, inputs):
        return self.activation_fn(inputs)
    
    def get_config(self):
        return {** super().get_config(), 'activation' : self.activation}

def l2_norm(x, axis = -1):
    return tf.math.l2_normalize(x, axis = axis)

def log_softmax(x, axis = -1):
    return tf.math.log_softmax(x, axis = axis)

def gelu(x):
    """ Gaussian Error Linear Unit """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.math.sqrt(2.0), x.dtype)))
    return x * tf.cast(cdf, x.dtype)

def glu(x, axis = -1):
    """ Gated Linear Unit activation function (equivalent to torch.nn.functional.glu) """
    a, b = tf.split(x, 2, axis = axis)
    return a * tf.sigmoid(b)

def soft_gelu(x):
    """ Smoother Gaussian Error Linear Unit"""
    cdf = 0.5 * (1.0 + tf.tanh((tf.math.sqrt(2. / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def gelu_new(x):
    """
    Gaussian Error Linear Unit. This is a smoother version of the GELU. Original paper: https://arxiv.org/abs/1606.0841
    Args:
        x: float Tensor to perform activation
    Returns:
        `x` with the GELU activation applied.
    """
    pi      = tf.cast(math.pi, x.dtype)
    coeff   = tf.cast(0.044715, x.dtype)
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))

    return x * cdf

def quick_gelu(x):
    return x * tf.sigmoid(1.702 * x)

def swish(x):
    """ Swish activation """
    return x * tf.sigmoid(x)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

LogSoftmax  = log_softmax
GLU         = glu
GeLU        = gelu
SoftGeLU    = soft_gelu
QuickGeLU   = quick_gelu
GeLUNew     = gelu_new
Swish       = swish
Mish        = mish
L2Norm      = l2_norm

_activations = {
    "log_softmax"   : LogSoftmax,
    "LogSoftmax"    : LogSoftmax,
    "L2Norm"        : L2Norm,
    "l2_norm"       : L2Norm,
    'l2_normalize'  : L2Norm,
    "glu"           : GLU,
    "gelu"          : GeLU,
    "gelu_new"      : gelu_new,
    "smooth_gelu"   : SoftGeLU,
    "quick_gelu"    : quick_gelu,
    "swish"         : Swish,
    "mish"          : Mish
}

def get_activation(activation, return_layer = True, ** kwargs):
    if activation is None or isinstance(activation, tf.keras.layers.Layer): return activation
    elif isinstance(activation, str):
        if activation == 'leaky': return tf.keras.layers.LeakyReLU(** kwargs)
        activation_fn = _activations.get(activation, activation)
        if not isinstance(activation_fn, str) and not return_layer: return activation_fn
        return tf.keras.layers.Activation(activation_fn)
    else:
        return tf.keras.layers.Activation(activation)