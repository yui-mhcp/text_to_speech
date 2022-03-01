
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

def log_softmax(x, epsilon = 1e-6, axis = -1):
    """ Log softmax (inspired from pytorch version) """
    return tf.math.log(tf.nn.softmax(x, axis = axis) + epsilon)

def gelu(x):
    """ Gaussian Error Linear Unit """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf

def soft_gelu(x):
    """ Smoother Gaussian Error Linear Unit"""
    cdf = 0.5 * (1.0 + f.tanh((tf.math.sqrt(2. / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def gelu_new(x):
    """
    Gaussian Error Linear Unit. This is a smoother version of the GELU. Original paper: https://arxiv.org/abs/1606.0841
    Args:
        x: float Tensor to perform activation
    Returns:
        `x` with the GELU activation applied.
    """
    pi      = tf.cast(math.pi, tf.float32)
    coeff   = tf.cast(0.044715, x.dtype)
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))

    return x * cdf

def swish(x):
    """ Swish activation """
    return x * tf.sigmoid(x)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

LogSoftmax  = log_softmax
GeLU        = gelu
SoftGeLU    = soft_gelu
GeLUNew     = gelu_new
Swish       = swish
Mish        = mish

_activations = {    
    "log_softmax"   : LogSoftmax,
    "LogSoftmax"    : LogSoftmax,
    "gelu"          : GeLU,
    "gelu_new"      : gelu_new,
    "smooth_gelu"   : SoftGeLU,
    "swish"         : Swish,
    "mish"          : Mish
}

def get_activation(activation, ** kwargs):
    if activation is None or isinstance(activation, tf.keras.layers.Layer): return activation
    elif isinstance(activation, str):
        if activation == 'leaky': return tf.keras.layers.LeakyReLU(** kwargs)
        return tf.keras.layers.Activation(_activations.get(activation, activation))
    else:
        return tf.keras.layers.Activation(activation)