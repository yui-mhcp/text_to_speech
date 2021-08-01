import tensorflow as tf

def log_softmax(x, epsilon = 1e-6):
    """ Log softmax (pytorch version) """
    return tf.math.log(tf.nn.softmax(x) + epsilon)

def gelu(x):
    """ Gaussian Error Linear Unit """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf

def soft_gelu(x):
    """ Smoother Gaussian Error Linear Unit"""
    cdf = 0.5 * (1.0 + f.tanh((tf.math.sqrt(2. / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def swish(x):
    """ Swish activation """
    return x * tf.sigmoid(x)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

LogSoftmax  = log_softmax
GeLU        = gelu
SoftGeLU    = soft_gelu
Swish       = swish
Mish        = mish

_activations = {    
    "log_softmax"   : LogSoftmax,
    "LogSoftmax"    : LogSoftmax,
    "gelu"          : GeLU,
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