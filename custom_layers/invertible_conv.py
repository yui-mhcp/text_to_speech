import tensorflow as tf

class Invertible1x1Conv(tf.keras.layers.Layer):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c, ** kwargs):
        super().__init__(** kwargs)
        self.c  = c
        
        self.conv = tf.keras.layers.Conv1D(
            filters     = c,
            kernel_size = 1,
            strides     = 1,
            padding     = 'same',
            use_bias    = False
        )
        
        self.built_inverse = False
        
    def build(self, input_shape):
        self.conv.build(input_shape)
        
        W = tf.transpose(tf.squeeze(self.conv.weights))

        if tf.linalg.det(W) < 0:
            self.init_random()
            self.build_inverse()
        else:
            self.build_inverse()

        super().build(input_shape)

    def init_random(self):
        # Sample a random orthonormal matrix to initialize weights
        W = tf.linalg.qr(tf.random.normal((self.c, self.c)))[0]

        # Ensure determinant is 1.0 not -1.0
        if tf.linalg.det(W) < 0:
            W = W.numpy()
            W[:,0] = -1*W[:,0]
        
        W = tf.reshape(W, [1, self.c, self.c])
        self.conv.set_weights([W])

    def build_inverse(self):
        W = tf.transpose(tf.squeeze(self.conv.weights))

        W_inverse = tf.transpose(tf.linalg.inv(W))
        self.W_inverse = tf.expand_dims(W_inverse, axis = 0)
        
        self.built_inverse = True
        
    def call(self, inputs, reverse = False):
        if reverse:
            return tf.nn.conv1d(inputs, self.W_inverse, stride = 1, padding = 'SAME')
        else:
            batch_size  = tf.cast(tf.shape(inputs)[0], tf.float32)
            group_size  = tf.cast(tf.shape(inputs)[2], tf.float32)
            n_of_groups = tf.cast(tf.shape(inputs)[1], tf.float32)
            
            W = tf.transpose(tf.squeeze(self.conv.weights))
            # Forward computation
            log_det_W = batch_size * n_of_groups * tf.math.log(tf.linalg.det(W))
            output = self.conv(inputs)
            return output, log_det_W

    def get_config(self):
        config = super().get_config()
        config['c'] = self.c
        
        return config