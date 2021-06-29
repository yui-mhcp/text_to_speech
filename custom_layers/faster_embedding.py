import tensorflow as tf

class FasterEmbedding(tf.keras.layers.Embedding):
    """Faster version of embedding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        inputs = tf.cast(tf.expand_dims(inputs, -1), tf.int32)
        outputs = tf.gather_nd(self.embeddings, inputs)
        return outputs
