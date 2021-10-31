import tensorflow as tf

from hparams.hparams import HParams

HParamsMHA = HParams(
    num_heads   = 8,
    attention_dim   = 512,
    attention_drop_rate    = 0.,
    
    mask_factor = -1e4,
    
    use_output_layer    = True,
    residual    = True,
    drop_rate   = 0.1,
    normalize   = True,
    epsilon     = 1e-6,
    norm_training   = True
)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams = HParamsMHA.extract(kwargs)
        
        assert self.hparams.attention_dim % self.hparams.num_heads == 0, "Attention_dim % num_heads != 0 !"
        
        self.num_heads = self.hparams.num_heads
        self.attention_dim  = self.hparams.attention_dim
        self.depth = self.hparams.attention_dim // self.hparams.num_heads
        
        self.wq = tf.keras.layers.Dense(self.hparams.attention_dim, name = "query_layer")
        self.wk = tf.keras.layers.Dense(self.hparams.attention_dim, name = "key_layer")
        self.wv = tf.keras.layers.Dense(self.hparams.attention_dim, name = "value_layer")
        
        self.attn_dropout   = tf.keras.layers.Dropout(self.hparams.attention_drop_rate) if self.hparams.attention_drop_rate > 0. else None

        self.output_layer   = tf.keras.layers.Dense(self.hparams.attention_dim) if self.hparams.use_output_layer else None
        self.dropout        = tf.keras.layers.Dropout(self.hparams.drop_rate)
        self.norm_layer     = tf.keras.layers.LayerNormalization(epsilon = self.hparams.epsilon) if self.hparams.normalize else None
        
    def split_heads(self, x, batch_size):
        """
            Split the last dimension into (num_heads, depth)
            Transpose the result such that the shape is (batch, num_heads, seq_len, depth)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask = None, training = False):
        """
            Attention(Q, K, T) = softmax(Q @ K^t / sqrt(d_k)) * V

            Arguments :
                - q : query shape == (..., seq_len_q, depth)
                - k : key shape == (..., seq_len_k, depth)
                - v : value shape == (..., seq_len_v, depth_v)
            Outputs : output, attention_weights
                - attention_weights shape == (..., seq_len_q, seq_len_k)
                - output shape == (..., seq_len_q, depth_v)
        """
        matmul_qk = tf.matmul(q, k, transpose_b = True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * self.hparams.mask_factor)

        # (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)

        if self.attn_dropout is not None:
            attention_weights = self.attn_dropout(attention_weights, training = training)
        # (..., seq_len_q, depth_v)
        output = tf.matmul(attention_weights, v) 

        return output, attention_weights
    
    def call(self, query, key, value, mask = None, training = False, return_attention = True):        
        batch_size = tf.shape(query)[0]
        
        q = self.wq(query)      # (batch_size, seq_len, d_model)
        k = self.wk(key)        # (batch_size, seq_len, d_model)
        v = self.wv(value)      # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)     # (batch, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)     # (batch, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)     # (batch, num_heads, seq_len_v, depth)
                
        # scaled_attention shape == (atch, num_heads, seq_len_q, depth)
        # attention_weights shape == (batch, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask, training)
        
        # batch, seq_len_q, num_heads, depth
        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])
        
        # (batch, seq_len_q, d_model)
        output = tf.reshape(scaled_attention, (batch_size, -1, self.attention_dim))
        
        if self.output_layer is not None:
            output = self.output_layer(output)
            
            if self.dropout is not None:    output = self.dropout(output, training = training)
            if self.hparams.residual:       output = output + query
            if self.norm_layer is not None:
                output = self.norm_layer(output, training = training and self.hparams.norm_training)
        
        return output, attention_weights if return_attention else output
    
    def get_config(self):
        config = super().get_config()
        return self.hparams + config
