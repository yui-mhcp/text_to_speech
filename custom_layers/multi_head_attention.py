
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
    normalize_input = False,
    norm_training   = True
)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams = HParamsMHA.extract(kwargs)
        
        self.residual       = self.hparams.residual
        self.mask_factor    = tf.cast(self.hparams.mask_factor, tf.float32)
        self.norm_training  = self.hparams.norm_training
        
        assert self.hparams.attention_dim % self.hparams.num_heads == 0, "Attention_dim % num_heads != 0 !"
        
        self.num_heads = self.hparams.num_heads
        self.attention_dim  = self.hparams.attention_dim
        self.depth = self.hparams.attention_dim // self.hparams.num_heads
        
        self.wq = tf.keras.layers.Dense(self.hparams.attention_dim, name = "query_layer")
        self.wk = tf.keras.layers.Dense(self.hparams.attention_dim, name = "key_layer")
        self.wv = tf.keras.layers.Dense(self.hparams.attention_dim, name = "value_layer")
        
        self.attn_dropout   = tf.keras.layers.Dropout(
            self.hparams.attention_drop_rate
        ) if self.hparams.attention_drop_rate > 0. else None

        self.output_layer   = tf.keras.layers.Dense(
            self.hparams.attention_dim
        ) if self.hparams.use_output_layer else None
        self.dropout        = tf.keras.layers.Dropout(self.hparams.drop_rate)
        self.inp_norm_layer = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'input_normalization'
        ) if self.hparams.normalize_input else None
        self.norm_layer     = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon
        ) if self.hparams.normalize else None
    
    def split_heads(self, x, batch_size):
        """
            Split the last dimension into (num_heads, depth)
            Transpose the result such that the shape is (batch, num_heads, seq_len, depth)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])
    
    def merge_heads(self, scaled_attention, batch_size):
        """ Merge heads' output """
        # batch, seq_len_q, num_heads, depth
        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])
        
        # (batch, seq_len_q, d_model)
        return tf.reshape(scaled_attention, (batch_size, -1, self.attention_dim))
    
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
            scaled_attention_logits = scaled_attention_logits * (1. - mask) + mask * self.mask_factor

        # (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)

        if self.attn_dropout is not None:
            attention_weights = self.attn_dropout(attention_weights, training = training)
        # (..., seq_len_q, depth_v)
        output = tf.matmul(attention_weights, v) 

        return output, attention_weights
    
    def call(self, query, key, value, mask = None, training = False, initial_state = None,
             return_attention = True, return_state = False, normalize_kv = True):
        batch_size = tf.shape(query)[0]
        
        q, k, v = query, key, value
        if self.inp_norm_layer is not None:
            q = self.inp_norm_layer(q, training = training and self.norm_training)
            if normalize_kv:
                k = self.inp_norm_layer(k, training = training and self.norm_training)
                v = self.inp_norm_layer(v, training = training and self.norm_training)

        q = self.wq(q)      # (batch_size, seq_len, d_model)
        k = self.wk(k)        # (batch_size, seq_len, d_model)
        v = self.wv(v)      # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)     # (batch, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)     # (batch, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)     # (batch, num_heads, seq_len_v, depth)
        
        if initial_state is not None:
            past_k, past_v = initial_state
            k = tf.concat([past_k, k], axis = -2)
            v = tf.concat([past_v, v], axis = -2)

        # scaled_attention shape == (atch, num_heads, seq_len_q, depth)
        # attention_weights shape == (batch, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask = mask, training = training
        )

        output = self.merge_heads(scaled_attention, batch_size)
        
        if self.output_layer is not None:
            output = self.output_layer(output)
            
            if self.dropout is not None:    output = self.dropout(output, training = training)
            if self.residual:       output = output + query
            if self.norm_layer is not None:
                output = self.norm_layer(output, training = training and self.norm_training)
        
        output = (output, )
        if return_state:        output = output + ((k, v), )
        if return_attention:    output = output + (attention_weights, )
        return output[0] if len(output) == 1 else output
    
    def get_config(self):
        config = super().get_config()
        return (self.hparams + config).get_config()
