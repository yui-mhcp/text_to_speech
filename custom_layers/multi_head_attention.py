
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

from loggers import timer
from hparams.hparams import HParams

HParamsMHA = HParams(
    use_bias    = True,
    num_heads   = 8,
    attention_dim   = 512,
    attention_drop_rate    = 0.,
    
    mask_factor = -1e4,
    
    use_output_layer    = True,
    output_dim  = None,
    residual    = True,
    drop_rate   = 0.1,
    normalize   = True,
    epsilon     = 1e-6,
    normalize_input = False,
    norm_training   = True
)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, name = None, ** kwargs):
        """
            Multi-Head Attention layer as described in the `Attention is All You Need` paper
            Note that I have added some additional logic that are used in classical Transformers architecture (such as the normalization and skip connection)
            
            Arguments (check HParamsMHA for an exhaustive list) :
                - attention_dim : the number of units (in total)
                - num_heads : the number of heads for the MHA
                    Note : attention_dim % num_heads == 0
                - use_bias  : whether to use bias for q / k / v weights (False in the `ClipCap` architecture)
                
                - use_output_layer  : whether to use the output layer or not
                - output_dim    : the number of units for the output layer (default to `attention_dim`)
                - residual  : whether to apply skip connection or not
                - normalize : whether to normalize the output or not
                - normalize_input   : whether to normalize the inputs or not
                    Note : if True and `residual = True`, the skip connection is applied with the un-normalized query (it is the behavior for the GPT-2 architecture)
        """
        super().__init__(name = name)
        self.hparams = HParamsMHA.extract(kwargs)
        
        self.residual       = self.hparams.residual
        self.mask_factor    = tf.cast(self.hparams.mask_factor, tf.float32)
        self.norm_training  = self.hparams.norm_training
        
        assert self.hparams.attention_dim % self.hparams.num_heads == 0, "Attention_dim % num_heads != 0 !"
        
        self.num_heads  = tf.cast(self.hparams.num_heads, tf.int32)
        self.attention_dim  = tf.cast(self.hparams.attention_dim, tf.int32)
        self.depth      = tf.cast(self.hparams.attention_dim // self.hparams.num_heads, tf.int32)
        
        self.sqrt_depth = tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        self.wq = tf.keras.layers.Dense(
            self.hparams.attention_dim, use_bias = self.hparams.use_bias, name = "query_layer"
        )
        self.wk = tf.keras.layers.Dense(
            self.hparams.attention_dim, use_bias = self.hparams.use_bias, name = "key_layer"
        )
        self.wv = tf.keras.layers.Dense(
            self.hparams.attention_dim, use_bias = self.hparams.use_bias, name = "value_layer"
        )
        
        self.attn_dropout   = tf.keras.layers.Dropout(
            self.hparams.attention_drop_rate
        ) if self.hparams.attention_drop_rate > 0. else None

        out_dim = self.hparams.output_dim if self.hparams.output_dim else self.hparams.attention_dim
        if self.residual and out_dim != self.hparams.attention_dim:
            raise ValueError('When `residual = True`, `attention_dim` must equal `output_dim`')
        self.output_layer   = tf.keras.layers.Dense(
            out_dim, name = 'output_layer'
        ) if self.hparams.use_output_layer else None
        self.dropout        = tf.keras.layers.Dropout(self.hparams.drop_rate)
        self.inp_norm_layer = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_input'
        ) if self.hparams.normalize_input else None
        self.norm_layer     = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_output'
        ) if self.hparams.normalize else None
    
    @timer
    def split_heads(self, x, batch_size):
        """
            Split the last dimension into (num_heads, depth)
            Transpose the result such that the shape is (batch, num_heads, seq_len, depth)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])
    
    @timer
    def merge_heads(self, scaled_attention, batch_size):
        """ Merge heads' output """
        # batch, seq_len_q, num_heads, depth
        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])
        
        # (batch, seq_len_q, d_model)
        return tf.reshape(scaled_attention, (batch_size, -1, self.attention_dim))
    
    @timer
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

        scaled_attention_logits = matmul_qk / self.sqrt_depth
        
        return self.compute_attention(scaled_attention_logits, v, mask = mask, training = training)

    @timer
    def compute_attention(self, attn_logits, v, mask = None, training = False):
        if mask is not None:
            attn_logits = attn_logits * (1. - mask) + mask * self.mask_factor

        # (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(attn_logits, axis = -1)

        if self.attn_dropout is not None:
            attention_weights = self.attn_dropout(attention_weights, training = training)
        # (..., seq_len_q, depth_v)
        output = tf.matmul(attention_weights, v) 

        return output, attention_weights
    
    @timer
    def process_qkv(self, query, key, value, training, normalize_kv, initial_state):
        batch_size = tf.shape(query)[0]
        
        q, k, v = query, key, value
        if self.inp_norm_layer is not None:
            q = self.inp_norm_layer(q, training = training and self.norm_training)
            if normalize_kv:
                k = self.inp_norm_layer(k, training = training and self.norm_training)
                v = self.inp_norm_layer(v, training = training and self.norm_training)

        q = self.wq(q)      # (batch_size, seq1_len, d_model)
        k = self.wk(k)      # (batch_size, seq2_len, d_model)
        v = self.wv(v)      # (batch_size, seq2_len, d_model)

        q = self.split_heads(q, batch_size)     # (batch, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)     # (batch, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)     # (batch, num_heads, seq_len_v, depth)
        
        if initial_state is not None:
            past_k, past_v = initial_state
            k = tf.concat([past_k, k], axis = -2)
            v = tf.concat([past_v, v], axis = -2)
        
        return q, k, v

    @timer(name = 'MHA call', log_if_root = False)
    def call(self,
             query,
             key,
             value,
             mask       = None,
             training   = False,
             initial_state  = None,
             return_attention   = True,
             return_state   = False,
             normalize_kv   = True,
             ** kwargs
            ):
        """
            Computes the logic for the Multi-Head Attention as described in the paper `Attention is All you Need`
            
            Arguments :
                - query         : the query with shape [batch_size, q_len, self.attention_dim]
                - key / value   : keys and values with shape [batch_size, k_len, self.attention_dim]
                    Note : in self-attention, query = keys = values
                - mask  :        the attention mask to apply
                - training      : whether it is a training pass or not
                - normalize_kv  : whether to normalize keys and values (if `self.normalize_input = True`)
                - initial_state : hidden states of the previous pass
                - return_{attention / state}    : whether to return the attention weights / hidden states
            Returns :
                - result    : the output of the layer
                - attention_weights (optional)  : the attention weights of shape [batch_size, self.num_heads, q_len, k_len]
                - hidden_states (optional)      : the hidden states of shape (key.shape, values.shape)
        """
        batch_size = tf.shape(query)[0]
        
        q, k, v = self.process_qkv(
            query, key, value, training, normalize_kv = normalize_kv, initial_state = initial_state
        )
        
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
    
    def get_output_shape(self, q, k, v, return_attention = True, return_state = False):
        _attn_shape = (q[0], self.num_heads.numpy(), q[1], k[1])
        _out_shape  = tuple(q[:-1]) + (self.attention_dim.numpy(), )
        out_shape = (_out_shape, )
        if return_state:        out_shape = out_shape + ((k, v), )
        if return_attention:    out_shape = out_shape + (_attn_shape, )
        return out_shape[0] if len(out_shape) == 1 else out_shape
    
    def get_config(self):
        config = super().get_config()
        return (self.hparams + config).get_config()
