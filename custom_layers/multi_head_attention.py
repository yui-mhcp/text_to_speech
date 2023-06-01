
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
    num_heads   = -1,
    attention_dim   = -1,
    is_cross_attention  = None,
    
    use_bias    = True,
    query_bias  = None,
    key_bias    = None,
    value_bias  = None,

    mask_factor = -1e4,
    attention_drop_rate    = 0.,
    
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
    _attr_to_set    = ('residual', 'is_cross_attention', 'norm_training')
    
    def __init__(self, num_heads, attention_dim, name = None, ** kwargs):
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
        self.supports_masking   = True
        
        kwargs.update({'num_heads' : num_heads, 'attention_dim' : attention_dim,})
        self.hparams = HParamsMHA.extract(kwargs)
        
        for k in ('query_bias', 'key_bias', 'value_bias'):
            if self.hparams[k] is None: self.hparams[k] = self.hparams.use_bias
        
        for attr in self._attr_to_set:
            setattr(self, attr, self.hparams[attr])

        #self.is_cross_attention = tf.cast(self.is_cross_attention, tf.bool)
        self.mask_factor    = tf.cast(self.hparams.mask_factor, tf.float32)
        
        assert self.hparams.attention_dim % self.hparams.num_heads == 0, "Attention_dim % num_heads != 0 !"
        
        self.num_heads  = tf.cast(self.hparams.num_heads, tf.int32)
        self.attention_dim  = tf.cast(self.hparams.attention_dim, tf.int32)
        self.depth      = tf.cast(self.hparams.attention_dim // self.hparams.num_heads, tf.int32)
        
        self.sqrt_depth = tf.math.sqrt(tf.cast(self.depth, tf.float32))

        self.wq = tf.keras.layers.Dense(
            self.hparams.attention_dim, use_bias = self.hparams.query_bias, name = "query_layer"
        )
        self.wk = tf.keras.layers.Dense(
            self.hparams.attention_dim, use_bias = self.hparams.key_bias, name = "key_layer"
        )
        self.wv = tf.keras.layers.Dense(
            self.hparams.attention_dim, use_bias = self.hparams.value_bias, name = "value_layer"
        )
        
        self.softmax = tf.keras.layers.Softmax()
        
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
    
    @property
    def output_dim(self):
        if not self.hparams.use_output_layer: return self.hparams.attention_dim
        return self.hparams.output_dim if self.hparams.output_dim else self.hparams.attention_dim
    
    def get_initial_state(self, query, batch_size = None, dtype = tf.float32):
        if batch_size is None: batch_size = tf.shape(query)[0]
        return (
            tf.zeros((batch_size, self.num_heads, 0, self.depth), dtype = dtype),
            tf.zeros((batch_size, self.num_heads, 0, self.depth), dtype = dtype)
        )
    
    def initialize_cache(self, query, key, value):
        batch_size  = tf.shape(key)[0]
        if self.is_cross_attention:
            return (
                self.split_heads(self.wk(key), batch_size),   # (batch_size, n_heads, seq_len, depth)
                self.split_heads(self.wv(value), batch_size)  # (batch_size, n_heads, seq_len, depth)
            )
        elif tf.shape(query)[1] > 0:
            q, k, v = self.process_qkv(
                query[:, :-1], key[:, :-1], value[:, :-1], False, True, None
            )
            return (k, v)
        else:
            return self.get_initial_state(query, batch_size, dtype = query.dtype)

    #@timer
    def split_heads(self, x, batch_size):
        """
            Split the last dimension into (num_heads, depth)
            Transpose the result such that the shape is (batch, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, tf.shape(x)[1], self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])
    
    #@timer
    def merge_heads(self, scaled_attention, batch_size):
        """ Merge heads' output """
        # batch, seq_len_q, num_heads, depth
        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])
        
        # (batch, seq_len_q, d_model)
        return tf.reshape(
            scaled_attention, (batch_size, tf.shape(scaled_attention)[1], self.attention_dim)
        )
    
    #@timer
    def scaled_dot_product_attention(self, q, k, v, mask = None, training = False):
        """
            Attention(Q, K, T) = softmax(mask(Q @ K^t) / sqrt(d_k)) * V

            Arguments :
                - q : query shape == (..., seq_len_q, depth)
                - k : key shape == (..., seq_len_k, depth)
                - v : value shape == (..., seq_len_v, depth_v)
            Outputs : output, attention_weights
                - attention_weights shape == (..., seq_len_q, seq_len_k)
                - output shape == (..., seq_len_q, depth_v)
        """
        # shape = (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b = True)

        scaled_attention_logits = matmul_qk / tf.cast(self.sqrt_depth, matmul_qk.dtype)
        
        return self.compute_attention(scaled_attention_logits, v, mask = mask, training = training)

    #@timer
    def compute_attention(self, attn_logits, v, mask = None, training = False):
        if mask is not None:
            mask_val = attn_logits.dtype.min if self.mask_factor == -1. else tf.cast(
                self.mask_factor, attn_logits.dtype
            )
            attn_logits = tf.where(mask, attn_logits, mask_val)
        #    attn_logits = attn_logits * mask + (1. - mask) * mask_val

        # (..., seq_len_q, seq_len_k)
        attention_weights = self.softmax(attn_logits)

        if self.attn_dropout is not None:
            attention_weights = self.attn_dropout(attention_weights, training = training)
        # (..., seq_len_q, depth_v)
        output = tf.matmul(attention_weights, v) 

        return output, attention_weights
    
    #@timer
    def process_qkv(self, query, key, value, training, normalize_kv, initial_state):
        #if (self.is_cross_attention) and (key is None or value is None):
        #    raise RuntimeError('key / value are required for cross-attention !')

        use_state   = initial_state is not None and tf.shape(initial_state[0])[-2] > 0
        batch_size  = tf.shape(query)[0]

        if self.inp_norm_layer is not None:
            query = self.inp_norm_layer(query, training = training)
            
            if normalize_kv and key is not None and value is not None:
                key     = self.inp_norm_layer(key, training = training and self.norm_training)
                value   = self.inp_norm_layer(value, training = training and self.norm_training)
        # shapes = (batch_size, seq_len_{q / k / v}, attention_dim)
        q = self.wq(query)  # (batch_size, seq1_len, d_model)
        q = self.split_heads(q, batch_size) # (batch, num_heads, seq_len_q, depth)
        
        if not self.is_cross_attention or not use_state:
            k   = key if key is not None else query
            v   = value if value is not None else query
            
            k   = self.split_heads(self.wk(k), batch_size)  # (batch_size, n_heads, seq2_len, depth)
            v   = self.split_heads(self.wv(v), batch_size)  # (batch_size, n_heads, seq2_len, depth)
            
            if use_state:
                past_k, past_v = initial_state
                k = tf.concat([past_k, k], axis = -2)
                v = tf.concat([past_v, v], axis = -2)
        else:
            k, v = initial_state

        return q, k, v

    #@timer(name = 'MHA call', log_if_root = True)
    def call(self,
             query,
             key    = None,
             value  = None,
             
             mask       = None,
             training   = False,
             normalize_kv   = True,
             
             initial_state  = None,
             return_attention   = True,
             return_state   = False,

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
        _out_shape  = tuple(q[:-1]) + (self.output_dim, )
        _kv_shape   = (k[0], self.hparams.num_heads, k[1], self.depth.numpy())
        
        out_shape = (_out_shape, )
        if return_state:        out_shape = out_shape + ((_kv_shape, _kv_shape), )
        if return_attention:    out_shape = out_shape + (_attn_shape, )
        return out_shape[0] if len(out_shape) == 1 else out_shape
    
    def get_config(self):
        return (self.hparams + super().get_config()).get_config()
