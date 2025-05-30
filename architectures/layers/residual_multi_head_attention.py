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

import keras
import keras.ops as K

from ..hparams import HParams

HParamsMHA = HParams(
    num_heads   = -1,
    attention_dim   = -1,
    
    scale   = True,
    multi_query = False,
    num_kv_heads    = -1,
    is_cross_attention  = None,
    
    use_bias    = True,
    query_bias  = None,
    key_bias    = None,
    value_bias  = None,

    drop_rate   = 0.1,
    mask_factor = -1e4,
    attention_drop_rate    = 0.,
    
    use_output_layer    = True,
    output_dim  = None,
    output_bias = True,
    
    residual    = True,
    
    normalize   = True,
    epsilon     = 1e-6,
    normalize_input = False
)

@keras.saving.register_keras_serializable('custom_layers')
class ResidualMultiHeadAttention(keras.layers.Layer):
    default_params  = HParamsMHA
    _attr_to_set    = [
        'attention_dim', 'num_heads', 'scale', 'residual', 'multi_query', 'is_cross_attention'
    ]
    
    def __init__(self, num_heads, attention_dim, *, norm_class = None, name = None, ** kwargs):
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
        if norm_class is None: norm_class = keras.layers.LayerNormalization
        
        super().__init__(name = name)
        self.supports_masking   = True
        
        kwargs.update({'num_heads' : num_heads, 'attention_dim' : attention_dim,})
        self.hparams = self.default_params.extract(kwargs)
        
        for k in ('query_bias', 'key_bias', 'value_bias', 'output_bias'):
            if self.hparams[k] is None: self.hparams[k] = self.hparams.use_bias
        
        for attr in self._attr_to_set: setattr(self, attr, self.hparams[attr])

        if self.hparams.mask_factor == -1.:
            self.mask_factor    = float('-inf')
        else:
            self.mask_factor    = float(self.hparams.mask_factor)
        
        assert self.hparams.attention_dim % self.hparams.num_heads == 0, "Attention_dim ({}) % num_heads ({}) != 0 !".format(self.hparams.attention_dim, self.hparams.num_heads)
        
        self.depth  = self.attention_dim // self.num_heads
        
        if self.hparams.multi_query:
            self.kv_heads   = 1
        elif self.hparams.num_kv_heads != -1:
            self.kv_heads   = self.hparams.num_kv_heads
        else:
            self.kv_heads   = self.num_heads
        self.kv_groups  = self.num_heads // self.kv_heads

        if self.residual and self.output_dim != self.hparams.attention_dim:
            raise ValueError('When `residual = True`, `attention_dim` must equal `output_dim`')

        self.sqrt_depth = K.sqrt(K.cast(self.depth, 'float32')) if self.scale else None

        kv_dim  = self.depth * self.kv_heads
        
        self.wq = keras.layers.Dense(
            self.attention_dim, use_bias = self.hparams.query_bias, name = "query_layer"
        )
        self.wk = keras.layers.Dense(
            kv_dim, use_bias = self.hparams.key_bias, name = "key_layer"
        )
        self.wv = keras.layers.Dense(
            kv_dim, use_bias = self.hparams.value_bias, name = "value_layer"
        )
        
        self.softmax    = keras.layers.Softmax()
        
        self.attn_dropout   = keras.layers.Dropout(
            self.hparams.attention_drop_rate
        ) if self.hparams.attention_drop_rate > 0. else None
        
        self.output_layer   = keras.layers.Dense(
            self.output_dim, use_bias = self.hparams.output_bias, name = 'output_layer'
        ) if self.hparams.use_output_layer else None
        self.dropout        = keras.layers.Dropout(self.hparams.drop_rate)
        self.inp_norm_layer = norm_class(
            epsilon = self.hparams.epsilon, name = 'norm_input'
        ) if self.hparams.normalize_input else None
        self.norm_layer     = norm_class(
            epsilon = self.hparams.epsilon, name = 'norm_output'
        ) if self.hparams.normalize else None
    
    def build(self, input_shape):
        super().build(input_shape)
        if self.inp_norm_layer is not None: self.inp_norm_layer.build(input_shape)
        self.wq.build(input_shape)
        self.wk.build(input_shape)
        self.wv.build(input_shape)
        if self.output_layer is not None: self.output_layer.build(input_shape)
        if self.norm_layer is not None: self.norm_layer.build((None, None, self.output_dim))

    @property
    def output_dim(self):
        if not self.hparams.use_output_layer: return self.hparams.attention_dim
        return self.hparams.output_dim if self.hparams.output_dim else self.hparams.attention_dim
    
    def split_heads(self, x, batch_size, num_heads):
        """
            Split the last dimension into (num_heads, depth)
            Transpose the result such that the shape is (batch, num_heads, seq_len, depth)
        """
        x = K.reshape(x, (batch_size, K.shape(x)[1], num_heads, self.depth))
        return K.transpose(x, [0, 2, 1, 3])
    
    def merge_heads(self, scaled_attention, batch_size):
        """ Merge heads' output """
        # batch, seq_len_q, num_heads, depth
        scaled_attention = K.transpose(scaled_attention, [0, 2, 1, 3])
        
        # (batch, seq_len_q, d_model)
        return K.reshape(
            scaled_attention, (batch_size, K.shape(scaled_attention)[1], self.attention_dim)
        )
    
    def scaled_dot_product_attention(self, q, k, v, mask = None, training = False, ** kwargs):
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
        if self.kv_groups > 1 and self.kv_heads > 1:
            k = K.repeat(k, self.kv_groups, axis = 1)
            v = K.repeat(v, self.kv_groups, axis = 1)
            
        # shape = (..., seq_len_q, seq_len_k)
        #attn_logits = K.matmul(q, K.transpose(k, [0, 1, 3, 2]))
        attn_logits = K.einsum('bhij,bhkj->bhik', q, k)

        if self.scale:
            attn_logits = attn_logits / K.cast(self.sqrt_depth, attn_logits.dtype)
        
        return self.compute_attention(
            attn_logits, v, mask = mask, training = training, ** kwargs
        )

    def compute_attention(self, attn_logits, v, mask = None, training = False, ** _):
        if mask is not None:
            attn_logits = K.where(
                mask, attn_logits, K.convert_to_tensor(self.mask_factor, attn_logits.dtype)
            )
        # (..., seq_len_q, seq_len_k)
        attention_weights = K.softmax(attn_logits, axis = -1)
        
        if self.attn_dropout is not None:
            attention_weights = self.attn_dropout(attention_weights, training = training)
        
        # (..., seq_len_q, depth_v)
        output = K.matmul(attention_weights, v) 

        return output, attention_weights
    
    def process_qkv(self,
                    query,
                    key,
                    value,
                    training,
                    batch_size,
                    normalize_kv,
                    initial_state,
                    ** _
                   ):
        if self.inp_norm_layer is not None:
            query = self.inp_norm_layer(query, training = training)
            
            if normalize_kv and key is not None and value is not None:
                key     = self.inp_norm_layer(key, training = training)
                value   = self.inp_norm_layer(value, training = training)
        # shapes = (batch_size, seq_len_{q / k / v}, attention_dim)
        q = self.wq(query)  # (batch_size, seq1_len, d_model)
        q = self.split_heads(q, batch_size, self.num_heads) # (batch, num_heads, seq_len_q, depth)
        
        if not self.is_cross_attention:
            k   = key if key is not None else query
            v   = value if value is not None else query
            
            k   = self.split_heads(self.wk(k), batch_size, self.kv_heads)
            v   = self.split_heads(self.wv(v), batch_size, self.kv_heads)
            
            if initial_state:
                past_k, past_v = initial_state
                k = K.concatenate([past_k, k], axis = -2)
                v = K.concatenate([past_v, v], axis = -2)
        elif not initial_state:
            k   = self.split_heads(self.wk(key), batch_size, self.kv_heads)
            v   = self.split_heads(self.wv(value), batch_size, self.kv_heads)
        else:
            k, v = initial_state

        return q, k, v

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
        batch_size = K.shape(query)[0]
        
        q, k, v = self.process_qkv(
            query,
            key,
            value,
            training    = training,
            batch_size  = batch_size,
            normalize_kv    = normalize_kv,
            initial_state   = initial_state,
            ** kwargs
        )
        
        # scaled_attention shape == (atch, num_heads, seq_len_q, depth)
        # attention_weights shape == (batch, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask = mask, training = training, ** kwargs
        )

        output = self.merge_heads(scaled_attention, batch_size)

        if self.output_layer is not None:
            output = self.output_layer(output)
            
            if self.dropout is not None:
                output = self.dropout(output, training = training)
            if self.residual:
                output = output + query
            if self.norm_layer is not None:
                output = self.norm_layer(output, training = training)
        
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
