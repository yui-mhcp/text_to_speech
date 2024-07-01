# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
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

from loggers import timer
from .residual_multi_head_attention import ResidualMultiHeadAttention

@keras.saving.register_keras_serializable('custom_layers')
class RotaryMultiHeadAttention(ResidualMultiHeadAttention):
    default_params  = ResidualMultiHeadAttention.default_params(base = 10000)
    _attr_to_set    = ResidualMultiHeadAttention._attr_to_set + ['base']
    
    def __init__(self, * args, ** kwargs):
        super().__init__(* args, ** kwargs)
        
        self.inv_freq = 1. / self.base ** (
            K.arange(0, self.depth, 2, dtype = 'float32') / self.depth
        )

    def get_rotary_embedding(self, seq_len, offset  = 0, dtype = 'float32'):
        return get_rotary_embedding(self.inv_freq, seq_len, offset, dtype)
    
    def apply_rotary_embedding(self, q, k, lengths = None, initial_state = None, sin = None, cos = None):
        if sin is None or cos is None:
            assert sin is not None
            offset = (lengths - 1) if initial_state else None
            
            sin, cos = self.get_rotary_embedding(K.shape(k)[-2], offset, q.dtype)
        
        q   = (q * cos) + (rotate_half(q) * sin)
        k   = (k * cos) + (rotate_half(k) * sin)
        return q, k
    
    @timer
    def process_qkv(self,
                    query,
                    key,
                    value,
                    training,
                    batch_size,
                    normalize_kv,
                    initial_state,
                    ** kwargs
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
            
            q, k = self.apply_rotary_embedding(q, k, initial_state = initial_state, ** kwargs)

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

def get_rotary_embedding(inv_freq, seq_len, offset = None, dtype = 'float32'):
    t = K.arange(seq_len, dtype = 'float32')[None]
    if offset is not None: t = t + K.cast(offset, 'float32')[:, None]

    freqs   = K.matmul(K.transpose(t, [1, 0]), inv_freq[None, :])
    emb     = K.concatenate([freqs, freqs], axis = 1)[None, None]
    return K.cast(K.sin(emb), dtype), K.cast(K.cos(emb), dtype)

def rotate_half(x):
    return K.concatenate([
        - x[..., K.shape(x)[-1] // 2 :],
        x[..., : K.shape(x)[-1] // 2]
    ], axis = -1)
