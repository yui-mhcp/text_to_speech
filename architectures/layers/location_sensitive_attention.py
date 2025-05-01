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

HParamsLSA = HParams(
    attention_dim   = 128,
    attention_filters   = 32,
    attention_kernel_size   = 31,
    probability_function    = 'softmax',
    concat_mode     = 2,
    cumulative      = True
)

@keras.saving.register_keras_serializable('custom_layers')
def LocationLayer(input_channels, attention_dim, filters, kernel_size, name = 'location_layer'):
    return keras.Sequential([
        keras.layers.Input(shape = (None, input_channels)),
        keras.layers.Conv1D(
            filters     = filters,
            kernel_size = kernel_size, 
            use_bias    = False,
            strides     = 1,
            padding     = "same",
            name = "location_conv"
        ),
        keras.layers.Dense(
            attention_dim, use_bias = False, name = "location_dense"
        )
    ], name = name)

@keras.saving.register_keras_serializable('tacotron2')
class LocationSensitiveAttention(keras.layers.Layer):
    def __init__(self, name = 'location_sensitive_attention', ** kwargs):
        super().__init__(name = name)
        self.hparams = HParamsLSA.extract(kwargs)
        assert self.hparams.concat_mode in (0, 1, 2)
        assert self.hparams.cumulative or self.hparams.concat_mode == 0
        
        self.attention_dim = self.hparams.attention_dim
        
        self.query_layer    = keras.layers.Dense(
            self.hparams.attention_dim, use_bias = False, name = "query_layer"
        )
        self.memory_layer   = keras.layers.Dense(
            self.hparams.attention_dim, use_bias = False, name = "memory_layer"
        )
        self.value_layer    = keras.layers.Dense(1, use_bias = False, name = "value_layer")
    
    def build(self, input_shape):
        super().build(input_shape)
        query_shape, memory_shape = input_shape
        
        self.memory_layer.build(memory_shape)
        self.query_layer.build(query_shape)
        
        self.location_layer = LocationLayer(
            input_channels  = 2 if self.hparams.concat_mode == 2 else 1,
            attention_dim   = self.hparams.attention_dim, 
            filters         = self.hparams.attention_filters, 
            kernel_size     = self.hparams.attention_kernel_size, 
            name = "location_layer"
        )
        self.value_layer.build((None, None, self.attention_dim))
    
    @property
    def state_size(self):
        return ((None, None), (None, None))
    
    @property
    def output_size(self):
        return self.attention_dim
    
    def get_initial_context(self, memory, batch_size = None):
        if batch_size is None: batch_size = K.shape(memory)[0]
        return K.zeros((batch_size, K.shape(memory)[-1]), dtype = memory.dtype)

    def get_initial_state(self, memory, batch_size = None):
        if batch_size is None: batch_size = K.shape(memory)[0]
        return (
            K.zeros((batch_size, K.shape(memory)[1]), dtype = memory.dtype),
            K.zeros((batch_size, K.shape(memory)[1]), dtype = memory.dtype)
        )

    def process_memory(self, memory, mask = None):
        if mask is not None:
            memory = K.where(K.expand_dims(mask, axis = 2), memory, 0.)

        processed_memory = self.memory_layer(memory)
        
        return memory, processed_memory
        
    def get_attention_scores(self, query, processed_memory, attn_weights_cat, mask = None):
        """
            Compute the attention weights
            
            Arguments :
                - query : the decoder's output with shape [B, n_mel_channels]
                - processed_memory  : the processed encoder's output with shape [B, seq_in_len, attention_dim]
                - attention_weights_cat : prev attention weights with shape [B, seq_in_len, {1 or 2}]
                - mask  : `bool` padding mask for the encoder's output with shape [B, seq_in_len]
            return : 
                - scores    : the attention scores with shape [B, seq_in_len]
        """
        processed_query = K.expand_dims(self.query_layer(query), axis = 1)
        processed_attention_weights = self.location_layer(attn_weights_cat)

        energies = self.value_layer(K.tanh(
            processed_query + processed_memory + processed_attention_weights
        ))
        energies = K.squeeze(energies, axis = -1)
        
        if mask is not None:
            energies = K.where(mask, energies, K.convert_to_tensor(float('-inf'), energies.dtype))
        
        return K.softmax(energies, axis = -1)
    
    def call(self,
             query,
             memory,
             processed_memory   = None,
             
             mask   = None,
             training   = False,
             initial_state  = None,
             
             ** kwargs
            ):
        """
            Compute the LocationSensitiveAttention
            
            Arguments :
                - query : the decoder last output with shape [B, embedding_dim]
                - memory    : the encoder's output with shape [B, seq_in_len, encoder_embedding_dim]
                - processed_memory  : the encoder's output processed with `self.memory_layer`
                    with shape [B, seq_in_len, self.attention_dim]
                
                - mask    : encoder's padding mask with shape [B, seq_in_len]
                - initial_state : tuple (prev_attn_weights, prev_attn_weights_cumulative)
                    both have shape [B, seq_in_len]
        """
        if initial_state is not None:
            prev_attn_weights, prev_attn_weights_cum = initial_state
        else:
            prev_attn_weights       = K.zeros((K.shape(query)[0], K.shape(memory)[1]))
            prev_attn_weights_cum   = K.zeros((K.shape(query)[0], K.shape(memory)[1]))
        
        if self.hparams.concat_mode == 0:
            attn_weights_cat = K.expand_dims(prev_attn_weights, axis = -1)
        elif self.hparams.concat_mode == 1:
            attn_weights_cat = K.expand_dims(prev_attn_weights_cum, axis = -1)
        else:
            attn_weights_cat = K.stack([
                prev_attn_weights, prev_attn_weights_cum
            ], axis = -1)
        
        if processed_memory is None:
            memory, processed_memory = self.process_memory(memory, mask = mask)
        
        attention_weights = self.get_attention_scores(
            query, processed_memory, attn_weights_cat, mask = mask
        )
        
        #print("attention_weights shape : {}".format(attention_weights.shape))
        attention_context = K.matmul(K.expand_dims(attention_weights, 1), memory)
        #print("attention_context shape : {}".format(attention_context.shape))
        #print("memory shape : {}".format(memory.shape))
        attention_context = K.squeeze(attention_context, axis = 1)
        
        if self.hparams.cumulative:
            new_attn_weights_cum = attention_weights + prev_attn_weights_cum
        else:
            new_attn_weights_cum = attention_weights
        
        return attention_context, (attention_weights, new_attn_weights_cum)
    
    def get_config(self):
        return (self.hparams + super().get_config()).get_config()
        