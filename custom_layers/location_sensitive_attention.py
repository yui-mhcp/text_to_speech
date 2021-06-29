import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv1D, Dense
from tensorflow.python.keras.engine import base_layer_utils

from hparams import HParams

HParamsLSA = HParams(
    attention_dim   = 128,
    attention_filters   = 32,
    attention_kernel_size   = 31,
    probability_function    = 'softmax',
    concat_mode     = 2,
    cumulative      = True
)

class LocationLayer(Layer):
    def __init__(self, attention_dim, attention_filters, 
                 attention_kernel_size, **kwargs):
        super(LocationLayer, self).__init__(**kwargs)
        self.attention_dim      = attention_dim
        self.attention_filters  = attention_filters
        self.attention_kernel_size  = attention_kernel_size
        
        self.location_conv = Conv1D(
            filters     = attention_filters,
            kernel_size = attention_kernel_size, 
            use_bias    = False,
            strides     = 1,
            padding     = "same",
            name = "location_conv"
        )
        self.location_dense = Dense(
            attention_dim, use_bias = False, name = "location_dense"
        )
        
    def call(self, inputs):
        processed = self.location_conv(inputs)
        processed = self.location_dense(processed)
        return processed
    
    def get_config(self):
        config = super(LocationLayer, self).get_config()
        config['attention_dim']         = self.attention_dim
        config['attention_filters']     = self.attention_filters
        config['attention_kernel_size'] = self.attention_kernel_size
        return config

class LocationSensitiveAttention(Layer):
    def __init__(self, name = 'location_sensitive_attention', **kwargs):
        super().__init__(name = name)
        self.hparams = HParamsLSA(** kwargs)
        assert self.hparams.concat_mode in (0, 1, 2)
        assert self.hparams.cumulative or self.hparams.concat_mode == 0
        
        self.default_probability_fn = tf.nn.softmax
        
        self.query_layer    = Dense(
            self.hparams.attention_dim, use_bias = False, name = "query_layer"
        )
        self.memory_layer   = Dense(
            self.hparams.attention_dim, use_bias = False, name = "memory_layer"
        )
        self.v  = Dense(1, use_bias = False, name = "value_layer")
        
        self.location_layer = LocationLayer(
            attention_dim   = self.hparams.attention_dim, 
            attention_filters       = self.hparams.attention_filters, 
            attention_kernel_size   = self.hparams.attention_kernel_size, 
            name = "location_layer"
        )
        
    def get_initial_weights(self, batch_size, size):
        """Get initial alignments."""
        return tf.zeros(shape = [batch_size, size], dtype = tf.float32)

    def get_initial_context(self, batch_size, size):
        """Get initial attention."""
        return tf.zeros(
            shape = [batch_size, size], dtype = tf.float32
        )
        
    def setup_memory(self, memory, memory_sequence_length = None, memory_mask = None):
        """
            Pre-process the memory before actually query the memory.
            This should only be called once at the first invocation of call().
            Args:
              memory: The memory to query; usually the output of an RNN encoder.
                This tensor should be shaped `[batch_size, max_time, ...]`.
              memory_sequence_length (optional): Sequence lengths for the batch
                entries in memory. If provided, the memory tensor rows are masked
                with zeros for values past the respective sequence lengths.
              memory_mask: (Optional) The boolean tensor with shape `[batch_size,
                max_time]`. For any value equal to False, the corresponding value
                in memory should be ignored.
        """
        if memory_sequence_length is not None and memory_mask is not None:
            raise ValueError(
                "memory_sequence_length and memory_mask cannot be "
                "used at same time for attention."
            )

        self.values = _prepare_memory(
            memory,
            memory_sequence_length  = memory_sequence_length,
            memory_mask = memory_mask,
            check_inner_dims_defined = False
        )
        # Mark the value as check since the memory and memory mask might not
        # passed from __call__(), which does not have proper keras metadata.
        # TODO(omalleyt12): Remove this hack once the mask the has proper
        # keras history.
        base_layer_utils.mark_checked(self.values)
        self.keys = self.memory_layer(self.values) if self.memory_layer else self.values

        self.batch_size = self.keys.shape[0] or tf.shape(self.keys)[0]
        self.memory_seq_len = self.keys.shape[1] or tf.shape(self.keys)[1]
        if memory_mask is not None or memory_sequence_length is not None:
            unwrapped_probability_fn = self.default_probability_fn

            def _mask_probability_fn(score, ** kwargs):
                return unwrapped_probability_fn(
                    _maybe_mask_score(
                        score,
                        memory_mask = memory_mask,
                        memory_sequence_length  = memory_sequence_length,
                        score_mask_value    = score.dtype.min,
                    ), ** kwargs
                )
            self.probability_fn = _mask_probability_fn
        self._memory_initialized = True
        
    def get_alignment_energies(self, query, attention_weights_cat, processed_memory):
        """
            inputs :
                - query : decoder output (batch, n_mel_channels * n_frames_per_step)
                - processed_memory : processed encoder outputs(batch, T_in, attention_dim)
                - attention_weights_cat : cumulative and prev attention weights (batch, max_time, 2)
            return : 
                - alignment (batch, time_steps, max_time)
        """
        
        processed_query = self.query_layer(tf.expand_dims(query, axis = 1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(tf.nn.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))
        energies = tf.squeeze(energies, axis = -1)
        return energies
    
    def call(self, inputs):
        """
            inputs = [query, prev_attn_weights, prev_attn_weights_cum]            
        """
        query, prev_attn_weights, prev_attn_weights_cum = inputs
        
        if self.hparams.concat_mode == 0:
            attn_weights_cat = tf.expand_dims(prev_attn_weights, axis = -1)
        elif self.hparams.concat_mode == 1:
            attn_weights_cat = tf.expand_dims(prev_attn_weights_cum, axis = -1)
        else:
            attn_weights_cat = tf.concat([
                tf.expand_dims(prev_attn_weights, -1),
                tf.expand_dims(prev_attn_weights_cum, -1)
            ], axis = -1)
        
        alignment = self.get_alignment_energies(
            query, attn_weights_cat, self.keys
        )
        
        #print("Alignment shape : {}".format(alignment.shape))
        attention_weights = self.probability_fn(alignment, axis = 1)
        #print("attention_weights shape : {}".format(attention_weights.shape))
        attention_context = tf.matmul(tf.expand_dims(attention_weights, 1), self.values)
        #print("attention_context shape : {}".format(attention_context.shape))
        #print("memory shape : {}".format(memory.shape))
        attention_context = tf.squeeze(attention_context, axis = 1)
        
        if self.hparams.cumulative:
            new_attn_weights_cum = attention_weights + prev_attn_weights_cum
        else:
            new_attn_weights_cum = attention_weights
        
        return attention_context, attention_weights, new_attn_weights_cum
    
    def get_config(self):
        config = super().get_config()
        return self.hparams + config
        
def _prepare_memory(memory, 
                    memory_sequence_length      = None, 
                    memory_mask = None, 
                    check_inner_dims_defined    = False
                   ):
    """
        Convert to tensor and possibly mask `memory`.
        Args:
          memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
          memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
          memory_mask: `boolean` tensor with shape [batch_size, max_time]. The
            memory should be skipped when the corresponding mask is False.
          check_inner_dims_defined: Python boolean.  If `True`, the `memory`
            argument's shape is checked to ensure all but the two outermost
            dimensions are fully defined.
        Returns:
          A (possibly masked), checked, new `memory`.
        Raises:
          ValueError: If `check_inner_dims_defined` is `True` and not
            `memory.shape[2:].is_fully_defined()`.
    """
    if check_inner_dims_defined:
        def _check_dims(m):
            if not m.shape[2:].is_fully_defined():
                raise ValueError(
                    "Expected memory %s to have fully defined inner dims, "
                    "but saw shape: %s" % (m.name, m.shape)
                )

        tf.nest.map_structure(_check_dims, memory)
    
    if memory_sequence_length is None and memory_mask is None:
        return memory
    elif memory_sequence_length is not None:
        seq_len_mask = tf.sequence_mask(
            memory_sequence_length,
            maxlen  = tf.shape(tf.nest.flatten(memory)[0])[1],
            dtype   = tf.nest.flatten(memory)[0].dtype,
        )
    else:
        # For memory_mask is not None
        seq_len_mask = tf.cast(memory_mask, dtype = tf.nest.flatten(memory)[0].dtype)

    def _maybe_mask(m, seq_len_mask):
        """ Mask the memory based on the memory mask. """
        rank = m.shape.ndims
        rank = rank if rank is not None else tf.rank(m)
        extra_ones = tf.ones(rank - 2, dtype=tf.int32)
        seq_len_mask = tf.reshape(
            seq_len_mask, tf.concat((tf.shape(seq_len_mask), extra_ones), 0)
        )
        return m * seq_len_mask

    return tf.nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)

def _maybe_mask_score(score,
                      memory_sequence_length    = None,
                      memory_mask       = None,
                      score_mask_value  = None
                     ):
    """Mask the attention score based on the masks."""
    if memory_sequence_length is None and memory_mask is None:
        return score
    if memory_sequence_length is not None and memory_mask is not None:
        raise ValueError(
            "memory_sequence_length and memory_mask can't be provided at same time."
        )
    if memory_sequence_length is not None:
        message = "All values in memory_sequence_length must greater than zero."
        with tf.control_dependencies([
            tf.debugging.assert_positive(  # pylint: disable=bad-continuation
                memory_sequence_length, message=message
            )]):
            memory_mask = tf.sequence_mask(
                memory_sequence_length, maxlen = tf.shape(score)[1]
            )
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(memory_mask, score, score_mask_values)
