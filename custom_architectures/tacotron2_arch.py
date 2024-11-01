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

import enum
import keras
import logging
import collections
import numpy as np
import keras.ops as K

from functools import cached_property

from utils import HParams, pad_to_multiple
from utils.keras_utils import TensorSpec, ops, graph_compile
from custom_layers import CustomRNNDropoutCell, CustomEmbedding, ConcatEmbedding, ConcatMode, HParamsLSA, LocationSensitiveAttention
from .current_blocks import _get_var
from .simple_models import simple_cnn

Tacotron2DecoderCellState = collections.namedtuple(
    "Tacotron2DecoderCellState", [
        "attention_rnn_state",
        "decoder_rnn_state",
        "attention_context",
        "attention_state"
    ]
)

Tacotron2DecoderState = collections.namedtuple(
    "Tacotron2DecoderState", [
        "t",
        
        "outputs",
        "lengths",
        "stop_tokens",
        "attention_weights",
        
        "finished",
        "cell_state",
        "main_attention"
    ]
)

Tacotron2InferenceOutput    = collections.namedtuple(
    "Tacotron2InferenceOutput", [
        "decoder_output", "mel", "stop_tokens", "attention_weights", "lengths"
    ]
)


HParamsTacotron2Encoder = HParams(
    vocab_size  = 148,
    
    pad_token   = 0,
    embedding_dim   = 512,
    n_conv  = 3,
    kernel_size     = 5,
    use_bias        = True,
    
    bnorm           = 'after',
    epsilon         = 1e-5,
    momentum        = 0.1,

    drop_rate       = 0.5,
    activation      = 'relu',
    
    n_speaker       = 1,
    speaker_embedding_dim   = None,
    concat_mode         = 'concat',
    linear_projection   = False,
    
    name    = 'encoder'
)

HParamsTacotron2Prenet  = HParams(
    sizes       = [256, 256],
    use_bias    = False,
    activation  = 'relu', 
    drop_rate   = 0.5,
    concat_speaker  = False,
    deterministic   = False,
    name        = 'prenet'
)

HParamsTacotron2Postnet = HParams(
    n_conv      = 5,
    filters     = 512,
    kernel_size = 5,
    use_bias    = True,
    
    bnorm       = 'after',
    epsilon     = 1e-5,
    momentum    = 0.1,
    
    drop_rate   = 0.5,
    activation  = 'tanh',
    final_activation    = None,
    linear_projection   = False,
    name    = 'postnet'
)

HParamsTacotron2Decoder = HParams(
    ** HParamsLSA.get_config(add_prefix = 'lsa'),
    ** HParamsTacotron2Prenet.get_config(add_prefix = 'prenet'),

    max_decoder_steps   = None,
    
    n_mel_channels  = 80,
    with_logits     = True,
    n_frames_per_step   = 1,
    pred_stop_on_mel    = False,
    
    attention_rnn_dim  = 1024, 
    p_attention_dropout    = 0.,
        
    decoder_n_lstm     = 1,
    decoder_rnn_dim    = 1024,
    p_decoder_dropout  = 0.
)

HParamsTacotron2    = HParams(
    ** HParamsTacotron2Encoder.get_config(add_prefix = 'encoder'),
    ** HParamsTacotron2Decoder,
    ** HParamsTacotron2Postnet.get_config(add_prefix = 'postnet'),
    speaker_concat_pos  = 'end',
    vocab_size  = 148
)

def _eye_init(shape, dtype):
    w = K.eye(shape[1], dtype = dtype)
    w = K.pad(w, [(0, shape[0] - shape[1]), (0, 0)])
    return w

@keras.saving.register_keras_serializable('tacotron2')
class Tacotron2Prenet(keras.Model, CustomRNNDropoutCell):
    def __init__(self,
                 sizes      = [256, 256],
                 use_bias   = False,
                 activation = 'relu', 
                 drop_rate  = 0.5,
                 concat_speaker = False,
                 deterministic  = False,
                 name       = 'prenet',
                 **kwargs
                ):
        super().__init__(name = name)
        
        self.sizes      = sizes
        self.use_bias   = use_bias
        self.activation = activation
        self.drop_rate  = drop_rate
        self.concat_speaker = concat_speaker
        self.deterministic  = deterministic
        
        self.denses = [
            keras.layers.Dense(
                size,
                use_bias    = _get_var(use_bias, i),
                activation  = _get_var(activation, i),
                name        = 'layer_{}'.format(i)
            ) for i, size in enumerate(sizes)
        ]
        self.seed_generator = keras.random.SeedGenerator(42)
    
    def build(self, input_shape):
        super().build(input_shape)
        
        self.dropout_shape = {}
        for i, inp_channels in enumerate([input_shape[-1]] + self.sizes[:-1]):
            self.denses[i].build((None, None, inp_channels))
            self.dropout_shape['dropout_{}'.format(i)] = (self.sizes[i], )
        
    def set_deterministic(self, deterministic, seed = 0):
        self.deterministic = deterministic

    def init_dropout_mask(self, ** kwargs):
        kwargs['training'] = not self.deterministic
        return super().init_dropout_mask(** kwargs)
    
    def call(self, inputs, training = False, spk_embedding = None, step = None):
        if spk_embedding is not None and self.concat_speaker:
            if len(K.shape(inputs)) == 3:
                spk_embedding   = K.tile(spk_embedding[:, None, :], [1, K.shape(inputs)[1], 1])
            inputs = K.concatenate([inputs, spk_embedding], axis = -1)
            
        x = inputs
        for i, layer in enumerate(self.denses):
            x = layer(x)
            if not self.deterministic:
                if step is None:
                    x = keras.random.dropout(x, self.drop_rate, seed = self.seed_generator)
                else:
                    x = self.dropout(x, step = step, name = f'dropout_{i}', training = True)
        
        return x
    
    def get_config(self):
        return {
            'sizes' : self.sizes,
            'use_bias'  : self.use_bias,
            'activation'    : self.activation,
            'drop_rate'     : self.drop_rate,
            'deterministic' : self.deterministic
        }

def Tacotron2Postnet(n_mel_channels = 80, name = 'postnet', ** kwargs):
    return simple_cnn(
        input_shape     = (None, n_mel_channels),
        output_shape    = n_mel_channels,
        
        add_mask_layer  = False,
        conv_type   = 'conv1d',
        strides     = 1,
        padding     = 'same',
        use_mask    = True,
        add_final_norm  = True,
        use_sequential  = False,
        
        flatten = False,
        dense_as_final  = False,
        
        name    = name,
        ** kwargs
    )


def Tacotron2Encoder(vocab_size,
                     embedding_dim  = 512,
                     pad_token  = 0,

                     use_mask   = True,
                     n_speaker       = 1,
                     
                     concat_pos = 'end',
                     concat_mode    = ConcatMode.CONCAT,
                     speaker_embedding_dim   = None,

                     linear_projection   = False,

                     name    = "encoder",
                     ** kwargs
                    ):
    def _maybe_concat_speaker(_pos, x, spk_input = None):
        if _pos not in concat_pos: return spk_input, x
        
        spk_embed = None
        if n_speaker > 1:
            if spk_input is None:
                spk_input = keras.layers.Input(shape = (1, ), dtype = 'int32', name = 'speaker_id')
            
            spk_embed = CustomEmbedding(
                n_speaker, speaker_embedding_dim, name = 'speaker_embedding'
            )(spk_input)
        elif speaker_embedding_dim:
            if spk_input is None:
                spk_input = keras.layers.Input(
                    shape = (speaker_embedding_dim, ), name = 'speaker_embedding'
                )
            spk_embed = spk_input
        else:
            return spk_input, x

        if speaker_embedding_dim != embedding_dim and concat_mode not in ('concat', ConcatMode.CONCAT):
            spk_embed = keras.layers.Dense(
                embedding_dim, name = 'embedding_resizing'
            )(spk_embed)

        mask = x._keras_mask

        spk_embed   = keras.layers.Masking()(spk_embed)
        output      = ConcatEmbedding(concat_mode = concat_mode)([x, spk_embed])
        output._keras_mask = mask

        if _pos == 'start':
            output = keras.layers.Dense(
                embedding_dim, kernel_initializer = _eye_init, name = 'embedding_projection'
            )(output)
            output._keras_mask = mask
        
        return spk_input, output

        
    if n_speaker > 1 and not speaker_embedding_dim:
        raise ValueError("If `n_speaker > 1`, you must specify `speaker_embedding_dim`")
    
    inp_text = keras.layers.Input(shape = (None,), dtype = 'int32', name = 'input_text')
    
    inputs, spk_input = inp_text, None

    embeddings = CustomEmbedding(
        vocab_size, embedding_dim, mask_value = pad_token, name = "{}_embeddings".format(name)
    )(inp_text)
    
    spk_input, embeddings = _maybe_concat_speaker('start', embeddings, spk_input)

    output = simple_cnn(
        inputs  = embeddings,
        output_shape    = embedding_dim,
        
        use_mask    = True,
        add_mask_layer  = False,
        use_manual_padding  = False,
        filters     = embedding_dim,
        conv_type   = 'conv1d',
        strides     = 1,
        padding     = 'same',

        flatten = True,
        flatten_type    = 'bi_lstm',
        flatten_kwargs  = {'units' : embedding_dim // 2, 'return_sequences' : True},
        
        dense_as_final  = False,
        return_output   = True,
        
        ** kwargs
    )

    spk_input, output = _maybe_concat_speaker('end', output, spk_input)
    
    if spk_input is not None: inputs = [inp_text, spk_input]
    
    if linear_projection:
        output = keras.layers.Dense(embedding_dim, name = 'projection')(output)
    
    return keras.Model(inputs, output, name = name)

@keras.saving.register_keras_serializable('tacotron2')
class Tacotron2DecoderCell(keras.layers.Layer):
    def __init__(self, name = 'decoder_cell', ** kwargs):
        super().__init__(name = name)

        self.hparams = HParamsTacotron2Decoder.extract(kwargs)
        self.attention_dim = self.hparams.lsa_attention_dim
        
        self.attention_rnn = keras.layers.LSTMCell(
            self.hparams.attention_rnn_dim,
            dropout             = self.hparams.p_attention_dropout,
            recurrent_dropout   = self.hparams.p_attention_dropout,
            name = 'attention_rnn'
        )
        
        self.attention_layer = LocationSensitiveAttention(
            ** self.hparams.get_config(prefix = 'lsa')
        )

        self.decoder_rnn = keras.layers.StackedRNNCells([
            keras.layers.LSTMCell(
                self.hparams.decoder_rnn_dim,
                dropout             = self.hparams.p_decoder_dropout,
                recurrent_dropout   = self.hparams.p_decoder_dropout,
                name = 'cell_{}'.format(i)
            ) for i in range(self.hparams.decoder_n_lstm)],
            name="decoder_rnn"
        )
    
    def build(self, input_shape):
        super().build(input_shape)
        prenet_out_shape, memory_shape = input_shape
        
        self.attention_rnn.build((None, prenet_out_shape[-1] + memory_shape[2]))
        self.attention_layer.build([
            (None, self.hparams.attention_rnn_dim), memory_shape
        ])
        self.decoder_rnn.build((None, self.hparams.attention_rnn_dim + memory_shape[2]))
    
    def get_dropout_mask(self, inputs, memory, state, training):
        self.attention_rnn.get_dropout_mask(inputs)
        self.attention_rnn.get_recurrent_dropout_mask(state.attention_rnn_state)
        
        batch_size, seq_len = K.shape(inputs)[0], K.shape(inputs)[1]
        for i, cell in enumerate(self.decoder_rnn.cells):
            if i == 0:
                size = self.hparams.attention_rnn_dim + K.shape(memory)[2]
            else:
                size = self.hparams.decoder_rnn_dim
            cell.get_dropout_mask(K.zeros((batch_size, seq_len, size), dtype = memory.dtype))
            cell.get_recurrent_dropout_mask(state.decoder_rnn_state[i])

    def reset_dropout_mask(self):
        self.attention_rnn.reset_dropout_mask()
        self.attention_rnn.reset_recurrent_dropout_mask()
        
        for cell in self.decoder_rnn.cells:
            cell.reset_dropout_mask()
            cell.reset_recurrent_dropout_mask()

    def get_initial_state(self, inputs, memory, rnn_states = None):
        attn_rnn_state, decoder_rnn_state = rnn_states if rnn_states is not None else (None, None)
        
        batch_size = K.shape(memory)[0]
        
        initial_attention_rnn_cell_states   = self.attention_rnn.get_initial_state(
            batch_size
        ) if attn_rnn_state is None else attn_rnn_state
        initial_decoder_rnn_cell_states     = self.decoder_rnn.get_initial_state(
            batch_size
        ) if decoder_rnn_state is None else decoder_rnn_state
        initial_context = self.attention_layer.get_initial_context(
            memory, batch_size = batch_size
        )
        initial_attention_state = self.attention_layer.get_initial_state(
            memory, batch_size = batch_size
        )
        return Tacotron2DecoderCellState(
            attention_rnn_state = initial_attention_rnn_cell_states,
            decoder_rnn_state   = initial_decoder_rnn_cell_states,
            attention_context   = initial_context,
            attention_state     = initial_attention_state
        )

    def process_memory(self, memory, mask = None):
        return self.attention_layer.process_memory(memory, mask = mask)
        
    def call(self,
             inputs,
             memory,
             state,
             training   = False,
             memory_mask    = None,
             processed_memory   = None
            ):
        """
            Compute new mel output based on current input (las predicted mel frame) and current state
            
            Arguments :
                - inputs    : the last output with shape [batch_size, n_mel_channels]
                - memory    : the encoder's output with shape [batch_size, seq_in_len, enc_emb_dim]
                - state     : a valid Tacotron2DecoderCellState
                - memory_mask   : padding mask for `memory` with shape [batch_size, seq_in_len]
                - processed_memory  : processed `memory` via the `self.process_memory(...)`, useful to avoid re-processing memory at each timestep
            Returns : ((next_output, stop_token), next_state)
                - next_output   : the new predicted mel frame(s)
                - stop_token    : the stop-token score
                - nex_state     : the new Tacotron2DecoderCellState
        """
        (
            attention_rnn_state,
            decoder_rnn_state,
            attention_context,
            attention_state
        ) = state
        # 2. concat prenet_out and prev context vector
        # then use it as input of attention lstm layer.
        attention_rnn_input = K.concatenate([inputs, attention_context], axis = -1)
        attention_rnn_output, new_attention_rnn_state = self.attention_rnn(
            attention_rnn_input, attention_rnn_state
        )

        # 3. compute context, alignment and cumulative alignment.
        attn_context, new_attn_state = self.attention_layer(
            attention_rnn_output,
            memory,
            initial_state   = attention_state,
            
            mask    = memory_mask,
            training    = training,
            processed_memory    = processed_memory
        )
        
        # 4. run decoder lstm(s)
        decoder_rnn_input = K.concatenate([attention_rnn_output, attn_context], axis = -1)
        decoder_rnn_output, new_decoder_rnn_state = self.decoder_rnn(
            decoder_rnn_input, decoder_rnn_state
        )

        decoder_rnn_out_cat = K.concatenate([
            decoder_rnn_output, attn_context
        ], axis = -1)

        # 7. return new states.
        new_states = Tacotron2DecoderCellState(
            attention_rnn_state = new_attention_rnn_state,
            decoder_rnn_state   = [new_decoder_rnn_state],
            attention_context   = attn_context,
            attention_state     = new_attn_state
        )

        return (decoder_rnn_out_cat, new_attn_state[0]), new_states
    
    def get_config(self):
        return (self.hparams + super().get_config()).get_config()

@keras.saving.register_keras_serializable('tacotron2')
class Tacotron2Decoder(keras.Model):
    def __init__(self, name = 'decoder', ** kwargs):
        super().__init__(name = name)
        self.hparams = HParamsTacotron2Decoder.extract(kwargs)

        self.prenet = Tacotron2Prenet(** self.hparams.get_config(prefix = 'prenet'))

        self.cell   = Tacotron2DecoderCell(** kwargs)
        
        self.linear_projection = keras.layers.Dense(
            units   = self.hparams.n_mel_channels * self.hparams.n_frames_per_step, 
            name    = 'linear_projection'
        )
        self.gate_layer = keras.layers.Dense(
            units       = self.hparams.n_frames_per_step, 
            activation  = 'sigmoid' if self.hparams.with_logits else None,
            name        = 'gate_output'
        )

        self.n_frames_per_step  = self.hparams.n_frames_per_step
        self.n_mel_channels     = self.hparams.n_mel_channels
        
        self.output_size    = self.n_mel_channels * self.n_frames_per_step
        self.supports_masking   = True
    
    def build(self, input_shape):
        super().build(input_shape)
        mel_shape, memory_shape = input_shape
        
        self.prenet.build(mel_shape)
        self.cell.build([(None, self.prenet.sizes[-1]), memory_shape])
        self.linear_projection.build((None, self.hparams.decoder_rnn_dim + memory_shape[2]))
        self.gate_layer.build((None, self.hparams.decoder_rnn_dim + memory_shape[2]))

    def call(self,
             inputs,
             encoder_output,
             speaker_embedding  = None,
             
             mask   = None,
             training   = False,
             encoder_mask   = None
            ):
        def step(inputs, state):
            (cell_output, _), new_cell_state = self.cell(
                inputs,
                memory  = memory,
                state   = state,
                processed_memory    = processed_memory,
                
                training    = training,
                memory_mask = encoder_mask
            )
            
            return cell_output, new_cell_state
        
        if isinstance(inputs, (list, tuple)):
            inputs, input_length = inputs
        else:
            input_length = None

        if mask is None:
            if input_length is not None:
                mask = K.arange(K.shape(inputs)[1])[None, :] <= input_length[:, None]
            else:
                mask = K.any(inputs != 0., axis = 2)
        
        initial_state = self.cell.get_initial_state(inputs, encoder_output)

        memory, processed_memory = self.cell.process_memory(encoder_output, mask = encoder_mask)
        
        prenet_out = self.prenet(inputs, training = training, spk_embedding = speaker_embedding)
        
        if training:
            self.cell.get_dropout_mask(prenet_out, memory, initial_state, training = training)

        _, cell_outputs, _ = keras.src.backend.rnn(
            step,
            inputs  = prenet_out,
            initial_states  = initial_state,
            zero_output_for_mask    = False,
            return_all_outputs  = True,
            time_major  = False
        )
        
        outputs = self.linear_projection(cell_outputs)
        
        if self.hparams.pred_stop_on_mel:
            stop_token_input = K.concatenate([cell_outputs, outputs], axis = 2)
        else:
            stop_token_input = cell_outputs
        
        stop_tokens = self.gate_layer(stop_token_input)
        
        if mask is not None:
            outputs     = K.where(mask[:, :, None], outputs, 0.)
        
        if self.n_frames_per_step > 1:
            batch_size  = K.shape(prenet_out)[0]
            seq_len     = K.shape(prenet_out)[1] * self.n_frames_per_step
            
            outputs     = K.reshape(outputs, [batch_size, seq_len, self.n_mel_channels])
            stop_tokens = K.reshape(stop_tokens, [batch_size, seq_len])
        else:
            stop_tokens = K.squeeze(stop_tokens, axis = 2)
        
        if training:
            self.cell.reset_dropout_mask()
        
        try:
            outputs._keras_mask = mask
            stop_tokens._keras_mask = mask
        except AttributeError:
            pass
        
        return outputs, stop_tokens, mask

    def infer(self,
              encoder_output,
              speaker_embedding  = None,

              inputs    = None,
              initial_state  = None,
             
              training   = False,
              encoder_mask   = None,
             
              max_length = None,
              early_stopping = True,
             
              attn_mask_offset   = None,
              attn_mask_win_len  = None
             ):
        def cond(last_frame, state):
            if not early_stopping: return True
            return K.logical_not(K.all(state.finished))
        
        def body(last_frame, state):
            if attn_mask_win_len is not None:
                center  = K.maximum(state.main_attention, attn_mask_offset)
                center  = K.minimum(center, encoder_length - attn_mask_win_len + attn_mask_offset)

                attn_mask   = K.logical_and(
                    center - attn_mask_offset <= _attn_mask,
                    _attn_mask <= center - attn_mask_offset + attn_mask_win_len
                )
                attn_mask   = K.logical_and(attn_mask, encoder_mask)
            else:
                attn_mask   = encoder_mask
            
            
            prenet_out = self.prenet(last_frame, step = state.t, spk_embedding = speaker_embedding)
            
            (cell_output, attn_weights), new_cell_state = self.cell(
                prenet_out,
                memory  = memory,
                state   = state.cell_state,
                processed_memory    = processed_memory,
                
                training    = training,
                memory_mask = attn_mask
            )
            
            frame = self.linear_projection(cell_output)
        
            if self.hparams.pred_stop_on_mel:
                stop_token_input = K.concatenate([cell_output, frame], axis = -1)
            else:
                stop_token_input = cell_output
            stop_token = self.gate_layer(stop_token_input)
            
            
            finished = K.logical_or(state.finished, stop_token[:, -1] > 0.5)
            lengths  = state.lengths + K.cast(K.logical_not(finished), 'int32')

            outputs     = K.slice_update(
                state.outputs, [0, state.t, 0], frame[:, None, :]
            )
            stop_tokens = K.slice_update(
                state.stop_tokens, [0, state.t, 0], stop_token[:, None, :]
            )
            attention_weights   = K.slice_update(
                state.attention_weights, [0, state.t, 0], attn_weights[:, None, :]
            )
                
            next_state = Tacotron2DecoderState(
                t   = state.t + 1,
                
                outputs = outputs,
                lengths = lengths,
                stop_tokens = stop_tokens,
                attention_weights   = attention_weights,
                
                finished    = finished,
                cell_state  = new_cell_state,
                main_attention  = K.argmax(attn_weights, axis = 1)
            )
            return frame, next_state
        
        batch_size = K.shape(encoder_output)[0]
        
        if max_length is None:
            max_length = self.hparams.max_decoder_steps

        if inputs is None or initial_state is None:
            inputs = K.zeros((batch_size, self.n_mel_channels), dtype = encoder_output.dtype)
        
            initial_state = self.cell.get_initial_state(
                inputs, encoder_output, rnn_states = initial_state
            )

        memory, processed_memory = self.cell.process_memory(encoder_output, mask = encoder_mask)
        _attn_mask  = K.arange(K.shape(encoder_output)[1], dtype = 'int32')[None] if attn_mask_win_len is not None else None
        
        encoder_length  = K.count_nonzero(encoder_mask, axis = 1)
        
        self.prenet.init_dropout_mask(batch_size = batch_size, seq_length = max_length)
        
        _, last_state = K.while_loop(
            cond    = cond,
            body    = body,
            loop_vars   = (
                inputs,
                Tacotron2DecoderState(
                    t   = K.zeros((), dtype = 'int32'),
                    
                    outputs = K.zeros(
                        (batch_size, max_length, self.output_size), dtype = 'float32'
                    ),
                    lengths = K.zeros((batch_size, ), dtype = 'int32'),
                    stop_tokens = K.zeros(
                        (batch_size, max_length, self.n_frames_per_step), dtype = 'float32'
                    ),
                    attention_weights   = K.zeros(
                        (batch_size, max_length, K.shape(memory)[1]), dtype = 'float32'
                    ),
                    
                    finished    = K.zeros((batch_size, ), dtype = 'bool'),
                    cell_state  = initial_state,
                    main_attention  = K.zeros((batch_size, ), dtype = 'int32')
                )
            ),
            maximum_iterations  = max_length
        )

        outputs, stop_tokens = last_state.outputs, last_state.stop_tokens
        if self.n_frames_per_step > 1:
            new_len = K.shape(outputs)[1] * self.n_frames_per_step
            outputs     = K.reshape(outputs, [batch_size, new_len, self.n_mel_channels])
            stop_tokens = K.reshape(stop_tokens, [batch_size, new_len])
        else:
            stop_tokens = stop_tokens[:, :, 0]
        
        mask = K.arange(K.shape(outputs)[1])[None] <= last_state.lengths
        
        self.prenet.reset_dropout_mask()
        
        return (outputs, stop_tokens, mask), last_state

@keras.saving.register_keras_serializable('tacotron2')
class Tacotron2(keras.Model):
    def __init__(self, vocab_size = 148, name = 'tacotron2', ** kwargs):
        super().__init__(name = name)
        kwargs.update({'vocab_size' : vocab_size, 'encoder_vocab_size' : vocab_size})
        self.hparams    = HParamsTacotron2(** kwargs)

        self.pad_token  = self.hparams.encoder_pad_token
        if self.speaker_embedding_dim:
            self.speaker_concat_pos = self.hparams.speaker_concat_pos
        else:
            self.speaker_concat_pos = ()
        self.maximum_iterations = self.hparams.max_decoder_steps
        
        self.hparams.prenet_concat_speaker = 'prenet' in self.speaker_concat_pos
        
        self.decoder    = Tacotron2Decoder(** self.hparams)
        
        self.build([(None, None), (None, None, self.hparams.n_mel_channels)])
    
    def build(self, input_shape):
        if self.built: return
        super().build(input_shape)
        with keras.name_scope('encoder'):
            self.encoder    = Tacotron2Encoder(
                ** self.hparams.get_config(prefix = 'encoder'),
                concat_pos  = self.speaker_concat_pos
            )
        
        decoder_inp_channels = self.hparams.n_mel_channels
        if 'prenet' in self.speaker_concat_pos:
            decoder_inp_channels += self.speaker_embedding_dim
        
        self.decoder.build([
            (None, None, decoder_inp_channels), (None, None, self.encoder.output_shape[-1])
        ])
        
        with keras.name_scope('postnet'):
            self.postnet    = Tacotron2Postnet(
                n_mel_channels  = self.hparams.n_mel_channels,
                ** self.hparams.get_config(prefix = 'postnet')
            )
            self.postnet.supports_masking = True

    @property
    def encoder_embedding_dim(self):
        return self.encoder.output_shape[-1]
    
    @property
    def speaker_embedding_dim(self):
        return self.hparams.encoder_speaker_embedding_dim

    def set_deterministic(self, deterministic, seed = 0):
        self.decoder.prenet.set_deterministic(deterministic, seed = seed)
    
    def call(self, inputs, initial_state = None, return_state = False, training = False):
        """
            Call logic (predict mel outputs with mel-target as input)
            Arguments : 
                - inputs : [input_ids, input_lengths, mel_target, mel_length]
                    - input_ids : encoder inputs (char)
                        shape == [batch_size, seq_len]
                    - input_lengths : length of input sentences
                        shape == [batch_size]
                    - mel_target    : mel target 
                        shape == [batch_size, output_seq_len, n_mel_channels]
                    - mel_lengths   : length of mels in mel_target
                        shape == [batch_size]
            Return : [decoder_output, mel_output, stop_tokens, alignment_history]
        """
        if len(inputs) == 2:
            encoder_input, decoder_input = inputs
        else:
            encoder_input, decoder_input = inputs[:-2], inputs[-2:]

        # encoder_output shape == [batch_size, seq_in_len, encoder_embedding_dim]
        encoder_output  = self.encoder(
            encoder_input, training = training
        )
        if getattr(encoder_output, '_keras_mask', None) is not None:
            encoder_mask = encoder_output._keras_mask
        else:
            tokens = encoder_input[0] if isinstance(encoder_input, (list, tuple)) else encoder_input
            encoder_mask = tokens != self.hparams.encoder_pad_token
        
        # decoder_output shape == [batch_size, seq_out_len, n_mel_channels]
        # stop_tokens shape == [batch_size, seq_out_len]
        decoder_output, stop_tokens, decoder_mask = self.decoder(
            decoder_input,
            speaker_embedding   = encoder_input[1] if self.speaker_embedding_dim else None,
            encoder_output  = encoder_output,
            encoder_mask    = encoder_mask,
            training    = training
        )

        postnet_output  = self.postnet(decoder_output, training = training, mask = decoder_mask)
        mel_outputs     = decoder_output + postnet_output
        
        return decoder_output, mel_outputs, stop_tokens

    def prepare_for_xla(self, *, inputs, max_length = None, padding_multiple = 64, ** kwargs):
        tokens = inputs[0] if isinstance(inputs, list) else inputs
        tokens = pad_to_multiple(
            tokens, padding_multiple, axis = 1, constant_values = self.pad_token
        )
        inputs = tokens if not isinstance(inputs, list) else [tokens] + inputs[1:]
        
        if max_length is not None:
            if ops.is_int(max_length):
                max_length += max_length % padding_multiple
            else:
                max_length = K.cast(max_length * tokens.shape[1], 'int32')
            kwargs['max_length'] = max_length
        
        kwargs['inputs'] = inputs
        return kwargs

    @cached_property
    def infer(self):
        signature = TensorSpec(shape = (None, None), dtype = 'int32')
        if self.speaker_embedding_dim:
            signature = [
                signature,
                TensorSpec(shape = (None, self.speaker_embedding_dim), dtype = 'float32')
            ]
        
        return graph_compile(
            self._infer,
            prefer_xla  = True,
            input_signature = [signature],
            prepare_for_xla = self.prepare_for_xla
        )
    
    def _infer(self,
              inputs    : TensorSpec(),
              training  = False,
              
              attn_mask_offset  = 0.5,
              attn_mask_win_len = None,
              
              max_length    : TensorSpec(shape = (), static = True) = None,
              early_stopping    = True,
              
              return_state  = False
             ):
        # encoder_output shape == [batch_size, seq_in_len, encoder_embedding_dim]
        encoder_output  = self.encoder(inputs, training = training)
        if getattr(encoder_output, '_keras_mask', None) is not None:
            encoder_mask = encoder_output._keras_mask
        else:
            tokens = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
            encoder_mask = tokens != self.hparams.encoder_pad_token
        
        if max_length is None:
            max_length = self.maximum_iterations
        elif ops.is_float(max_length):
            length = K.max(K.count_nonzero(encoder_mask, axis = 1))
            max_length  = K.cast(
                K.cast(length, 'float32') * max_length, 'int32'
            )

        if attn_mask_win_len is not None and ops.is_float(attn_mask_offset):
            attn_mask_offset = K.cast(
                K.cast(attn_mask_win_len, 'float32') * attn_mask_offset, 'int32'
            )
        # decoder_output shape == [batch_size, seq_out_len, n_mel_channels]
        # stop_tokens shape == [batch_size, seq_out_len]
        (decoder_output, stop_tokens, decoder_mask), last_state = self.decoder.infer(
            speaker_embedding   = inputs[1] if self.speaker_embedding_dim else None,
            encoder_output  = encoder_output,
            encoder_mask    = encoder_mask,
            training    = training,
            
            attn_mask_offset    = attn_mask_offset,
            attn_mask_win_len   = attn_mask_win_len,
            
            max_length  = max_length,
            early_stopping  = early_stopping
        )
        if keras.backend.backend() == 'jax':
            decoder_mask = None
        
        postnet_output  = self.postnet(decoder_output, training = training, mask = decoder_mask)

        mel_outputs     = decoder_output + postnet_output
        
        return Tacotron2InferenceOutput(
            mel     = mel_outputs,
            lengths = last_state.lengths,
            stop_tokens = stop_tokens,
            attention_weights   = last_state.attention_weights,
            decoder_output  = decoder_output
        )
    
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

def pytorch_tacotron(to_gpu = True, eval_mode = True):
    import torch

    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    if eval_mode: tacotron2 = tacotron2.eval()
    if to_gpu: tacotron2 = tacotron2.to('cuda')
    
    return tacotron2
        