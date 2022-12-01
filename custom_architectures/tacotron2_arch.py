
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

import enum
import logging
import collections
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *

from utils import plot, get_enum_item
from hparams import HParamsTacotron2, HParamsTacotron2Decoder
from custom_layers import FasterEmbedding, ConcatEmbedding, ConcatMode, LocationSensitiveAttention

Tacotron2DecoderCellState = collections.namedtuple(
    "Tacotron2DecoderCellState", [
        "time",
        "attention_rnn_state",
        "decoder_rnn_state",
        "attention_context",
        "attention_state",
        "alignment_history"
    ]
)

Tacotron2DecoderState = collections.namedtuple(
    "Tacotron2DecoderState", [
        "time",
        "last_frame",
        "finished",
        "cell_state",
        "outputs",
        "stop_tokens",
        "lengths"
    ]
)


def _get_var(_vars, i):
    if callable(_vars): return _vars(i)
    elif isinstance(_vars, list): return _vars[i]
    else: return _vars

class Prenet(tf.keras.Model):
    def __init__(self,
                 sizes      = [256, 256],
                 use_bias   = False,
                 activation = 'relu', 
                 drop_rate  = 0.5,
                 deterministic  = False,
                 name       = 'prenet',
                 **kwargs
                ):
        super().__init__(name = name)
        
        self.sizes      = sizes
        self.use_bias   = use_bias
        self.activation = activation
        self.drop_rate  = drop_rate
        self.deterministic  = deterministic
        
        self.denses = [
            tf.keras.layers.Dense(
                size,
                use_bias    = _get_var(use_bias, i),
                activation  = _get_var(activation, i),
                name        = '{}_layer_{}'.format(name, i+1)
            ) for i, size in enumerate(sizes)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)
    
    def set_deterministic(self, deterministic, seed = 0):
        self.deterministic = deterministic

    def call(self, inputs, training = False):
        x = inputs
        for layer in self.denses:
            x = self.dropout(layer(x), training = not self.deterministic)
        return x
    
    def get_config(self):
        return {
            'sizes' : self.sizes,
            'use_bias'  : self.use_bias,
            'activation'    : self.activation,
            'drop_rate'     : self.drop_rate,
            'deterministic' : self.deterministic
        }

def Postnet(n_mel_channels = 80, name = 'postnet', ** kwargs):
    from custom_architectures import get_architecture
    return get_architecture(
        architecture_name   = 'simple_cnn',
        input_shape     = (None, n_mel_channels),
        output_shape    = n_mel_channels,
        
        add_mask_layer  = False,
        conv_type   = 'conv1d',
        strides     = 1,
        padding     = 'same',
        use_mask    = True,
        add_final_norm = True,
        
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
                     speaker_embedding_dim   = None,
                     concat_mode         = ConcatMode.CONCAT,

                     linear_projection   = False,

                     name    = "encoder",
                     ** kwargs
                    ):
    if n_speaker > 1 and not speaker_embedding_dim:
        raise ValueError("If `n_speaker > 1`, you must specify `speaker_embedding_dim`")

    from custom_architectures import get_architecture
    
    inp_text = tf.keras.layers.Input(shape = (None,), dtype = tf.int32, name = 'input_text')
    
    embeddings = FasterEmbedding(
        vocab_size, embedding_dim, mask_value = pad_token, name = "{}_embeddings".format(name)
    )(inp_text)
    
    output = get_architecture(
        architecture_name   = 'simple_cnn',
        inputs  = embeddings,
        output_shape    = embedding_dim,
        
        use_mask    = True,
        add_mask_layer  = False,
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
    
    inputs, spk_input, spk_embed = inp_text, None, None
    if n_speaker > 1:
        spk_input = tf.keras.layers.Input(shape = (1, ), dtype = tf.int32, name = 'speaker_id')
        spk_embed = FasterEmbedding(
            n_speaker, speaker_embedding_dim, name = 'speaker_embedding'
        )(spk_input)
    elif speaker_embedding_dim:
        spk_input = tf.keras.layers.Input(shape = (speaker_embedding_dim, ), name = 'speaker_embedding')
        spk_embed = spk_input
    
    if speaker_embedding_dim:
        concat_mode = get_enum_item(concat_mode, ConcatMode)
        if speaker_embedding_dim != embedding_dim and concat_mode != ConcatMode.CONCAT:
            spk_embed = tf.keras.layers.Dense(
                embedding_dim, name = 'embedding_resizing'
            )(spk_embed)
        
        inputs = [inp_text, spk_input]
        output = ConcatEmbedding(concat_mode = concat_mode)([output, spk_embed])
    
    if linear_projection:
        output = tf.keras.layers.Dense(embedding_dim, name = 'projection')(output)
    
    return tf.keras.Model(inputs, output, name = name)

class Tacotron2DecoderCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, name = 'decoder_cell', ** kwargs):
        super().__init__(name = name)

        self.hparams = HParamsTacotron2Decoder.extract(kwargs)
        self.attention_dim = self.hparams.lsa_attention_dim
        
        self.prenet = Prenet(** self.hparams.get_config(prefix = 'prenet'))
        
        self.attention_rnn = tf.keras.layers.LSTMCell(
            self.hparams.attention_rnn_dim,
            dropout             = self.hparams.p_attention_dropout,
            recurrent_dropout   = self.hparams.p_attention_dropout,
            name = '{}_attention_rnn'.format(name)
        )
        
        self.attention_layer = LocationSensitiveAttention(
            ** self.hparams.get_config(prefix = 'lsa')
        )

        self.decoder_rnn = tf.keras.layers.StackedRNNCells([
            tf.keras.layers.LSTMCell(
                self.hparams.decoder_rnn_dim,
                dropout             = self.hparams.p_decoder_dropout,
                recurrent_dropout   = self.hparams.p_decoder_dropout,
                name = 'decoder_rnn_cell_{}'.format(i)
            ) for i in range(self.hparams.decoder_n_lstm)],
            name="{}_decoder_rnn".format(name)
        )
        
        self.linear_projection = tf.keras.layers.Dense(
            units   = self.hparams.n_mel_channels * self.hparams.n_frames_per_step, 
            name    = '{}_linear_projection'.format(name)
        )
        
        self.gate_layer = tf.keras.layers.Dense(
            units       = self.hparams.n_frames_per_step, 
            activation  = 'sigmoid' if self.hparams.with_logits else None,
            name        = '{}_gate_output'.format(name)
        )
        
    @property
    def output_size(self):
        return self.hparams.n_mel_channels * self.hparams.n_frames_per_step

    @property
    def state_signature(self):
        return Tacotron2DecoderCellState(
            time    = tf.TensorSpec(shape = (), dtype = tf.int32),
            attention_rnn_state = tf.nest.map_structure(
                lambda s: tf.TensorSpec(shape = (None, s), dtype = tf.float32), self.attention_rnn.state_size
            ),
            decoder_rnn_state   = tf.nest.map_structure(
                lambda s: tf.TensorSpec(shape = (None, s), dtype = tf.float32), self.decoder_rnn.state_size
            ),
            attention_context   = tf.TensorSpec(shape = (None, None), dtype = tf.float32),
            attention_state     = tuple([
                tf.TensorSpec(shape = s, dtype = tf.float32) for s in self.attention_layer.state_size
            ]),
            alignment_history   = tf.TensorSpec(shape = (), dtype = tf.float32)
        )
    
    def get_initial_state(self, inputs, memory):
        batch_size = tf.shape(memory)[0]
        
        self.attention_rnn.reset_dropout_mask()
        self.attention_rnn.reset_recurrent_dropout_mask()
        
        for cell in self.decoder_rnn.cells:
            cell.reset_dropout_mask()
            cell.reset_recurrent_dropout_mask()
        
        initial_attention_rnn_cell_states = self.attention_rnn.get_initial_state(
            None, batch_size, dtype = tf.float32
        )
        initial_decoder_rnn_cell_states = self.decoder_rnn.get_initial_state(
            None, batch_size, dtype = tf.float32
        )
        initial_context = self.attention_layer.get_initial_context(
            None, memory, batch_size = batch_size, dtype = tf.float32
        )
        initial_attention_state = self.attention_layer.get_initial_state(
            None, memory, batch_size = batch_size, dtype = tf.float32
        )
        initial_alignment_history = tf.TensorArray(
            dtype = tf.float32, size = 0, dynamic_size = True, element_shape = (None, None)
        )
        return Tacotron2DecoderCellState(
            time        = tf.zeros([], dtype = tf.int32),
            attention_rnn_state = initial_attention_rnn_cell_states,
            decoder_rnn_state   = initial_decoder_rnn_cell_states,
            attention_context   = initial_context,
            attention_state     = initial_attention_state,
            alignment_history   = initial_alignment_history
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
        # 1. apply prenet for decoder_input.
        prenet_out = self.prenet(inputs, training = training)  # [batch_size, dim]
        
        # 2. concat prenet_out and prev context vector
        # then use it as input of attention lstm layer.
        attention_rnn_input = tf.concat([prenet_out, state.attention_context], axis = -1)
        attention_rnn_output, new_attention_rnn_state = self.attention_rnn(
            attention_rnn_input, state.attention_rnn_state
        )

        # 3. compute context, alignment and cumulative alignment.
        
        attn_context, new_attn_state = self.attention_layer(
            attention_rnn_output,
            memory,
            initial_state   = state.attention_state,
            
            mask    = memory_mask,
            training    = training,
            processed_memory    = processed_memory
        )
        
        # 4. run decoder lstm(s)
        decoder_rnn_input = tf.concat([attention_rnn_output, attn_context], axis = -1)
        decoder_rnn_output, new_decoder_rnn_state = self.decoder_rnn(
            decoder_rnn_input, state.decoder_rnn_state
        )

        decoder_rnn_out_cat = tf.concat([
            decoder_rnn_output, attn_context
        ], axis = -1)
        
        # 5. compute frame feature and stop token.
        decoder_outputs = self.linear_projection(decoder_rnn_out_cat)
        
        if self.hparams.pred_stop_on_mel:
            stop_token_input = tf.concat([decoder_rnn_output, decoder_outputs], axis = -1)
        else:
            stop_token_input = decoder_rnn_out_cat
        
        stop_tokens = self.gate_layer(stop_token_input)

        # 6. save alignment history to visualize.
        alignment_history = state.alignment_history.write(state.time, new_attn_state[0])

        # 7. return new states.
        new_states = Tacotron2DecoderCellState(
            time    = state.time + 1,
            attention_rnn_state = new_attention_rnn_state,
            decoder_rnn_state   = new_decoder_rnn_state,
            attention_context   = attn_context,
            attention_state     = new_attn_state,
            alignment_history   = alignment_history
        )

        return (decoder_outputs, stop_tokens), new_states
    
    def get_config(self):
        return (self.hparams + super().get_config()).get_config()

class Tacotron2Decoder(tf.keras.Model):
    def __init__(self, name = 'decoder', ** kwargs):
        super().__init__(name = name)
        self.cell   = Tacotron2DecoderCell(** kwargs)
        
        self.n_frames_per_step  = self.cell.hparams.n_frames_per_step
        self.n_mel_channels = self.cell.hparams.n_mel_channels
        
        self.output_size    = self.n_mel_channels * self.n_frames_per_step
    
    def call(self,
             inputs,
             encoder_output,
             
             input_length   = None,
             initial_state  = None,
             
             mask   = None,
             training   = False,
             encoder_mask   = None,
             
             max_length = None,
             early_stopping = False
            ):
        def cond(t, last_frame, finished, cell_state, outputs, stop_tokens, lengths):
            return t < max_length and not (early_stopping and tf.reduce_all(finished))
        
        def body(t, last_frame, finished, cell_state, outputs, stop_tokens, lengths):
            input_frame = last_frame if inputs is None else inputs[:, t]
            
            (frame, stop_token), new_cell_state = self.cell(
                input_frame,
                memory  = memory,
                state   = cell_state,
                processed_memory    = processed_memory,
                
                training    = training,
                memory_mask = encoder_mask
            )
            
            finished = tf.logical_or(finished, stop_token[:, -1] > 0.5)
            lengths  = lengths + tf.cast(tf.logical_not(finished), tf.int32)
            
            if mask is not None:
                frame = tf.where(tf.expand_dims(mask[:, t], axis = -1), frame, 0.)
            
            outputs     = outputs.write(t, frame)
            stop_tokens = stop_tokens.write(t, stop_token)
            
            return Tacotron2DecoderState(
                time    = t + 1,
                last_frame  = frame,
                finished    = finished,
                cell_state  = new_cell_state,
                outputs     = outputs,
                stop_tokens = stop_tokens,
                lengths     = lengths
            )

        batch_size = tf.shape(encoder_output)[0]
        
        if inputs is not None:
            if isinstance(inputs, (list, tuple)):
                inputs, input_length = inputs

            if max_length is None:
                max_length = tf.shape(inputs)[1]

            if mask is None and input_length is not None:
                mask = tf.sequence_mask(
                    input_length, tf.shape(inputs)[1], dtype = tf.bool
                )

        if initial_state is None:
            initial_state = self.cell.get_initial_state(inputs, encoder_output)

        memory, processed_memory = self.cell.process_memory(encoder_output, mask = encoder_mask)

        dynamic_size    = True if inputs is None else False
        
        outputs     = tf.TensorArray(
            size    = 0 if dynamic_size else max_length,
            dtype   = tf.float32,
            dynamic_size    = dynamic_size,
            element_shape   = (encoder_output.shape[0], self.output_size)
        )
        stop_tokens = tf.TensorArray(
            size    = 0 if dynamic_size else max_length,
            dtype   = tf.float32,
            dynamic_size    = dynamic_size,
            element_shape   = (encoder_output.shape[0], self.n_frames_per_step)
        )

        last_state = tf.while_loop(
            cond    = cond,
            body    = body,
            loop_vars   = Tacotron2DecoderState(
                time    = tf.zeros([], dtype = tf.int32),
                last_frame  = tf.zeros((batch_size, self.n_mel_channels), dtype = tf.float32),
                finished    = tf.fill((batch_size, ), False),
                cell_state  = initial_state,
                outputs     = outputs,
                stop_tokens = stop_tokens,
                lengths     = tf.zeros((batch_size, ), dtype = tf.int32)
            ),
            maximum_iterations  = max_length,
            parallel_iterations = 32,
            swap_memory = True
        )
        
        outputs = tf.reshape(tf.transpose(
            last_state.outputs.stack(), [1, 0, 2]
        ), [batch_size, -1, self.n_mel_channels])
        stop_tokens = tf.reshape(tf.transpose(
            last_state.stop_tokens.stack(), [1, 0, 2]
        ), [batch_size, -1])
        
        if mask is None:
            mask = tf.sequence_mask(last_state.lengths, tf.shape(outputs)[1])
        
        return (outputs, stop_tokens, mask), last_state.cell_state

    def call_for_loop(self,
                      inputs,
                      encoder_output,

                      input_length   = None,
                      initial_state  = None,
             
                      mask   = None,
                      training   = False,
                      encoder_mask   = None,
             
                      max_length = None
                     ):
        if isinstance(inputs, (list, tuple)):
            inputs, input_length = inputs
        
        if max_length is None:
            max_length = tf.shape(inputs)[1]
        
        if mask is None and input_length is not None:
            mask = tf.sequence_mask(
                input_length, tf.shape(inputs)[1], dtype = tf.bool
            )

        if initial_state is None:
            initial_state = self.cell.get_initial_state(inputs, encoder_output)

        processed_memory = self.cell.process_memory(encoder_output, mask = encoder_mask)

        outputs     = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True)
        stop_tokens = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True)

        state = initial_state
        for t in tf.range(max_length):
            (frame, stop_token), state = self.cell(
                inputs[:, t],
                memory  = encoder_output,
                state   = state,
                processed_memory    = processed_memory,
                
                training    = training,
                memory_mask = encoder_mask
            )
            
            if mask is not None:
                frame = tf.where(tf.expand_dims(mask[:, t], axis = -1), frame, 0.)
            
            outputs     = outputs.write(t, frame)
            stop_tokens = stop_tokens.write(t, stop_token)

        outputs = tf.reshape(tf.transpose(
            outputs.stack(), [1, 0, 2]
        ), [tf.shape(encoder_output)[0], -1, self.n_mel_channels])
        stop_tokens = tf.reshape(tf.transpose(
            stop_tokens.stack(), [1, 0, 2]
        ), [tf.shape(encoder_output)[0], -1])
        
        return (outputs, stop_tokens, mask), state

class Tacotron2(tf.keras.Model):
    def __init__(self, name = 'tacotron2', ** kwargs):
        super().__init__(name = name)
        self.hparams    = HParamsTacotron2(** kwargs)
        
        self.maximum_iterations = self.hparams.max_decoder_steps
        
        self.encoder    = Tacotron2Encoder(
            vocab_size  = self.hparams.vocab_size, 
            ** self.hparams.get_config(prefix = 'encoder')
        )
        
        self.decoder    = Tacotron2Decoder(** self.hparams)
        
        self.postnet    = Postnet(
            n_mel_channels  = self.hparams.n_mel_channels,
            ** self.hparams.get_config(prefix = 'postnet')
        )
    
    def _build(self):
        input_text = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

        mel_outputs = np.random.normal(
            size = (1, 20, self.hparams.n_mel_channels)
        ).astype(np.float32)
        mel_lengths = np.array([20])
        
        encoder_input = input_text
        if self.hparams.encoder_speaker_embedding_dim:
            if self.hparams.encoder_n_speaker > 1:
                spk_input = np.array([0])
            else:
                spk_input = np.random.normal(
                    size = (1, self.hparams.encoder_speaker_embedding_dim)
                ).astype(np.float32)
            encoder_input = [input_text, spk_input]
        
        self([encoder_input, [mel_outputs, mel_lengths]], training = False)

    @property
    def encoder_embedding_dim(self):
        return self.encoder.output_shape[-1]
    
    def set_deterministic(self, deterministic, seed = 0):
        self.decoder.cell.prenet.set_deterministic(deterministic, seed = seed)
    
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
        encoder_mask    = encoder_output._keras_mask
        
        # decoder_output shape == [batch_size, seq_out_len, n_mel_channels]
        # stop_tokens shape == [batch_size, seq_out_len]
        (decoder_output, stop_tokens, decoder_mask), last_state = self.decoder(
            decoder_input,
            encoder_output  = encoder_output,
            encoder_mask    = encoder_mask,
            training    = training
        )

        postnet_output  = self.postnet(decoder_output, training = training, mask = decoder_mask)

        mel_outputs     = decoder_output + postnet_output

        alignment_history = tf.transpose(
            last_state.alignment_history.stack(), [1, 0, 2]
        )
        
        outputs = (decoder_output, mel_outputs, stop_tokens, alignment_history)
        return outputs if not return_state else (outputs, last_state)

    @tf.function(experimental_relax_shapes = True)
    def infer(self,
              inputs,
              training  = False,
              max_length    = -1,
              early_stopping    = True,
              return_state  = False,
              ** kwargs
             ):
        if max_length <= 0: max_length = self.maximum_iterations

        # encoder_output shape == [batch_size, seq_in_len, encoder_embedding_dim]
        encoder_output  = self.encoder(inputs, training = training)
        encoder_mask    = encoder_output._keras_mask
        
        # decoder_output shape == [batch_size, seq_out_len, n_mel_channels]
        # stop_tokens shape == [batch_size, seq_out_len]
        (decoder_output, stop_tokens, decoder_mask), last_state = self.decoder(
            None,
            encoder_output  = encoder_output,
            encoder_mask    = encoder_mask,
            training    = training,
            max_length  = max_length,
            early_stopping  = early_stopping
        )

        postnet_output  = self.postnet(decoder_output, training = training, mask = decoder_mask)

        mel_outputs     = decoder_output + postnet_output

        alignment_history = tf.transpose(
            last_state.alignment_history.stack(), [1, 0, 2]
        )
        
        outputs = (decoder_output, mel_outputs, stop_tokens, alignment_history)
        return outputs if not return_state else (outputs, last_state)
    
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
        
custom_functions    = {
    'nvidia_tacotron'   : pytorch_tacotron,
    
    'Prenet'    : Prenet,
    'Postnet'   : Postnet,
    'Tacotron2Encoder'  : Tacotron2Encoder,
    'Tacotron2Decoder'  : Tacotron2Decoder,
    'Tacotron2DecoderCell'  : Tacotron2DecoderCell,
    'Tacotron2' : Tacotron2
}

custom_objects  = {
    'LocationSensitiveAttention' : LocationSensitiveAttention,
    'Prenet'    : Prenet,
    'Tacotron2Decoder'  : Tacotron2Decoder,
    'Tacotron2DecoderCell'  : Tacotron2DecoderCell,
    'Tacotron2' : Tacotron2
}
