import collections
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *

from utils import plot
from hparams import HParamsTacotron2, HParamsTacotron2Decoder
from custom_layers import LocationSensitiveAttention, FasterEmbedding
from custom_architectures.current_blocks import Conv1DBN
from custom_architectures.dynamic_decoder import BaseDecoder, dynamic_decode

_rnn_impl = 2

TacotronDecoderCellState = collections.namedtuple(
    "TacotronDecoderCellState",
    [
        "time",
        "attention_rnn_state",
        "decoder_rnn_state",
        "attention_context",
        "attention_weights",
        "attention_weights_cum",
        "alignment_history",
    ]
)

TacotronDecoderOutput = collections.namedtuple(
    "TacotronDecoderOutput", ("mel_output", "token_output", "sample_id")
)


def _get_var(_vars, i):
    if callable(_vars): return _vars(i)
    elif isinstance(_vars, list): return _vars[i]
    else: return _vars
    
class Tacotron2Sampler:
    """ Tacotron2 sampler (inspired from tftts Tacotron2Sampler) """
    def __init__(self, hparams):
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step  = hparams.n_frames_per_step
        self.add_go_frame   = hparams.add_go_frame
        self.remove_last_frame  = hparams.remove_last_frame
        self.early_stopping    = hparams.early_stopping
        self.teacher_forcing_mode   = hparams.teacher_forcing_mode
        
        self.step   = tf.Variable(
            hparams.init_step, dtype = tf.float32, trainable = False, name = 'step'
        )
        self.init_ratio = tf.constant(hparams.init_ratio, dtype = tf.float32)
        self.final_ratio    = tf.constant(hparams.final_ratio, dtype = tf.float32)
        
        self.init_decrease_step = tf.constant(hparams.init_decrease_step, dtype = tf.float32)
        self.decreasing_steps = tf.constant(hparams.decreasing_steps, dtype = tf.float32)
        self.final_decrease_step = self.init_decrease_step + self.decreasing_steps
        
        self.decrease_factor    = tf.constant(
            (self.init_ratio - self.final_ratio) / hparams.decreasing_steps,
            dtype = tf.float32
        )
        
        self._ratio = tf.Variable(
            self.get_ratio(self.step), dtype = tf.float32, trainable = False,
            name = 'teacher_forcing_ratio'
        )
        self._teacher_forcing   = False
    
    def set_step(self, step):
        self.step.assign(tf.cast(step, tf.float32))
        self._ratio.assign(self.get_ratio(step))
        
    def get_ratio(self, step, teacher_forcing_mode = None):
        if teacher_forcing_mode is None: teacher_forcing_mode = self.teacher_forcing_mode
        step = tf.cast(step, tf.float32)
        if teacher_forcing_mode == 'constant':
            return self.init_ratio
        elif teacher_forcing_mode == 'linear':
            if step < self.init_decrease_step:
                return self.init_ratio
            elif step > self.final_decrease_step:
                return self.final_ratio
            else:
                return self.init_ratio - self.decrease_factor * (step - self.init_decrease_step)
        elif teacher_forcing_mode == 'random':
            return tf.random.uniform(
                (), minval = self.final_ratio, maxval = self.init_ratio
            )
        elif teacher_forcing_mode == 'random_linear':
            if step < self.init_decrease_step:
                return self.init_ratio
            elif step > self.final_decrease_step:
                return tf.random.uniform(
                    (), minval = self.final_ratio, maxval = self.init_ratio
                )
            else:
                return tf.random.uniform(
                    (), 
                    maxval = self.init_ratio, 
                    minval = self.init_ratio - self.decrease_factor * (step - self.init_decrease_step)
                )
            
    def plot(self, steps = None, teacher_forcing_mode = None,
             title = 'teacher forcing ratio over steps', 
             xlabel = 'steps', ylabel = 'ratio', ylim = (0,1), ** kwargs):
        if steps is None: steps = self.step
        data = [float(self.get_ratio(i, teacher_forcing_mode)) for i in range(int(steps))]
        
        return plot(
            data, title = title, xlabel = xlabel, ylabel = ylabel, ylim = ylim, ** kwargs
        )
        
    def finish_training_step(self):
        self.step.assign_add(1.)
        self._ratio.assign(self.get_ratio(self.step))
    
    def setup_target(self, mel_target, mel_lengths):
        """Setup ground-truth mel outputs for decoder."""
        frame_0 = self.n_frames_per_step - 1 if self.add_go_frame else 0
        self.mel_lengths    = mel_lengths
        self.mel_target     = mel_target[
            :, frame_0 :: self.n_frames_per_step, :
        ]
        
        self.set_batch_size(tf.shape(mel_target)[0])
        self.max_lengths = tf.tile([tf.shape(self.mel_target)[1]], [self._batch_size])
        if self.remove_last_frame:
            self.max_lengths = self.max_lengths - 1
        self._teacher_forcing   = True        

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    @property
    def reduction_factor(self):
        return self.n_frames_per_step

    def set_batch_size(self, batch_size):
        self._teacher_forcing   = False
        self._batch_size = batch_size

    def initialize(self):
        """ Return (Finished, next_inputs). """
        if self._teacher_forcing and not self.add_go_frame:
            initial_input = self.mel_target[:,0,:]
        else:
            initial_input = tf.tile([[0.0]], [self._batch_size, self.n_mel_channels])
        
        return (
            tf.tile([False], [self._batch_size]),
            initial_input
        )

    def sample(self, time, outputs, state):
        return tf.tile([0], [self._batch_size])

    def next_inputs(self,
                    time,
                    pred_outputs,
                    state,
                    sample_ids,
                    stop_token_prediction,
                    training    = False,
                    **kwargs,
                   ):
        if self._teacher_forcing:
            if not self.add_go_frame: time = time + 1
            finished = time >= self.max_lengths
            if tf.reduce_all(finished):
                if training: self.finish_training_step()
    
                next_inputs = tf.zeros(
                    [self._batch_size, self.n_mel_channels], dtype = tf.float32
                )
            else:
                next_inputs = (
                    self._ratio * self.mel_target[:, time, :]
                    + (1.0 - self._ratio) * pred_outputs[:, -self.n_mel_channels :]
                )
            next_state = state
            return (finished, next_inputs, next_state)
        else:
            finished = tf.cast(tf.round(stop_token_prediction), tf.bool)
            finished = tf.math.logical_and(tf.reduce_all(finished), self.early_stopping)
            next_inputs = pred_outputs[:, -self.n_mel_channels :]
            next_state = state
            return (finished, next_inputs, next_state)

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
        super(Prenet, self).__init__(name = name)
        
        self.sizes      = sizes
        self.use_bias   = use_bias
        self.activation = activation
        self.drop_rate  = drop_rate
        self.deterministic  = deterministic
        
        self.denses = [
            Dense(
                size, use_bias = use_bias, activation = activation, 
                name = '{}_layer_{}'.format(name, i+1)
            ) for i, size in enumerate(sizes)
        ]
        self.dropout = Dropout(drop_rate)
        
    def call(self, inputs, training = False):
        x = inputs
        for layer in self.denses:
            x = layer(x)
            x = self.dropout(x, training = not self.deterministic)
        return x
    
    def get_config(self):
        config = {}
        config['sizes']     = self.sizes
        config['use_bias']  = self.use_bias
        config['activation']    = self.activation
        config['drop_rate']     = self.drop_rate
        config['deterministic'] = self.deterministic
        return config

def Postnet(n_mel_channels  = 80, 
            n_convolutions  = 5, 
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
            
            name    = 'postnet',
            ** kwargs
           ):
    model = tf.keras.Sequential(name = name)

    for i in range(n_convolutions):
        config = {
            'filters'        : _get_var(filters, i),
            'kernel_size'    : _get_var(kernel_size, i),
            'use_bias'       : _get_var(use_bias, i),
            'strides'        : 1,
            'padding'        : 'same',
            'activation'     : _get_var(activation, i),
            'bnorm'          : _get_var(bnorm, i),
            'momentum'       : _get_var(momentum, i),
            'epsilon'        : _get_var(epsilon, i),
            'drop_rate'      : _get_var(drop_rate, i),
            'name'           : '{}_{}'.format(name, i+1)
        }
        if i == n_convolutions - 1 and not linear_projection: # last layer
            config['filters']       = n_mel_channels
            config['activation']    = final_activation
            config['drop_rate']     = 0.
        
        Conv1DBN(model, ** config)
    
    if linear_projection:
        model.add(Dense(
            n_mel_channels,
            activation  = final_activation,
            name    = '{}_projection'.format(name)
        ))
        
    return model
    

class Tacotron2Encoder(tf.keras.Model):
    def __init__(self,
                 vocab_size, 
                 embedding_dims  = 512,
                    
                 n_convolutions  = 3,
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
                    
                 name    = "encoder"
                ):
        if n_speaker > 1 and not speaker_embedding_dim:
            raise ValueError("If n_speaker > 1, you must specify speaker_embedding_dim")
        
        super().__init__(name = name)
        self.vocab_size     = vocab_size
        self.embedding_dims     = embedding_dims
                    
        self.n_convolutions     = n_convolutions
        self.kernel_size    = kernel_size
        self.use_bias       = use_bias
    
        self.bnorm      = bnorm
        self.epsilon    = epsilon
        self.momentum   = momentum

        self.drop_rate  = drop_rate
        self.activation = activation
        
        self.n_speaker  = n_speaker
        self.speaker_embedding_dim  = speaker_embedding_dim
        self.concat_mode        = concat_mode
        self.linear_projection  = linear_projection
        
        self.embedding_layer = FasterEmbedding(
            vocab_size, embedding_dims, name = "{}_embeddings".format(name)
        )
        
        self.convs = tf.keras.Sequential(name = '{}_convs'.format(name))
        
        for i in range(n_convolutions):
            config = {
                'filters'        : embedding_dims,
                'kernel_size'    : _get_var(kernel_size, i),
                'use_bias'       : _get_var(use_bias, i),
                'strides'        : 1,
                'padding'        : 'same',
                'activation'     : _get_var(activation, i),
                'bnorm'          : _get_var(bnorm, i),
                'momentum'       : _get_var(momentum, i),
                'epsilon'        : _get_var(epsilon, i),
                'drop_rate'      : _get_var(drop_rate, i),
                'name'           : '{}_{}'.format(name, i+1)
            }
            Conv1DBN(self.convs, ** config)
        
        self.bi_lstm_layer = Bidirectional(LSTM(
            embedding_dims // 2, return_sequences = True, implementation = _rnn_impl
        ), name = "{}_lstm".format(name))
        
        if n_speaker > 1:
            self.speaker_embedding_layer = Embedding(
                n_speaker, speaker_embedding_dim,
                name = '{}_speaker_embedding'.format(name)
            )
        
        self.expand_embedding_layer = None
        if speaker_embedding_dim and speaker_embedding_dim != embedding_dims and concat_mode != 'concat':
            self.expand_embedding_layer = Dense(
                embedding_dims, name = '{}_expand_spk_embedding'.format(name)
            )
        
        self.projection_layer   = None
        if linear_projection:
            self.projection_layer = Dense(
                embedding_dims, use_bias = False, name = '{}_projection'.format(name)
            )
    
    @property
    def use_speaker_id(self):
        return self.n_speaker > 1
    
    @property
    def use_speaker_embedding(self):
        return self.speaker_embedding_dim is not None
    
    @property
    def embedding_dim(self):
        if self.linear_projection or not self.use_speaker_embedding:
            return self.embedding_dims
        return self.embedding_dims + self.speaker_embedding_dim
    
    def _concat(self, rnn_out, speaker_embedding):
        if self.concat_mode == 'add':
            return rnn_out + speaker_embedding
        elif self.concat_mode == 'sub':
            return rnn_out - speaker_embedding
        elif self.concat_mode == 'mul':
            return rnn_out * speaker_embedding
        elif self.concat_mode == 'div':
            return rnn_out / speaker_embedding
        else:
            sequence_length = tf.shape(rnn_out)[1]

            speaker_embedding = tf.expand_dims(speaker_embedding, 1)
            speaker_embedding = tf.tile(speaker_embedding, [1, sequence_length, 1])

            return tf.concat([rnn_out, speaker_embedding], axis = -1)
    
    def call(self, inputs, mask = None, training = False):
        if self.use_speaker_embedding:
            input_text, input_speaker = inputs
        else:
            input_text, input_speaker = inputs, None
        
        text_embedding = self.embedding_layer(input_text, training = training)
        
        conv_out = self.convs(text_embedding, training = training)
        
        rnn_out = self.bi_lstm_layer(conv_out, mask = mask, training = training)
        
        output = rnn_out
        
        if self.use_speaker_embedding:
            if self.use_speaker_id:
                speaker_embedding = self.speaker_embedding_layer(input_speaker)
            else:
                speaker_embedding = input_speaker
            
            if self.expand_embedding_layer is not None:
                speaker_embedding = self.expand_embedding_layer(speaker_ids)
        
            output = self._concat(output, speaker_embedding)
        
        if self.linear_projection:
            output = self.projection_layer(output)
        
        return output
        
    def get_config(self):
        config['vocab_size']     = self.vocab_size
        config['embedding_dims']     = self.embedding_dims
                    
        config['n_convolutions']     = self.n_convolutions
        config['kernel_size']    = self.kernel_size
        config['use_bias']       = self.use_bias
    
        config['bnorm']      = self.bnorm
        config['epsilon']    = self.epsilon
        config['momentum']   = self.momentum

        config['drop_rate']  = self.drop_rate
        config['activation'] = self.activation
                    
        config['n_speaker']  = self.n_speaker
        config['speaker_embedding_dim']  = self.speaker_embedding_dim
        config['concat_mode']        = self.concat_mode
        config['linear_projection']  = self.linear_projection
        
        return config

class Tacotron2DecoderCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, name = 'decoder_cell', ** kwargs):
        super().__init__(name = name)
        self.encoder_embedding_dim  = None
        self.encoder_seq_length     = None
        
        self.hparams = HParamsTacotron2Decoder(** kwargs)
        self.attention_dim = self.hparams.lsa_attention_dim
        
        self.prenet = Prenet(** self.hparams.get_config(prefix = 'prenet'))
        
        self.attention_rnn = LSTMCell(
            self.hparams.attention_rnn_dim,
            dropout             = self.hparams.p_attention_dropout,
            recurrent_dropout   = self.hparams.p_attention_dropout,
            implementation      = _rnn_impl,
            name = '{}_attention_rnn'.format(name)
        )
        
        self.attention_layer = LocationSensitiveAttention(
            ** self.hparams.get_config(prefix = 'lsa')
        )

        self.decoder_rnn = tf.keras.layers.StackedRNNCells([
            LSTMCell(
                self.hparams.decoder_rnn_dim,
                dropout             = self.hparams.p_decoder_dropout,
                recurrent_dropout   = self.hparams.p_decoder_dropout,
                implementation      = _rnn_impl,
                name = 'decoder_rnn_cell_{}'.format(i)
            ) for i in range(self.hparams.decoder_n_lstm)],
            name="{}_decoder_rnn".format(name)
        )
        
        self.linear_projection = Dense(
            units   = self.hparams.n_mel_channels * self.hparams.n_frames_per_step, 
            name    = '{}_linear_projection'.format(name)
        )
        
        self.gate_layer = Dense(
            units       = self.hparams.n_frames_per_step, 
            activation  = 'sigmoid' if self.hparams.with_logits else None,
            name        = '{}_gate_output'.format(name)
        )
        
    @property
    def output_size(self):
        """Return output (mel) size."""
        return self.linear_projection.units

    @property
    def state_size(self):
        """Return hidden state size."""
        return TacotronDecoderCellState(
            time    = tf.TensorShape([]),
            attention_rnn_state = self.attention_lstm.state_size,
            decoder_rnn_state   = self.decoder_rnn.state_size,
            attention_context   = self.attention_dim,
            attention_weights   = self.encoder_seq_length,
            attention_weights_cum   = self.encoder_seq_length,
            alignment_history   = ()
        )
            
    def set_encoder_embedding_dim(self, encoder_embedding_dim):
        self.encoder_embedding_dim = encoder_embedding_dim
        
    def set_encoder_seq_length(self, encoder_seq_length):
        self.encoder_seq_length = encoder_seq_length
    
    def get_initial_state(self, batch_size):
        """ Get initial states. """
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
            batch_size, size = self.encoder_embedding_dim
        )
        initial_attention_weights = self.attention_layer.get_initial_weights(
            batch_size, size = self.encoder_seq_length
        )
        initial_alignment_history = tf.TensorArray(
            dtype = tf.float32, size = 0, dynamic_size = True
        )
        return TacotronDecoderCellState(
            time        = tf.zeros([], dtype = tf.int32),
            attention_rnn_state = initial_attention_rnn_cell_states,
            decoder_rnn_state   = initial_decoder_rnn_cell_states,
            attention_context   = initial_context,
            attention_weights   = initial_attention_weights,
            attention_weights_cum   = initial_attention_weights,
            alignment_history   = initial_alignment_history,
        )
                
    def call(self, inputs, states, training = False):
        """
            Compute (mel_output, stop_token), next_state for this timestep. 
            Inputs : 
                - last_mel_outputs : shape == [batch_size, n_mel_channels]
        """
        # 1. apply prenet for decoder_input.
        prenet_out = self.prenet(inputs, training = training)  # [batch_size, dim]
        
        # 2. concat prenet_out and prev context vector
        # then use it as input of attention lstm layer.
        attention_rnn_input = tf.concat([prenet_out, states.attention_context], axis=-1)
        attention_rnn_output, new_attention_rnn_state = self.attention_rnn(
            attention_rnn_input, states.attention_rnn_state
        )

        # 3. compute context, alignment and cumulative alignment.
        prev_attn_weights       = states.attention_weights
        prev_attn_weights_cum   = states.attention_weights_cum
        prev_alignment_history  = states.alignment_history
        
        attn_context, attn_weights, attn_weights_cum = self.attention_layer(
            [attention_rnn_output, prev_attn_weights, prev_attn_weights_cum], 
            training = training,
        )
        
        # 4. run decoder lstm(s)
        decoder_rnn_input = tf.concat([attention_rnn_output, attn_context], axis = -1)
        decoder_rnn_output, new_decoder_rnn_state = self.decoder_rnn(
            decoder_rnn_input, states.decoder_rnn_state
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
        alignment_history = prev_alignment_history.write(states.time, attn_weights)

        # 7. return new states.
        new_states = TacotronDecoderCellState(
            time    = states.time + 1,
            attention_rnn_state = new_attention_rnn_state,
            decoder_rnn_state   = new_decoder_rnn_state,
            attention_context   = attn_context,
            attention_weights   = attn_weights,
            attention_weights_cum   = attn_weights_cum,
            alignment_history   = alignment_history,
        )

        return (decoder_outputs, stop_tokens), new_states
    
    def get_config(self):
        return self.hparams.get_config()
        
class Tacotron2Decoder(BaseDecoder):
    def __init__(self,
                 sampler,
                 decoder_cell,
                 output_layer = None,
                 name = 'decoder'
                ):
        """Initial variables."""
        super().__init__(name = name)
        self.cell       = decoder_cell
        self.sampler    = sampler
        self.output_layer   = output_layer

    def setup_decoder_init_state(self, decoder_init_state):
        self.initial_state = decoder_init_state

    def initialize(self, **kwargs):
        return self.sampler.initialize() + (self.initial_state,)

    @property
    def output_size(self):
        return TacotronDecoderOutput(
            mel_output  = tf.nest.map_structure(
                lambda shape: tf.TensorShape(shape), self.cell.output_size
            ),
            token_output    = tf.TensorShape(self.sampler.reduction_factor),
            sample_id       = self.sampler.sample_ids_shape,  # tf.TensorShape([])
        )

    @property
    def output_dtype(self):
        return TacotronDecoderOutput(
            tf.float32, tf.float32, self.sampler.sample_ids_dtype
        )

    @property
    def batch_size(self):
        return self.sampler._batch_size
        
    def step(self, time, inputs, state, training = False):
        (mel_outputs, stop_tokens), cell_state = self.cell(
            inputs, state, training = training
        )
        
        if self.output_layer is not None:
            mel_outputs = self.output_layer(mel_outputs)
        
        sample_ids = self.sampler.sample(
            time = time, outputs = mel_outputs, state = cell_state
        )
        (finished, next_inputs, next_state) = self.sampler.next_inputs(
            time        = time,
            pred_outputs    = mel_outputs,
            state       = cell_state,
            sample_ids  = sample_ids,
            stop_token_prediction   = stop_tokens,
            training    = training
        )

        outputs = TacotronDecoderOutput(mel_outputs, stop_tokens, sample_ids)
        return (outputs, next_state, next_inputs, finished)

class Tacotron2(tf.keras.Model):
    def __init__(self, name = 'tacotron2', ** kwargs):
        super().__init__(name = name)
        self.hparams    = HParamsTacotron2(** kwargs)
        
        self.maximum_iterations = self.hparams.max_decoder_steps
        
        sampler         = Tacotron2Sampler(self.hparams)
        decoder_cell    = Tacotron2DecoderCell(** self.hparams)
        
        self.encoder    = Tacotron2Encoder(
            vocab_size  = self.hparams.vocab_size, 
            ** self.hparams.get_config(prefix = 'encoder')
        )
                
        self.decoder    = Tacotron2Decoder(
            sampler     = sampler,
            decoder_cell    = decoder_cell,
            output_layer    = None
        )
                
        self.postnet    = Postnet(
            n_mel_channels  = self.hparams.n_mel_channels,
            ** self.hparams.get_config(prefix = 'postnet')
        )
        
        decoder_cell.set_encoder_embedding_dim(self.encoder.embedding_dim)
            
    @property
    def encoder_embedding_dim(self):
        return self.encoder.output_shape[-1]
    
    @property
    def sampler(self):
        return self.decoder.sampler
    
    def set_step(self, step):
        self.sampler.set_step(step)
    
    def setup_maximum_iterations(self, maximum_iterations):
        """ Call only for inference. """
        self.maximum_iterations = maximum_iterations

    def _build(self):
        input_text = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        input_lengths = np.array([9])
        mel_outputs = np.random.normal(
            size = (1, 20, self.hparams.n_mel_channels)
        ).astype(np.float32)
        mel_lengths = np.array([20])
        
        if self.encoder.use_speaker_embedding:
            if self.encoder.use_speaker_id:
                spk_input = np.array([0])
            else:
                spk_input = np.random.normal(
                    size = (1, self.encoder.speaker_embedding_dim)
                ).astype(np.float32)
            encoder_input = [input_text, input_lengths, spk_input]
        else:
            encoder_input = [input_text, input_lengths]
        
        inputs = encoder_input + [mel_outputs, mel_lengths]
        self(inputs, training = False)

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
        if self.encoder.use_speaker_embedding:
            input_text, input_lengths, input_speaker, mel_target, mel_lengths = inputs
            encoder_input = [input_text, input_speaker]
        else:
            input_text, input_lengths, mel_target, mel_lengths = inputs
            encoder_input = input_text

        # create input-mask based on input_lengths
        input_mask = tf.sequence_mask(
            input_lengths,
            maxlen  = tf.reduce_max(input_lengths),
            name    = "input_sequence_masks",
        )

        # Encoder Step.
        encoder_output = self.encoder(
            encoder_input, mask = input_mask, training = training
        )

        batch_size = tf.shape(encoder_output)[0]
        encoder_seq_len = tf.shape(encoder_output)[1]

        # Setup some initial placeholders for decoder step. Include:
        # 1. mel_gts, mel_lengths for teacher forcing mode.
        # 2. alignment_size for attention size.
        # 3. initial state for decoder cell.
        # 4. memory (encoder hidden state) for attention mechanism.
        self.sampler.setup_target(mel_target = mel_target, mel_lengths = mel_lengths)

        self.decoder.cell.set_encoder_seq_length(encoder_seq_len)
        self.decoder.setup_decoder_init_state(
            self.decoder.cell.get_initial_state(batch_size) if initial_state is None else initial_state
        )
        self.decoder.cell.attention_layer.setup_memory(
            memory = encoder_output,
            memory_sequence_length = input_lengths,  # use for mask attention.
        )

        # run decode step.
        (
            (frames_prediction, stop_token_prediction, _),
            final_decoder_state,
            _,
        ) = dynamic_decode(
            self.decoder,
            maximum_iterations  = None,
            training    = training,
            swap_memory = True
        )

        decoder_outputs = tf.reshape(
            frames_prediction, [batch_size, -1, self.hparams.n_mel_channels]
        )
        stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

        postnet_output = self.postnet(decoder_outputs, training = training)

        mel_outputs = decoder_outputs + postnet_output

        alignment_history = tf.transpose(
            final_decoder_state.alignment_history.stack(), [1, 0, 2]
        )
        if return_state:
            return (
                decoder_outputs,
                mel_outputs,
                stop_token_prediction,
                alignment_history
            ), final_decoder_state
        
        return decoder_outputs, mel_outputs, stop_token_prediction, alignment_history

    @tf.function(experimental_relax_shapes = True,
                 #input_signature = [
                 #    tf.TensorSpec([None, None], dtype=tf.int32, name="input_text"),
                 #    tf.TensorSpec([None], dtype=tf.int32, name="input_lengths")
                 #]
                )
    def infer(self, inputs, input_lengths, ** kwargs):
        """Call logic."""
        # create input-mask based on input_lengths
        input_mask = tf.sequence_mask(
            input_lengths,
            maxlen=tf.reduce_max(input_lengths),
            name="input_sequence_masks",
        )

        # Encoder Step.
        encoder_hidden_states = self.encoder(
            inputs, mask = input_mask, training = False
        )

        batch_size = tf.shape(encoder_hidden_states)[0]
        encoder_seq_len = tf.shape(encoder_hidden_states)[1]

        # Setup some initial placeholders for decoder step. Include:
        # 1. batch_size for inference.
        # 2. alignment_size for attention size.
        # 3. initial state for decoder cell.
        # 4. memory (encoder hidden state) for attention mechanism.
        # 5. window front/back to solve long sentence synthesize problems. (call after setup memory.)
        self.decoder.sampler.set_batch_size(batch_size)

        self.decoder.cell.set_encoder_seq_length(encoder_seq_len)
        self.decoder.setup_decoder_init_state(
            self.decoder.cell.get_initial_state(batch_size)
        )
        self.decoder.cell.attention_layer.setup_memory(
            memory=encoder_hidden_states,
            memory_sequence_length=input_lengths,  # use for mask attention.
        )

        # run decode step.
        (
            (frames_prediction, stop_token_prediction, _),
            final_decoder_state,
            _,
        ) = dynamic_decode(
            self.decoder,
            maximum_iterations = self.maximum_iterations,
            training = False
        )

        decoder_outputs = tf.reshape(
            frames_prediction, [batch_size, -1, self.hparams.n_mel_channels]
        )
        stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

        postnet_output = self.postnet(decoder_outputs, training = False)

        mel_outputs = decoder_outputs + postnet_output

        alignment_history = tf.transpose(
            final_decoder_state.alignment_history.stack(), [1, 0, 2]
        )

        return decoder_outputs, mel_outputs, stop_token_prediction, alignment_history
    
        
    def get_config(self):
        config = self.hparams.get_config()
        config['init_step'] = int(tf.cast(self.sampler.step, tf.int32))
        return config
    
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
    'Tacotron2Encoder'  : Tacotron2Encoder,
    'Tacotron2Decoder'  : Tacotron2Decoder,
    'Tacotron2DecoderCell'  : Tacotron2DecoderCell,
    'Tacotron2' : Tacotron2
}
    
