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

import logging
import collections
import keras.ops as K

from keras import tree

from loggers import timer, time_logger
from utils.keras_utils import TensorSpec, ops

logger = logging.getLogger(__name__)

InferenceConfig = collections.namedtuple(
    "InferenceConfig", [
        "use_xla",
        "use_cache",
        "is_transformer",
        "is_encoder_decoder",
        
        "return_logits",
        "skip_attention",
        "return_attention",
        "return_last_attention",
        "return_only_cross_attention",

        "max_steps",
        "max_length",
        "init_length",
        "start_attention_slice"
    ]
)

InferenceState  = collections.namedtuple(
    "InferenceState", [
        "t", "finished", "state", "padding_mask"
    ]
)

InferenceOutput = collections.namedtuple(
    "InferenceOutput", [
        "tokens", "lengths", "scores", "logits", "state", "attention_weights"
    ]
)

@timer
def infer(self,
          tokens    : TensorSpec(shape = (None, None), dtype = 'int32') = None,
          lengths   : TensorSpec(shape = (None, ), dtype = 'int32')     = None,
          encoder_output    : TensorSpec(dtype = 'float')   = None,
          initial_state  = None,
          prefix    : TensorSpec(dtype = 'float')   = None,

          step_fn    = None,
          training   = False,
          enc_padding_mask  : TensorSpec(shape = (None, None), dtype = 'float') = None,

          method    = 'greedy',
          
          batch_size : TensorSpec(shape = (), dtype = 'int32') = None,
          early_stopping = True,
          is_transformer = False,

          use_cache  = True,
          return_state   = False,
          return_logits  = False,
          return_attention   = False,
          return_last_attention  = False,
          return_only_cross_attention    = True,

          ** kwargs
         ):
    if step_fn is None: step_fn = self
    if initial_state:
        assert prefix is None, 'The `prefix` should be None when providing an `initial_state`'
        assert tokens is not None, 'You must provide `tokens` when providing an `initial_state`'
        use_cache = True
    
    empty_tokens = prefix is None and tokens is None
    
    with time_logger.timer('initialization'):
        if batch_size is None:
            batch_size = _get_batch_size(tokens, encoder_output, prefix)

        state_length, init_length, max_steps, max_length = get_inference_lengths(
            self, tokens = tokens, prefix = prefix, initial_state = initial_state, ** kwargs
        )

        if tokens is None and prefix is None:
            tokens  = K.full((batch_size, 1), self.sos_token, dtype = 'int32')
            lengths = K.ones((batch_size, ), dtype = 'int32')
        elif tokens is None:
            lengths = K.full((batch_size, ), K.shape(prefix)[1], dtype = 'int32')
        elif lengths is None:
            lengths = K.count_nonzero(tokens != self.pad_token, axis = 1)
        
        generated   = K.full((batch_size, max_steps), self.pad_token, dtype = 'int32')
        padding_mask    = K.ones((batch_size, init_length), dtype = 'bool')
    
    config  = InferenceConfig(
        use_xla = not ops.executing_eagerly() or ops.is_jax_backend(),
        use_cache   = use_cache,
        is_transformer  = is_transformer,
        is_encoder_decoder  = encoder_output is not None,
        
        return_logits   = return_logits,
        skip_attention  = not (return_attention or return_last_attention),
        return_attention    = return_attention,
        return_last_attention   = return_last_attention,
        return_only_cross_attention = return_only_cross_attention,

        max_steps   = max_steps,
        max_length  = max_length,
        init_length = init_length,
        start_attention_slice   = init_length - state_length - 1
    )

    if isinstance(method, str): method = _inference_methods[method]
    return method(
        self,
        tokens,
        InferenceOutput(
            tokens  = generated,
            lengths = lengths,
            scores  = K.zeros((batch_size, ), dtype = self.compute_dtype),
            logits  = None,
            state   = (),
            attention_weights   = None
        ),
        InferenceState(
            t   = K.zeros((), dtype = 'int32'),
            state   = initial_state,
            finished    = K.zeros((batch_size, ), dtype = 'bool'),
            padding_mask    = padding_mask
        ),
        config  = config,
        
        step_fn = step_fn,
        
        prefix  = prefix,
        training    = training,
        early_stopping  = early_stopping,
        encoder_output  = encoder_output,
        enc_padding_mask    = enc_padding_mask,
        
        ** kwargs
    )

@timer
def infer_greedy(self,
                 initial_inputs,
                 outputs,
                 loop_state,
                 config,
                 
                 step_fn    = None,
                 training   = False,
                 early_stopping  = True,

                 prefix  = None,
                 encoder_output  = None,
                 enc_padding_mask    = None,
                 
                 return_state   = False,
                 
                 ** kwargs
                ):
    @timer
    def cond(inputs, outputs, loop_state):
        if not early_stopping: return True
        return K.logical_not(K.all(loop_state.finished))

    @timer
    def body(inputs, outputs, loop_state, first_iter = False):
        if logger.isEnabledFor(logging.DEBUG) and ops.is_tensorflow_backend():
            import tensorflow as tf
            tf.print('Step :', loop_state.t, '\n  Inputs (', tf.shape(inputs), ') :', inputs[:, -5:],'\n  Mask (', tf.shape(loop_state.padding_mask), ') :', loop_state.padding_mask[:, -5:])
        
        model_out = step_fn(
            inputs,
            lengths = outputs.lengths,
            encoder_output  = encoder_output,
            initial_state   = loop_state.state if config.use_cache else None,
            prefix      = prefix if not loop_state.state else None,
            
            training    = training,
            apply_softmax   = False,
            padding_mask    = loop_state.padding_mask,
            enc_padding_mask    = enc_padding_mask,
            
            return_state    = config.use_cache or return_state,
            return_attention    = config.return_attention,
            return_last_attention   = config.return_last_attention,
            return_only_cross_attention = config.return_only_cross_attention,
            return_mask = False,
            as_dict = True
        )
        logits = model_out.output
        if len(K.shape(logits)) == 3:
            idx = -1 if config.use_cache or first_iter or not config.use_xla else config.init_length + loop_state.t - 1
            logits = logits[:, idx, :]
        
        logits  = process_logits(
            logits,
            lengths = outputs.lengths,
            tokens  = outputs.tokens,
            state   = loop_state,
            ** kwargs
        )
        next_token, next_token_score = select_next_token(logits, n = 1, ** kwargs)

        next_token  = K.where(loop_state.finished, self.pad_token, next_token[:, 0])
        scores      = outputs.scores + K.where(
            loop_state.finished,
            K.convert_to_tensor(0, dtype = next_token_score.dtype),
            next_token_score[:, 0]
        )
        
        finished    = K.logical_or(loop_state.finished, next_token == self.eos_token)
        lengths     = outputs.lengths + K.cast(K.logical_not(finished), 'int32')
        
        generated   = K.scatter_update(
            outputs.tokens,
            K.stack([
                K.arange(batch_size), K.broadcast_to(loop_state.t, [batch_size])
            ], axis = -1),
            next_token
        )
        
        if config.use_cache:
            next_inputs = next_token[:, None]
        elif not config.use_xla:
            if inputs is not None:
                next_inputs = K.concatenate([inputs, next_token[:, None]], axis = 1)
            else:
                next_inputs = next_token[:, None]
        elif initial_inputs is not None:
            next_inputs = K.concatenate([initial_inputs, generated], axis = 1)
        else:
            next_inputs = generated
        
        next_outputs = InferenceOutput(
            tokens  = generated,
            lengths = lengths,
            scores  = scores,
            logits  = update_logits(
                outputs.logits, logits, state = loop_state, config = config
            ),
            state   = outputs.state,
            attention_weights = update_attention_weights(
                outputs.attention_weights, model_out.attention_weights, loop_state, config
            )
        )
        next_state = update_state(
            state = loop_state, output = model_out, finished = finished, config = config,
            first_iter = first_iter
        )
        return next_inputs, next_outputs, next_state

    batch_size = K.shape(initial_inputs)[0]

    start_lengths = outputs.lengths
    with time_logger.timer('first step'):
        inputs, outputs, state = body(
            initial_inputs, outputs, loop_state, first_iter = True
        )

    _, outputs, state = K.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = (inputs, outputs, state),
        maximum_iterations  = config.max_steps - 1
    )

    return InferenceOutput(
        tokens  = outputs.tokens,
        lengths = outputs.lengths - start_lengths,
        scores  = outputs.scores,
        logits  = outputs.logits if config.return_logits else None,
        state   = state.state if return_state else None,
        attention_weights = None if config.skip_attention else outputs.attention_weights
    )

def infer_beam_search(self,
                      inputs,
                      outputs,
                      loop_state,
                      config,
                 
                      step_fn    = None,
                      training   = False,
                      early_stopping  = True,

                      prefix  = None,
                      encoder_output  = None,
                      enc_padding_mask    = None,
                 
                      num_beams    = 10,
                      num_sentences    = 1,

                      length_power : TensorSpec(shape = (), dtype = 'float') = None,

                      return_state   = False,
                     
                      ** kwargs
                     ):
    @timer
    def cond(inputs, outputs, loop_state):
        if not early_stopping: return True
        return K.logical_not(K.all(K.reshape(
            loop_state.finished, [batch_size, num_beams]
        )[:, : num_sentences]))

    @timer
    def body(inputs, outputs, loop_state, first_iter = False):
        if kwargs.get('debug', False) and ops.is_tensorflow_backend():
            import tensorflow as tf
            tf.print('Step :', loop_state.t, '\n  Inputs (', tf.shape(inputs), ') :', inputs[:, -5:],'\n  Mask (', tf.shape(mask), ') :', mask[:, -5:])
        
        model_out = step_fn(
            inputs,
            lengths = outputs.lengths,
            encoder_output  = encoder_output,
            initial_state   = loop_state.state if config.use_cache else None,
            prefix      = prefix if not loop_state.state else None,
            
            training    = training,
            apply_softmax   = False,
            padding_mask    = loop_state.padding_mask,
            enc_padding_mask    = enc_padding_mask,
            
            return_state    = config.use_cache or return_state,
            return_attention    = config.return_attention,
            return_last_attention   = config.return_last_attention,
            return_only_cross_attention = config.return_only_cross_attention,
            return_mask = False,
            as_dict = True
        )
        logits = model_out.output
        if len(K.shape(logits)) == 3: logits = logits[:, -1, :]
        
        logits  = process_logits(
            logits,
            lengths = outputs.lengths,
            tokens  = outputs.tokens,
            state   = loop_state,
            ** kwargs
        )
        # for finished sentences, only keep the EOS token with a score of 0
        # such that it does not decrease the current score of the sentence
        logits_with_scores  = outputs.scores[:, None] + K.where(
            loop_state.finished[:, None], eos_mask, logits
        )
        # reshape logits to [batch_size, vocab_size * num_beams]
        reshaped_logits  = logits_with_scores
        if length_power is not None:
            reshaped_logits  = reshaped_logits / K.cast(
                outputs.lengths + 1, logits.dtype
            )[:, None] ** length_power

        reshaped_logits = K.reshape(reshaped_logits, [batch_size, num_beams * self.vocab_size])
        if first_iter: reshaped_logits = reshaped_logits[:, : self.vocab_size]
        # the returned token scores are not used as they take into account the length normalization
        next_token, _ = select_next_token(
            reshaped_logits, n = num_beams, ** kwargs
        )
        next_token  = K.reshape(next_token, [effective_batch_size])
        
        beam_index  = next_token // self.vocab_size + batch_idx_add
        next_token  = next_token % self.vocab_size
        # for each data, the correct beams are gathered
        lengths     = K.take(outputs.lengths,     beam_index, axis = 0)
        finished    = K.take(loop_state.finished, beam_index, axis = 0)
        
        logits_with_scores  = K.take(logits_with_scores, beam_index, axis = 0)
        scores      = K.take_along_axis(logits_with_scores, next_token[:, None], axis = 1)[:, 0]
        
        next_token  = K.where(finished, self.pad_token, next_token)

        finished    = K.logical_or(finished, K.equal(next_token, self.eos_token))
        lengths     = lengths + K.cast(K.logical_not(finished), lengths.dtype)

        generated   = K.scatter_update(
            K.take(outputs.tokens, beam_index, axis = 0),
            K.stack([
                K.arange(effective_batch_size),
                K.broadcast_to(loop_state.t, [effective_batch_size])
            ], axis = -1),
            next_token
        )
        
        if config.use_cache:
            next_inputs = next_token[:, None]
        elif not config.use_xla:
            if inputs is not None:
                next_inputs = K.concatenate([inputs, next_token[:, None]], axis = 1)
            else:
                next_inputs = next_token[:, None]
        elif inputs is not None:
            next_inputs = K.concatenate([inputs, generated], axis = 1)
        else:
            next_inputs = generated
        
        next_outputs = InferenceOutput(
            tokens  = generated,
            lengths = lengths,
            scores  = scores,
            logits  = update_logits(
                outputs.logits, logits, state = loop_state, config = config, beam_index = beam_index
            ),
            state   = outputs.state,
            attention_weights = update_attention_weights(
                outputs.attention_weights, model_out.attention_weights, loop_state, config, beam_index = beam_index
            )
        )
        next_state = update_state(
            state = loop_state, output = model_out, finished = finished, config = config,
            first_iter = first_iter, beam_index = beam_index
        )
        return next_inputs, next_outputs, next_state

    batch_size  = inputs.shape[0] if inputs.shape[0] is not None else K.shape(inputs)[0]
    effective_batch_size    = batch_size * num_beams

    batch_idx_add   = K.repeat(K.arange(batch_size), num_beams)
    eos_mask        = K.scatter_update(
        K.full((1, self.vocab_size), float('-inf')), [[0, self.pad_token]], [0.]
    )

    inputs, outputs, loop_state, encoder_output, enc_padding_mask, prefix = tree.map_structure(
        lambda x: K.repeat(x, num_beams, axis = 0) if K.is_tensor(x) and len(K.shape(x)) != 0 else x,
        (inputs, outputs, loop_state, encoder_output, enc_padding_mask, prefix)
    )
    
    start_lengths = outputs.lengths
    with time_logger.timer('first step'):
        inputs, outputs, state = body(
            inputs, outputs, loop_state, first_iter = True
        )

    _, outputs, state = K.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = (inputs, outputs, state),
        maximum_iterations  = config.max_steps - 1
    )

    return tree.map_structure(
        lambda t: K.reshape(
            t, K.concatenate([
                K.convert_to_tensor([batch_size, num_beams], 'int32'),
                K.cast(K.shape(t)[1:], 'int32')
            ], axis = 0)
        )[:, : num_sentences] if t is not None and len(K.shape(t)) != 0 else t,
        InferenceOutput(
            tokens  = outputs.tokens,
            lengths = outputs.lengths - start_lengths,
            scores  = outputs.scores,
            logits  = outputs.logits if config.return_logits else None,
            state   = state.state if return_state else None,
            attention_weights = None if config.skip_attention else outputs.attention_weights
        )
    )

def get_inference_lengths(self,
                          tokens   = None,
                          prefix   = None,
                          initial_state    = None,

                          max_length   = None,
                          max_new_tokens   = None,
                          ** _
                         ):
    """
        Return the information about initial length and maximal steps / length
        
        Arguments :
            - tokens    : the provided tokens from an initial prompt
            - prefix    : prefix to put before the token embeddings (e.g., ClipCap)
            - initial_state : provided initial state
            
            - max_length    : maximal output length
            - max_new_tokens    : maximal number of tokens to generate
        Return :
            - state_length  : `initial_state` length, corresponds to the number of non-zero value in the time dimension of the state
            - init_length   : initial sequence length
                Equivalent to the prefix length + the number of `tokens`
                If neither `prefix` nor `tokens` are provided, equals to `1` to include sos_token
            - max_steps : maximal number of generation steps. This is equivalent to `max_length - init_length`
            - max_length    : the maximal number of outputs after generation
    """
    state_length, init_length = 0, 0
    if initial_state and isinstance(initial_state, dict):
        from .transformers_arch.transformer_arch import _get_state_step
        
        state_length    = _get_state_length(initial_state)
        init_length     += state_length
    elif prefix is not None:
        init_length     += K.shape(prefix)[1]
    
    if tokens is not None:
        init_length += K.shape(tokens)[1]
    elif prefix is None:
        init_length += 1
    
    if max_length is None:
        if max_new_tokens is None:
            max_length = getattr(self, 'max_input_length', init_length + 512)
        else:
            max_length = init_length + max_new_tokens
    
    return (state_length, init_length, max_length - init_length, max_length)

@timer
def process_logits(scores,
                   lengths,
                   *,
                   temperature  = None,
                   length_temperature   = None,
                   logits_filter    = None,
                   ** kwargs
                  ):
    """
        Computes logits (i.e. log-probabilities) based on models' output (scores)
        
        Arguments :
            - scores    : the models' last output with shape [batch_size, vocab_size]
            - lengths   : the tokens' lengths with shape [batch_size, 1]
            - temperature   : the softmax' temperature
                - a temperature < 1 will emphasize the scores' differences
                - a temperature > 1 will reduce the scores' difference
                - a temperature of 0 is equivalent to `argmax` (ignored here)
            - length_temperature    : a custom temperature based on the lengths
                - a temperature > 0 will encourage longer sentences
                - a temperature < 0 will encourage shorter sentences
                - a temperature of 0 has no effect (ignored)
            - logits_filter : a callable that takes `scores` (1st argument) and `kwargs` and returns the filtered `scores`
    """
    if temperature is not None:
        scores = scores / K.cast(temperature, scores.dtype)
    
    if length_temperature is not None:
        scores = scores * (
            K.cast(lengths + 1, scores.dtype) ** K.cast(length_temperature, scores.dtype)
        )
    
    if logits_filter is not None:
        if callable(logits_filter):
            scores = K.cast(logits_filter(scores, ** kwargs), scores.dtype)
        elif K.is_tensor(logits_filter):
            from utils.text import remove_batch_tokens
            scores = K.cast(remove_batch_tokens(scores, logits_filter), scores.dtype)
            
    return K.log_softmax(scores, axis = -1)

@timer
def select_next_token(logits, n, *, temperature = None, ** kwargs):
    """
        Returns top-`k` best scores either greedyly (if `temperature == 0.`) else randomly
        Arguments :
            - logits    : the unnormalized log-probabilities for each word ([batch_size, vocab_size])
            - n         : the number of samples to return
            - temperature   : the softmax' temperature (if `0` takes the argmax (or top-k))
            - dtype     : the result's dtype
        Returns :
            - token : `Tensor` with shape [batch_size, n] and dtype `dtype`
    """
    if temperature is None:
        if n <= 1:
            indices = K.argmax(logits, axis = 1)[:, None]
            return indices, K.take_along_axis(logits, indices, axis = 1)
        
        values, indices = K.top_k(logits, k = n)
        return indices, values
    
    indices = keras.random.categorical(logits, n)
    scores  = K.take_along_axis(logits, indices, axis = 1)
    return indices, scores

def update_logits(prev_logits, logits, config, state, beam_index = None):
    if not config.return_logits: return prev_logits if prev_logits is not None else ()
    
    if beam_index is not None:
        if prev_logits is not None: prev_logits = K.take(prev_logits, beam_index, axis = 0)
        logits = K.take(logits, beam_index, axis = 0)

    logits = logits[:, None, :]
    if not config.use_xla:
        return _update_logits_eager(prev_logits, logits, state = state, config = config)
    return _update_logits_xla(prev_logits, logits, state = state, config = config)

def _update_logits_eager(prev_logits, logits, state, config):
    return logits if prev_logits is None else K.concatenate([prev_logits, logits], axis = 1)

def _update_logits_xla(prev_logits, logits, state, config):
    if prev_logits is None:
        padding = [[0, 0], [0, config.max_steps - 1], [0, 0]]
        return K.pad(logits, padding)
    
    return K.slice_update(
        prev_logits, K.array([0, 1, 0], 'int32') * state.t, logits
    )

def update_attention_weights(prev_attention, new_attention, state, config, beam_index = None):
    if config.skip_attention:
        return prev_attention if prev_attention is not None else ()
    
    if beam_index is not None:
        if prev_attention is not None:
            prev_attention = tree.map_structure(
                lambda t: K.take(t, beam_index, axis = 0), prev_attention
            )
        new_attention   = tree.map_structure(
            lambda t: K.take(t, beam_index, axis = 0), new_attention
        )
        
    if not config.use_xla:
        return _update_attention_weights_eager(prev_attention, new_attention, state, config)
    return _update_attention_weights_xla(prev_attention, new_attention, state, config)

def _update_attention_weights_eager(prev_attention, new_attention, state, config):
    if not config.use_cache: return new_attention
    # if the model is not a Transformer
    if not isinstance(new_attention, dict):
        if prev_attention is None: return new_attention[:, None, :]
        return K.concatenate([prev_attention, new_attention[:, None, :]], axis = 1)
    
    if prev_attention is None:
        return new_attention

    def _concat_attentions(key, prev, attn):
        if 'enc' not in key: prev = K.pad(prev, [[0, 0], [0, 0], [0, 0], [0, 1]])
        return K.concatenate([prev, attn], axis = -2)
    
    return {
        k : _concat_attentions(k, prev_attention[k], new_attention[k])
        for k in prev_attention.keys()
    }

def _update_attention_weights_xla(prev_attention, new_attention, state, config):
    # if the model is not a Transformer
    if not isinstance(new_attention, dict):
        new_attention = new_attention[:, None, :]
        if prev_attention is None:
            padding = [[0, 0], [0, config.max_steps - 1], [0, 0]]
            return K.pad(new_attention, padding)
        
        start_indices = K.array([0, 1, 0], 'int32') * state.t
        return K.slice_update(prev_attention, start_indices, new_attention)
    
    if prev_attention is None:
        def init_attention(key, attn):
            pad     = config.max_length - config.init_length
            padding = [[0, 0], [0, 0], [0, pad], [0, 0 if 'enc' in key else pad]]
            return K.pad(attn[:, :, :, :], padding)
        
        return {k : init_attention(k, v) for k, v in new_attention.items()}
    
    def update_attention(key, attn, new_attn):
        new_attn = new_attn[:, :, -1:, :]
        if 'enc' not in key and config.use_cache:
            new_value   = new_attn[:, :, :, -1:]
            
            last_idx    = K.array([0, 0, 0, 1], 'int32') * (config.max_length - 1)
            start_idx   = K.array([0, 0, 0, 1], 'int32') * (config.init_length + state.t - 1)
            new_attn    = K.slice_update(new_attn, last_idx, K.zeros_like(new_value))
            new_attn    = K.slice_update(new_attn, start_idx, new_value)
        
        start_slice = K.array([0, 0, 1, 0], 'int32') * (
            config.start_attention_slice + state.t
        )
        return K.slice_update(attn, start_slice, new_attn)
    
    return {
        k : update_attention(k, prev_attention[k], new_attention[k])
        for k in prev_attention.keys()
    }

def update_state(state, output, finished, config, beam_index = None, first_iter = False):
    _update_hidden_state_fn = _update_hidden_state_xla if config.use_xla else _update_hidden_state_eager
    _update_mask_fn = _update_padding_mask_xla if config.use_xla else _update_padding_mask_eager
    
    mask    = state.padding_mask
    next_hidden_state = output.state
    if beam_index is not None:
        mask = K.take(mask, beam_index, axis = 0)
        if config.use_cache:
            if not isinstance(next_hidden_state, dict) or not config.is_encoder_decoder:
                next_hidden_state = tree.map_structure(
                    lambda t: K.take(t, beam_index, axis = 0) if len(K.shape(t)) > 0 else t,
                    next_hidden_state
                )
            else:
                # the encoder output states are identical for all sentences within a beam
                # therefore, no need of gathering them, which safe time and memory !
                next_hidden_state = {
                    k : (
                        (
                            K.take(v[0][0], beam_index, axis = 0),
                            K.take(v[0][1], beam_index, axis = 0)
                        ), v[1]
                    ) for k, v in next_hidden_state.items()
                }

    return InferenceState(
        t   = state.t + 1,
        state   = _update_hidden_state_fn(
            state, next_hidden_state, config = config, first_iter = first_iter
        ) if config.use_cache else (),
        finished    = finished,
        padding_mask    = _update_mask_fn(
            state, mask, finished, config = config, first_iter = first_iter
        )
    )


def _update_hidden_state_eager(state, next_hidden_state, config, first_iter = False):
    return next_hidden_state

def _update_hidden_state_xla(state, next_hidden_state, config, first_iter = False):
    if not config.is_transformer:  return next_hidden_state
    
    if first_iter:
        num_padding = config.max_length - config.init_length - 1
        padding     = [[0, 0], [0, 0], [0, num_padding], [0, 0]]
        
        new_state   = {}
        for layer, layer_state in next_hidden_state.items():
            self_state  = layer_state[0] if config.is_encoder_decoder else layer_state
            
            new_self_state  = (
                K.pad(self_state[0], padding), K.pad(self_state[1], padding)
            )
            new_state[layer] = new_self_state if not config.is_encoder_decoder else (
                new_self_state, layer_state[1]
            )
        return new_state

    start_slice = [0, 0, config.init_length - 1 + state.t, 0]
    
    new_state   = {}
    for layer, layer_state in next_hidden_state.items():
        updated = layer_state[0] if config.is_encoder_decoder else layer_state

        updated = (
            K.slice_update(updated[0][:, :, :-1, :], start_slice, updated[0][:, :, -1:, :]),
            K.slice_update(updated[1][:, :, :-1, :], start_slice, updated[1][:, :, -1:, :])
        )

        new_state[layer] = (updated, layer_state[1]) if config.is_encoder_decoder else updated
    
    return new_state

def _update_padding_mask_eager(state, mask, finished, config, first_iter = False):
    return K.concatenate([mask, ~finished[:, None]], axis = 1)

def _update_padding_mask_xla(state, mask, finished, config, first_iter = False):
    if first_iter:
        n_at_end    = 1 if config.use_cache else 0
        batch_size  = K.shape(mask)[0]
        mask    = K.concatenate([
            mask,
            K.zeros((batch_size, config.max_length - config.init_length - n_at_end), mask.dtype),
            K.ones((batch_size, n_at_end), dtype = mask.dtype)
        ], axis = 1)
    
    slice_idx   = config.init_length + state.t
    if config.use_cache: slice_idx -= 1
    return K.slice_update(
        mask, [0, slice_idx], ~finished[:, None]
    )

def _get_batch_size(* args, default = 1):
    for data in args:
        if data is not None: return len(data)
    return default

_inference_methods  = {
    'greedy'    : infer_greedy,
    'sample'    : lambda * args, ** kwargs: infer_greedy(* args, use_sampling = True, ** kwargs),
    'beam'      : infer_beam_search
}