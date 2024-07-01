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
import logging
import collections
import keras.ops as K

from keras import tree

from utils.hparams import HParams
from loggers import timer, time_logger
from utils.keras_utils import TensorSpec, ops, graph_compile
from utils.sequence_utils import pad_to_multiple
from custom_layers import get_activation, ResidualMultiHeadAttention, HParamsMHA

logger  = logging.getLogger(__name__)

TransformerOutput = collections.namedtuple(
    "TransformerOutput", [
        "output",
        "state",
        "logits",
        "attention_weights",
        "hidden_states",
        "mask"
    ]
)


_base_enc_dec_kwargs    = {
    'num_layers'    : 4,
    'normalize_output'  : False
}
_shared_config          = [
    'num_layers',
    'embedding_dim',
    
    'epsilon',
    'normalize',
    'normalize_output',
    
    'mha_num_heads',
    
    'ffn_dim',
    'ffn_use_bias',
    'ffn_use_up_proj',
    'ffn_activation',
    
    'drop_rate'
]

HParamsTransformerLayer = HParams(
    ** HParamsMHA.get_config(add_prefix = 'mha'),
    ** HParamsMHA.get_config(add_prefix = 'enc_mha'),
    embedding_dim   = 512,
    normalize   = 'after',
    epsilon     = 1e-12,
    drop_rate   = 0.1,
    mha_ffn_in_parallel = False,
    use_encoder_attention   = False,
    encoder_embedding_dim   = None,
    use_causal_attention    = False,
    ffn_dim     = 1024,
    ffn_use_bias    = True,
    ffn_activation  = 'relu',
    ffn_use_up_proj = False,
    norm_training   = True      # whether to allow `training = True` or not
)
HParamsTransformerBlock = HParamsTransformerLayer(** _base_enc_dec_kwargs)

HParamsTransformerEncoder   = HParamsTransformerBlock
HParamsTransformerDecoder   = HParamsTransformerBlock(
    use_encoder_attention = True, use_causal_attention = True
)

def set_mask(t, mask):
    if keras.backend.backend() == 'jax': return t
    try:
        t._keras_mask = mask
    except AttributeError:
        pass
    return t

def _get_state_length(state):
    """
        Returns the length of state (i.e. the 3rd dimension of any item)
        `state` is a dict of {layer_name : state(s)} meaning that the saved `state(s)` is either :
        - a tuple `(k, v)` with `{k / v}.shape == [batch_size, num_heads, seq_length, mha_depth]`
        - a tuple of tuple `((k, v), (enc_k, enc_v))` where `k` has the same shape as above
        Therefore, taking `state[0][0]` either returns : `k[0]` or `k`.
        In both cases, the dimension `-2` is the expected sequence length, it is the easiest way to get the information without taking care of the possibly nested tuple
    """
    return K.shape(keras.tree.flatten(state)[0])[-2] if state else 0

def _get_state_step(state):
    if not state: return 0
    return 1 + K.cast(K.max(K.where(
        keras.tree.flatten(state)[0][0, 0, :, 0] != 0.
    )), 'int32')

def format_output(output,
                  state     = None,
                  logits    = None,
                  attn_weights  = None,
                  hidden_states = None,
                  mask      = None,
                  types     = None,
                  
                  return_state      = False,
                  return_logits     = False,
                  return_attention  = False,
                  return_hidden_states  = False,
                  return_mask       = False,
                  return_types      = False,
                  
                  as_dict       = False,
                  ** kwargs
                 ):
    def _maybe_add(out, key, value, should_return):
        return out if value is None or not should_return else (out + (value, ))
    
    if as_dict:
        return TransformerOutput(
            output  = output,
            state   = state if return_state else None,
            logits  = logits if return_logits else None,
            attention_weights   = attn_weights if return_attention else None,
            hidden_states   = hidden_states if return_hidden_states else None,
            mask    = mask if return_mask else None
        )
    
    out = (output, )
    
    out = _maybe_add(out, 'state',          state,        should_return = return_state)
    out = _maybe_add(out, 'logits',         logits,       should_return = return_logits)
    out = _maybe_add(out, 'attention',      attn_weights, should_return = return_attention)
    out = _maybe_add(out, 'hidden_states',  hidden_states,  should_return = return_hidden_states)
    out = _maybe_add(out, 'mask',           mask,         should_return = return_mask)
    out = _maybe_add(out, 'types',          types,        should_return = return_types)
    
    return out[0] if not as_dict and len(out) == 1 else out

@timer
def build_padding_mask(seq,
                       pad_value    = 0,
                       maxlen   = None,
                       mask     = None,
                       state    = None,
                       dtype    = 'bool'
                      ):
    """
        Return padding mask matching attention shape [batch_size, 1, 1, max(lengths)]
        The mask is `False` (or 0) if the value should be masked and `True` (or 1) otherwise
    """
    if isinstance(mask, list): mask = mask[0]
    if mask is not None:
        if mask.dtype != dtype: mask = K.cast(mask, dtype)
        return mask if len(mask.shape) == 4 else mask[:, None, None, :]
    
    if len(seq.shape) == 2:
        mask = K.not_equal(seq, pad_value)
    elif len(seq.shape) == 3:
        mask = K.any(seq != K.cast(pad_value, seq.dtype), axis = -1)
    else:
        raise ValueError('Unsupported sequence shape : {}'.format(K.shape(seq)))
        
        mask = K.cond(
            K.all(mask[:, 0]),
            lambda: mask,
            lambda: K.slice_update(mask, [0, 0], K.ones((len(mask), 1), dtype = 'bool'))
        )
    
    if dtype != 'bool': mask = K.cast(mask, dtype)
    
    if state:
        mask = K.concatenate([
            K.cast(keras.tree.flatten(state)[0][:, 0, :, 0] != 0., dtype), mask
        ], axis = 1)
    
    return mask[:, None, None, :]

@timer
def build_look_ahead_mask(length, maxlen = None, dtype = 'bool'):
    """
        Creates a `look ahead` mask with shape [batch_size, 1, size, size]
        The mask is `False` (or 0) if the value should be masked and `True` (or 1) otherwise
    """
    mask = K.tril(K.ones((1, 1, length, length), dtype = dtype))
    if maxlen is not None:
        mask = K.pad(
            mask, [(0, 0), (0, 0), (0, 0), (maxlen - length, 0)], constant_values = K.ones((), dtype)
        )
    return mask

def combine_masks(padding_mask, look_ahead_mask):
    if padding_mask.dtype == 'bool':
        return K.logical_and(look_ahead_mask, padding_mask)
    return K.minimum(look_ahead_mask, padding_mask)

@timer
def build_mask(inputs,
               use_causal_attention,
               *,
               
               mask = None,
               padding_mask = None,
               look_ahead_mask  = None,
               initial_state    = None,
               embedded = None,
               
               dtype    = 'bool',
               ** kwargs
              ):
    if isinstance(mask, list): mask = mask[0]
    if mask is not None:
        if len(mask.shape) == 4: return mask
        padding_mask = mask

    batch   = K.shape(embedded if embedded is not None else inputs)[0]
    seq_len = K.shape(embedded if embedded is not None else inputs)[1]
    offset  = _get_state_length(initial_state)
    maxlen  = seq_len + offset
    
    padding_mask = build_padding_mask(
        inputs,
        mask = padding_mask,
        maxlen = maxlen,
        dtype = dtype,
        state = initial_state
    )
    
    if not use_causal_attention: return padding_mask

    if look_ahead_mask is None:
        look_ahead_mask = build_look_ahead_mask(
            seq_len, maxlen if initial_state else None, dtype = dtype
        )
    
    return combine_masks(padding_mask, look_ahead_mask)

def FeedForwardNetwork(ffn_dim,
                       activation,
                       embedding_dim,
                       use_bias     = True,
                       use_up_proj  = False,
                       name = 'ffn'
                      ):
    if isinstance(ffn_dim, float): ffn_dim = int(ffn_dim * embedding_dim)
    
    inputs = keras.layers.Input(shape = (None, embedding_dim))
    
    x = keras.layers.Dense(ffn_dim, use_bias = use_bias, name = 'dense_1')(inputs)
    x = get_activation(activation)(x)
    
    if use_up_proj:
        proj = keras.layers.Dense(ffn_dim, use_bias = use_bias, name = 'up_proj')(inputs)
        x    = keras.layers.Multiply()([proj, x])
    
    x = keras.layers.Dense(embedding_dim, use_bias = use_bias, name = 'dense_2')(x)
    model = keras.Model(inputs, x, name = name)
    model.supports_masking = True
    return model

class FeedForwardNetworkOld(keras.Model):
    def __init__(self,
                 ffn_dim,
                 ffn_activation,
                 embedding_dim,
                 use_bias   = True,
                 use_up_proj    = False,
                 name = 'ffn'
                ):
        """
            Simple 2-`Dense` sequential network with an activation function between the 2 layers.
            
            Arguments :
                - ffn_dim   : the 1st layer's number of units
                - ffn_activation    : the activation function between the 2 layers
                - embedding_dim     : the Transformers' depth (the number of units for the 2nd layer)
        """
        super().__init__(name = name)
        
        if isinstance(ffn_dim, float): ffn_dim = int(ffn_dim * embedding_dim)
        self.ffn_dim    = ffn_dim
        self.use_bias   = use_bias
        self.use_up_proj    = use_up_proj
        self.ffn_activation = ffn_activation
        self.embedding_dim  = embedding_dim
        
        self.dense_1    = keras.layers.Dense(ffn_dim, use_bias = use_bias, name = 'dense_1')
        self.up_proj    = keras.layers.Dense(ffn_dim, use_bias = use_bias, name = 'up_proj') if use_up_proj else None
        self.act        = get_activation(ffn_activation)
        self.dense_2    = keras.layers.Dense(embedding_dim, use_bias = use_bias, name = 'dense_2')
    
    def call(self, inputs, training = False):
        x = self.dense_1(inputs)
        if self.act is not None: x = self.act(x)
        if self.use_up_proj:     x = x * self.up_proj(inputs)
        return self.dense_2(x)

    def get_config(self):
        return {
            'name'  : self.name,
            'ffn_dim'   : self.ffn_dim,
            'use_bias'  : self.use_bias,
            'use_up_proj'   : self.use_up_proj,
            'ffn_activation'    : self.ffn_activation,
            'embedding_dim' : self.embedding_dim
        }

@keras.saving.register_keras_serializable('transformers')
class TransformerLayer(keras.layers.Layer):
    _attr_to_set = [
        'normalize', 'mha_ffn_in_parallel', 'norm_training', 'use_causal_attention', 'use_encoder_attention'
    ]
    
    def __init__(self,
                 embedding_dim,
                 *,
                 
                 name   = None,
                 mha_class  = ResidualMultiHeadAttention,
                 norm_class = keras.layers.LayerNormalization,
                 
                 ** kwargs
                ):
        """
            A fully customizable Transformer layer.
            It handles:
                - self-attention    : when Q = K = V
                    The 1st MHA is by default a self-attention layer
                    - In Encoder-only       : there is only 1 self-MHA
                    - In Encoder-Decoder    : there is 1 self-MHA followed by a causal-MHA
                - causal attention  : when using the masking operator
                    Set `use_causal_attention = True` in the constructor
                    The 2nd attention (if `use_encoder_attention = True`) is by default causal
                - Encoder-Decoder mode  : uses 2 MHA (a self-MHA followed by a causal-MHA)
                    Set `use_encoder_attention = True` in the constructor.
                    Note that the 2nd MHA is not a self-MHA as K and V are the `encoder_output` call argument
            
            See the `HParamsTransformerLayer` class for an exhaustive list of configuration. 
                Those starting with `ffn_` are related to the feed-forward network
                Those starting with `mha_` are related to the 1st MHA
                Those starting with `enc_mha_` are related to the 2nd MHA (ignored if `use_encoder_attention = False`)
                
                - normalize : where to apply the `LayerNormalization`
                    - before    : directly on the layer's input
                    - middle    : just before the FFN call but it does not normalize the FFN's residual !
                    `ffn_out = mha_out + norm(ffn(mha_out))` (it is used by `GPT-2` models)
                    - after     : default case where the normalization is applied on the FFN's output
                - use_causal_attention  : whether to use the masking operator or not (on the 1st MHA)
                - use_encoder_attention : whether to use 1 or 2 MHA
            
            Note that the `epsilon` and `norm_training` are propagated to the MHA
        """
        super().__init__(name = name)
        self.supports_masking   = True

        hparams = HParamsTransformerLayer
        if mha_class != ResidualMultiHeadAttention and hasattr(mha_class, 'default_params'):
            hparams = hparams(
                ** mha_class.default_params.get_config(add_prefix = 'mha'),
                ** mha_class.default_params.get_config(add_prefix = 'enc_mha')
            )
        
        self.hparams    = hparams.extract(kwargs)
        self.hparams    = self.hparams(
            embedding_dim   = embedding_dim,
            
            mha_epsilon     = self.hparams.epsilon,
            mha_output_dim  = embedding_dim,
            mha_is_cross_attention  = False,
            
            enc_mha_epsilon     = self.hparams.epsilon,
            enc_mha_output_dim  = embedding_dim,
            enc_mha_is_cross_attention  = True
        )
        if self.hparams.mha_attention_dim == -1 or self.hparams.mha_residual:
            self.hparams.mha_attention_dim = embedding_dim
        if self.hparams.enc_mha_attention_dim == -1 or self.hparams.enc_mha_residual:
            self.hparams.enc_mha_attention_dim = embedding_dim
        if self.hparams.enc_mha_num_heads == -1:
            self.hparams.enc_mha_num_heads = self.hparams.mha_num_heads
        
        for attr in self._attr_to_set:
            setattr(self, attr, self.hparams[attr])
        
        self.attention  = mha_class(
            ** self.hparams.get_config(prefix = 'mha'), norm_class = norm_class, name = 'mha'
        )
        self.enc_attention  = mha_class(
            ** self.hparams.get_config(prefix = 'enc_mha'), norm_class = norm_class, name = 'enc_mha'
        ) if self.use_encoder_attention else None
        
        self.norm   = norm_class(
            epsilon = self.hparams.epsilon, name = 'norm'
        ) if self.hparams.normalize else None
        self.dropout    = keras.layers.Dropout(self.hparams.drop_rate)
    
    def build(self, input_shape):
        if self.built: return
        super().build(input_shape)
        self.attention.build(input_shape)
        if self.enc_attention is not None:
            self.enc_attention.build((None, None, self.hparams.embedding_dim))
        
        self.norm.build((None, None, self.hparams.embedding_dim))

        with keras.name_scope('ffn'):
            self.ffn    = FeedForwardNetwork(
                ffn_dim = self.hparams.ffn_dim,
                activation  = self.hparams.ffn_activation,
                embedding_dim   = self.hparams.embedding_dim,
                use_bias    = self.hparams.ffn_use_bias,
                use_up_proj = self.hparams.ffn_use_up_proj,
                name = 'ffn'
            )

        
    def compute_mask(self, inputs, mask = None, ** kwargs):
        return build_mask(
            inputs, self.use_causal_attention, mask = mask, ** kwargs
        )

    def call(self,
             inputs,
             *,
             lengths    = None,
             encoder_output = None,
             
             initial_state  = None,
             
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             enc_padding_mask   = None,
             
             attention_kwargs   = {},
             cross_attention_kwargs = {},
             
             training   = False,
             return_state   = False,
             return_attention   = True
            ):
        """
            Arguments :
                - inputs    : the layers' input (the query) with shape [B, q_len, embedding_dim]
                - encoder_output    : encoder output with shape [B, in_seq_len, encoder_embedding_dim]
                
                - initial_state     : state to use (typically the previous iteration state)
                
                - mask  : the mask to use for the 1st MHA
                - padding_mask  : the padding mask for the 1st MHA          [B, 1, seq_len, seq_len]
                - look_ahead_mask   : the causal mask for the 1st MHA       [B, 1, 1, seq_len]
                - enc_padding_mask  : the padding mask used for the 2nd MHA [B, 1, 1, in_seq_len]
                
                - training  : whether it is training / inference phase
                - return_state      : whether to return the internal state or not
                - return_attention  : whether to return attention weights or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : self-attention weights for each head of the MHA
        """
        if self.enc_attention is not None and encoder_output is None:
            raise RuntimeError("You must provide `encoder_output` when using encoder attention")
        elif self.enc_attention is None and encoder_output is not None:
            raise RuntimeError("`encoder_output` is provided, but the layer has no cross-attention")
        
        if mask is None:
            mask = self.compute_mask(
                inputs,
                padding_mask    = padding_mask,
                look_ahead_mask = look_ahead_mask,
                initial_state   = initial_state
            )

        if self.normalize == 'before':
            inputs = self.norm(inputs, training = training)

        attn_state, enc_attn_state = None, None
        if initial_state:
            if self.enc_attention is not None:
                attn_state, enc_attn_state = initial_state
            else:
                attn_state = initial_state

        with time_logger.timer('self MHA call'):
            attn_outputs    = self.attention(
                inputs,
                mask    = mask,
                training    = training,
                initial_state   = attn_state,
                return_attention    = return_attention,
                return_state    = return_state,
                normalize_kv    = True,
                ** attention_kwargs
            )
        if not isinstance(attn_outputs, tuple): attn_outputs = (attn_outputs, )
        attn_out = attn_outputs[0]

        if self.enc_attention is not None:
            with time_logger.timer('cross MHA call'):
                enc_attn_outputs = self.enc_attention(
                    attn_out,
                    encoder_output,
                    encoder_output,
                    mask    = enc_padding_mask,
                    training    = training,
                    initial_state   = enc_attn_state,
                    return_attention    = return_attention,
                    return_state    = return_state,
                    normalize_kv    = False,
                    ** cross_attention_kwargs
                )
            attn_out = enc_attn_outputs
            if return_attention or return_state:
                attn_out    = enc_attn_outputs[0]
                attn_outputs    = tuple((o1, o2) for o1, o2 in zip(attn_outputs, enc_attn_outputs))

        if not self.mha_ffn_in_parallel:
            ffn_in = attn_out
            if self.normalize == 'middle':
                ffn_in = self.norm(ffn_in, training = training)
        else:
            ffn_in = inputs
            if self.attention.inp_norm_layer is not None:
                ffn_in = self.attention.inp_norm_layer(
                    ffn_in, training = training
                )
        
        with time_logger.timer('FFN call'):
            ffn_output  = self.ffn(ffn_in, training = training)
            ffn_output  = self.dropout(ffn_output, training = training)

        output  = ffn_output + attn_out
        
        if self.mha_ffn_in_parallel:
            output = output + inputs
        
        if self.normalize == 'after':
            output = self.norm(output, training = training and self.norm_training)
        
        set_mask(output, mask)
        return output if len(attn_outputs) == 1 else (
            (output,) + tree.map_structure(lambda t: set_mask(t, mask), attn_outputs[1:])
        )
    
    def get_output_shape(self,
                         input_shape,
                         encoder_output = None,
                         return_state   = False,
                         return_attention   = True,
                         ** _
                        ):
        attn_out_shape    = self.attention.get_output_shape(
            input_shape, input_shape, input_shape,
            return_attention = return_attention, return_state = return_state
        )
        
        if self.enc_attention is not None:
            if encoder_output is None:
                raise ValueError("You must provide encoder output when using encoder attention !")
            
            enc_attn_out_shape = self.enc_attention.get_output_shape(
                input_shape, encoder_output, encoder_output,
                return_attention = return_attention, return_state = return_state
            )
            if return_attention or return_state:
                attn_out_shape  = (enc_attn_out_shape[0], ) + tuple(
                    (o1, o2) for o1, o2 in zip(attn_out_shape, enc_attn_out_shape)
                )[1:]
        elif encoder_output is not None:
            raise ValueError(
                "You cannot pass `encoder_output` when `self.use_encoder_attention` is False !"
            )
        
        return attn_out_shape
    
    def get_config(self):
        return (self.hparams + super().get_config()).get_config()

@keras.saving.register_keras_serializable('transformers')
class TransformerBlock(keras.Model):
    default_params  = HParamsTransformerBlock
    _attr_to_set    = [
        'embedding_dim', 'norm_training', 'use_causal_attention'
    ]
    
    def __init__(self, embedding_dim, num_layers, name = None, ** kwargs):
        """ Simply a list of `num_layers` TransformerLayer applied sequentially """
        super().__init__(name = name)
        
        kwargs.update({'embedding_dim' : embedding_dim, 'num_layers' : num_layers})
        self.hparams    = self.default_params.extract(kwargs)
        remaining_kwargs    = {k : v for k, v in kwargs.items() if k not in self.hparams}
        
        for attr_name in self._attr_to_set:
            setattr(self, attr_name, self.hparams[attr_name])
        
        self._init_input_layers(** kwargs)
        
        self.transformer_layers = [TransformerLayer(
            name = 'layer_{}'.format(i), ** {** remaining_kwargs, ** self.hparams}
        ) for i in range(self.hparams.num_layers)]
        
        self.norm       = kwargs.get('norm_class', keras.layers.LayerNormalization)(
            epsilon = self.hparams.epsilon, name = 'norm_final'
        ) if self.hparams.normalize_output else None
    
    def build(self, input_shape):
        if self.built: return
        super(TransformerBlock, self).build(input_shape)
        for layer in self.transformer_layers: layer.build((None, None, self.embedding_dim))
        if self.norm is not None: self.norm.build((None, None, self.embedding_dim))
        
    @property
    def output_dim(self):
        return self.embedding_dim
    
    def _init_input_layers(self, ** kwargs):
        pass

    def __len__(self):
        return len(self.transformer_layers)
    
    def __getitem__(self, idx):
        return self.transformer_layers[idx]
    
    def freeze(self, trainable = False):
        """ Set all `self.transformer_layers.trainable` to `trainable` """
        for layer in self.transformer_layers: layer.trainable = trainable
    
    def prepare_input(self, inputs, *, additional_inputs = (), ** kwargs):
        return inputs
    
    def compute_output(self, output, training = False, ** kwargs):
        if self.norm is not None:
            output = self.norm(output, training = training)
        
        return output
    
    def call(self,
             inputs,
             *,
             
             lengths    = None,
             encoder_output = None,
             initial_state  = None,
             
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             enc_padding_mask   = None,
             
             training   = False,
             attention_kwargs   = None,
             
             first_layer_idx    = None,
             last_layer_idx     = None,
             
             return_state       = False,
             return_attention   = False,
             return_last_attention  = False,
             return_only_cross_attention    = False,
             return_last_hidden_states  = False,
             return_hidden_states   = False,
             return_mask        = False,
             as_dict    = False,
             
             ** kwargs
            ):
        """
            See the TransformerLayer for more information
            
            Arguments :
                - inputs    : block inputs with shape [batch_size, seq_len, embedding_dim], embedded inputs
                - mask      : attention mask (padding mask based in inputs)
                - training  : whether it is training / inference phase
                - return_attention  : whether to return attention weights or not
                - return_states     : whether to return intermediate representation or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : dict self-attention weights for each head of the MHA of each layer
        """
        if last_layer_idx is None:  last_layer_idx = len(self.transformer_layers)
        if attention_kwargs is None: attention_kwargs = {}
        
        states              = {} if return_state else None
        attention_weights   = {} if return_attention or return_last_attention else None
        hidden_states       = {} if return_hidden_states or return_last_hidden_states else None

        additional_inputs   = []
        if isinstance(inputs, (list, tuple)):
            inputs, additional_inputs = inputs[0], inputs[1:]
        
        output = inputs
        if first_layer_idx is None:
            first_layer_idx = 0
            output = self.prepare_input(
                output,
                lengths = lengths,
                initial_state   = initial_state,
                additional_inputs   = additional_inputs,
                attention_kwargs    = attention_kwargs,
                
                training    = training,
                padding_mask    = padding_mask,
                mask    = mask,
                ** kwargs
            )
            if hasattr(output, '_keras_mask'):  padding_mask = output._keras_mask

        mask = self[0].compute_mask(
            inputs,
            embedded    = output,
            padding_mask    = padding_mask,
            look_ahead_mask = look_ahead_mask,
            initial_state   = initial_state
        )

        for i, layer in enumerate(self.transformer_layers[first_layer_idx : last_layer_idx], start = first_layer_idx):
            output, state, attn_weights = layer(
                output,
                lengths = lengths,
                encoder_output  = encoder_output,
                initial_state   = initial_state.get(layer.name, None) if initial_state else None,
                attention_kwargs    = attention_kwargs,
                
                training    = training,
                
                mask    = mask,
                enc_padding_mask    = enc_padding_mask,
                
                return_attention    = True,
                return_state    = True,
            )
            if return_state:
                states[layer.name] = state
            
            if return_attention or (return_last_attention and i == len(self) - 1):
                if layer.enc_attention is None:
                    attention_weights['attn_{}'.format(layer.name)] = attn_weights
                else:
                    if not return_only_cross_attention:
                        attention_weights['attn_{}'.format(layer.name)] = attn_weights[0]
                    attention_weights['enc_attn_{}'.format(layer.name)] = attn_weights[1]
            
            if return_hidden_states or (return_last_hidden_states and i == len(self) - 1):
                hidden_states['state_{}'.format(layer.name)] = output
        
        if last_layer_idx >= len(self.transformer_layers):
            output = self.compute_output(
                output,
                mask    = mask,
                training    = training,
                inputs  = inputs,
                ** kwargs
            )
        
        return format_output(
            output,
            state   = states,
            attn_weights    = attention_weights,
            hidden_states   = hidden_states,
            mask    = mask,
            
            return_state        = return_state,
            return_attention    = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states or return_last_hidden_states,
            return_mask         = return_mask,
            as_dict = as_dict
        )
    
    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import (
            _transformer_patterns, name_based_partial_transfer_learning
        )
        kwargs.setdefault('patterns', _transformer_patterns)

        return name_based_partial_transfer_learning(self, pretrained, ** kwargs)

    def get_output_shape(self,
                         inputs,
                         encoder_output = None,
                         return_state   = None,
                         return_attention   = None,
                         return_last_attention  = None,
                         return_only_cross_attention    = None,
                         return_last_hidden_states  = None,
                         return_hidden_states   = None,
                         return_mask        = None,
                         as_dict    = False,
                         ** _
                        ):
        output_shape    = inputs[:-1] + (self.output_dim, )
        
        mask_shape  = None
        
        states_shape              = collections.OrderedDict() if return_state else None
        attention_weights_shape   = collections.OrderedDict() if return_attention or return_last_attention else None
        hidden_states_shape       = collections.OrderedDict() if return_hidden_states or return_last_hidden_states else None
        
        output = inputs
        for i, layer in enumerate(self.transformer_layers):
            output, state, attn_weights = layer.get_output_shape(
                output,
                encoder_output  = encoder_output,
                return_attention    = True,
                return_state        = True
            )
            if return_state:
                states_shape[layer.name] = state
            
            if return_attention or (return_last_attention == True and i == len(self) - 1):
                if layer.enc_attention is None:
                    attention_weights_shape['attn_{}'.format(layer.name)] = attn_weights
                else:
                    if not return_only_cross_attention:
                        attention_weights_shape['attn_{}'.format(layer.name)] = attn_weights[0]
                    attention_weights_shape['enc_attn_{}'.format(layer.name)] = attn_weights[1]
            
            if return_hidden_states or (return_last_hidden_states and i == len(self) - 1):
                hidden_states_shape['state_{}'.format(layer.name)] = output
        
        return format_output(
            output_shape,
            state   = states_shape,
            attn_weights    = attention_weights_shape,
            hidden_states   = hidden_states_shape,
            mask    = mask_shape,
            
            return_state        = return_state,
            return_attention    = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states or return_last_hidden_states,
            return_mask         = return_mask,
            as_dict = as_dict
        )
    
    def get_config(self):
        return self.hparams.get_config()
    

@keras.saving.register_keras_serializable('transformers')
class TransformerEncoder(TransformerBlock):
    default_params = HParamsTransformerEncoder

@keras.saving.register_keras_serializable('transformers')
class TransformerDecoder(TransformerBlock):
    default_params = HParamsTransformerDecoder

@keras.saving.register_keras_serializable('transformers')
class Transformer(keras.Model):
    encoder_class   = TransformerEncoder
    decoder_class   = TransformerDecoder
    
    _shared_keys    = _shared_config
    _attr_to_set    = []
    
    @classmethod
    @property
    def default_params(cls):
        return HParams(
            ** cls.encoder_class.default_params.get_config(add_prefix = 'encoder'),
            ** cls.decoder_class.default_params.get_config(add_prefix = 'decoder'),
            ** {k : None for k in cls._shared_keys}
        )
    
    def __init__(self,
                 name = None,
                 shared_layers = {},
                 
                 encoder    = None,
                 encoder_wrapper = None,
                 encoder_wrapper_params = None,
                 
                 decoder    = None,
                 decoder_wrapper = None,
                 decoder_wrapper_params = None,
                 
                 ** kwargs
                ):
        super().__init__(name = name)
        
        if encoder is not None: self.encoder_class = encoder.__class__
        if decoder is not None: self.decoder_class = decoder.__class__
        
        # Init the default parameters`
        default_params  = self.default_params
        # Maybe adds parameters for wrappers (if any)
        if encoder_wrapper is None: encoder_wrapper = lambda x, ** kwargs: x
        elif encoder_wrapper_params is not None or hasattr(encoder_wrapper, 'default_params'):
            if encoder_wrapper_params is None:
                encoder_wrapper_params = encoder_wrapper.default_params
            default_params = default_params(
                ** encoder_wrapper_params.get_config(add_prefix = 'encoder')
            )
        
        if decoder_wrapper is None: decoder_wrapper = lambda x, ** kwargs: x
        elif decoder_wrapper_params is not None or hasattr(decoder_wrapper, 'default_params'):
            if decoder_wrapper_params is None:
                decoder_wrapper_params = decoder_wrapper.default_params
            default_params = default_params(
                ** decoder_wrapper_params.get_config(add_prefix = 'decoder')
            )
        
        self.hparams = default_params.extract(kwargs)
        # Allow to have different embedding dim for encoder and decoder
        _shared = {}
        for k in self._shared_keys:
            if self.hparams[k] is not None:
                _shared.update({
                    'encoder_{}'.format(k) : self.hparams[k],
                    'decoder_{}'.format(k) : self.hparams[k]
                })
        self.hparams.update(_shared)
        
        for attr_name in self._attr_to_set:
            setattr(self, attr_name, self.hparams[attr_name])
        
        # Effectively builds the encoder and decoder classes (with possible wrappers)
        if encoder is None:
            encoder = self.encoder_class(
                ** self.hparams.get_config(prefix = 'encoder'), ** shared_layers, name = 'encoder'
            )
        self.encoder = encoder_wrapper(encoder, ** self.hparams.get_config(prefix = 'encoder'))
        
        if decoder is None:
            decoder = self.decoder_class(
                ** self.hparams.get_config(prefix = 'decoder'), ** shared_layers, name = 'decoder'
            )
        self.decoder = decoder_wrapper(decoder, ** self.hparams.get_config(prefix = 'decoder'))
    
    def build(self, input_shape):
        if self.built: return
        super().build(input_shape)
        enc_shape, dec_shape = input_shape if isinstance(input_shape[0], (list, tuple)) else (input_shape, input_shape)
        self.encoder.build(enc_shape)
        self.decoder.build(enc_shape)
    
    def encode(self,
               inputs,
               *,
               
               mask = None,
               training = False,
               
               return_state    = False,
               return_attention    = False,
               return_hidden_states    = False,
               return_mask     = False,
               as_dict     = True,
               
               ** kwargs
              ):
        return self.encoder(
            inputs,
            
            mask    = mask,
            training    = training,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = return_mask,
            as_dict     = as_dict,
            
            ** kwargs
        )

    def call(self,
             inputs,
             decoder_input  = None,
             initial_state  = None,
             
             training   = False,
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             enc_padding_mask   = None,
             
             return_state       = False,
             return_attention   = False,
             return_last_attention  = False,
             return_hidden_states   = False,
             return_mask        = False,
             as_dict    = False,
             
             ** kwargs
            ):
        encoder_input = inputs
        if isinstance(inputs, (list, tuple)) and decoder_input is None:
            encoder_input, decoder_input = inputs
        
        encoder_outputs = self.encoder(
            encoder_input,

            mask    = enc_padding_mask,
            training    = training,

            return_state    = False,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = True,
            as_dict = True,

            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('encoder_')}
        )
        
        with time_logger.timer('decoder call'):
            decoder_outputs = self.decoder(
                decoder_input,
                initial_state   = initial_state,

                encoder_output  = encoder_outputs.output,
                enc_padding_mask    = encoder_outputs.mask,

                mask    = mask,
                training    = training,
                padding_mask    = padding_mask,
                look_ahead_mask = look_ahead_mask,

                return_state    = return_state,
                return_attention    = return_attention,
                return_last_attention   = return_last_attention,
                return_hidden_states    = return_hidden_states,
                return_mask     = return_mask,
                as_dict     = True,

                ** {k : v for k, v in kwargs.items() if not k.startswith('encoder_')}
            )

        return format_output(
            decoder_outputs.output,
            state   = decoder_outputs.state,
            attn_weights    = (encoder_outputs.attention_weights, decoder_outputs.attention_weights),
            hidden_states   = (encoder_outputs.hidden_states, decoder_outputs.hidden_states),
            mask    = (encoder_outputs.mask, decoder_outputs.mask),
            
            return_state            = return_state,
            return_attention        = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = return_mask,
            as_dict = as_dict
        )
    
    @timer(name = 'Transformer inference')
    @graph_compile(
        reduce_retracing = True, support_xla = True, follow_type_hints = True, cast_kwargs = False,
        prepare_for_xla = lambda self, * args, ** kwargs: (self.prepare_for_xla(
            * args, ** kwargs
        ) if hasattr(self, 'prepare_for_xla') else ((self, ) + args, kwargs))
    )
    def infer(self,
              inputs    : TensorSpec(),
              *,
              
              tokens    : TensorSpec(shape = (None, None), dtype = 'int32') = None,
              initial_state : TensorSpec() = None,

              enc_padding_mask  : TensorSpec() = None,
              padding_mask  : TensorSpec() = None,
              training  = False,
              
              flatten   = False,
              flat_length   : int = None,
              return_state  = False,
              return_attention  = False,
              return_last_attention = False,
              return_hidden_states  = False,
              return_mask   = False,
              as_dict       = True,

              ** kwargs
             ):
        encoder_outputs = self.encoder(
            inputs,
            
            mask    = enc_padding_mask,
            training    = training,
            
            return_state    = False,
            return_attention    = False,
            return_hidden_states    = False,
            return_mask     = True,
            as_dict     = True,
            
            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('encoder_')}
        )
        encoded, mask = encoder_outputs.output, encoder_outputs.mask
        if flatten:
            raise NotImplementedErr()
            encoded = K.boolean_mask(
                K.reshape(encoded, [-1, self.encoder.embedding_dim]),
                K.reshape(mask, [-1])
            )[None]
            if flat_length is not None: encoded = encoded[:, : flat_length]
            mask    = None

        return self.decoder.infer(
            encoder_output  = encoded,
            enc_padding_mask    = mask,
            
            training    = training,
            initial_state   = initial_state,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = return_mask,
            
            ** {k : v for k, v in kwargs.items() if not k.startswith('encoder_')}
        )
    
    def get_config(self):
        if type(self) is Transformer:
            return {
                'encoder'   : keras.saving.serialize_keras_object(self.encoder),
                'decoder'   : keras.saving.serialize_keras_object(self.decoder)
            }
        return self.hparams.get_config()

    @classmethod
    def from_config(cls, config, custom_objects = None):
        if 'encoder' in config and 'decoder' in config:
            config.update({
                'encoder'   : keras.saving.deserialize_keras_object(config['encoder']),
                'decoder'   : keras.saving.deserialize_keras_object(config['decoder'])
            })
        return cls(** config)
