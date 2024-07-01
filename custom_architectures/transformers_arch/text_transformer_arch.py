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
import keras.ops as K

from functools import partial

from loggers import timer
from utils.hparams import HParams
from custom_layers import CustomEmbedding
from utils.keras_utils import TensorSpec, ops, graph_compile
from custom_architectures.generation_utils import infer as infer_method
from .transformer_arch import *

logger = logging.getLogger(__name__)

HParamsTransformerTokenEmbedding = HParams(
    vocab_size  = None,
    embedding_dim   = None,
    max_input_length    = None,
    max_token_types     = 0,
    
    scale_embeddings   = False,
    normalize_embeddings   = True,
    
    repeat_position     = None,
    positional_offset   = None,

    epsilon     = 1e-6,
    drop_rate   = 0.1
)

_shared_config = [
    'vocab_size', 'sos_token', 'eos_token', 'pad_token', 'max_input_length',
    'scale_embedding', 'normalize_embeddings', 'positional_offset'
]

HParamsTextTransformerBlock = HParamsTransformerBlock(
    ** HParamsTransformerTokenEmbedding,
    sos_token   = -1,
    eos_token   = -1,
    pad_token   = 0
)
HParamsTextTransformerEncoder = HParamsTextTransformerBlock(** HParamsTransformerEncoder)
HParamsTextTransformerDecoder = HParamsTextTransformerBlock(** HParamsTransformerDecoder)

@keras.saving.register_keras_serializable('transformers')
class TransformerTokenEmbedding(keras.layers.Layer):
    _attr_to_set = [
        'embedding_dim', 'vocab_size', 'scale_embeddings', 'positional_offset', 'repeat_position'
    ]
    
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 max_input_length,
                 
                 token_embedding    = None,
                 positional_embedding   = None,
                 
                 name = 'embeddings',
                 ** kwargs
                ):
        super().__init__(name = name)
        
        self.hparams = HParamsTransformerTokenEmbedding.extract(kwargs)
        self.hparams = self.hparams(
            vocab_size      = vocab_size,
            embedding_dim   = embedding_dim,
            max_input_length    = max_input_length
        )
        
        for attr_name in self._attr_to_set:
            setattr(self, attr_name, self.hparams[attr_name])
        
        if self.repeat_position == -1: self.repeat_position = None
        self._max_input_length  = self.hparams.max_input_length
        self.embedding_factor   = K.sqrt(float(embedding_dim)) if self.scale_embeddings else None
        
        # Set token embedding layer
        if token_embedding is None:
            token_embedding = CustomEmbedding(
                self.vocab_size, self.embedding_dim, name = 'token_embedding'
            )
        
        self.token_embedding_layer = token_embedding
        
        # Set token type embedding layer (if required)
        self.token_type_embedding_layer = None
        if self.hparams.max_token_types:
            self.token_type_embedding_layer = keras.layers.Embedding(
                self.hparams.max_token_types, self.embedding_dim, name = "token_type_embedding"
            )
        
        # Set positional embedding layer
        if positional_embedding is None and self.max_input_length > 1:
            positional_embedding    = keras.layers.Embedding(
                self.max_input_length, self.embedding_dim, name = "pos_embeddings"
            )
        self.pos_embedding_layer    = positional_embedding
        
        # Set normalization layer
        self.norm       = keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_embedding'
        ) if self.hparams.normalize_embeddings else None
        self.dropout    = keras.layers.Dropout(
            self.hparams.drop_rate
        ) if self.hparams.drop_rate > 0. else None

    def build(self, input_shape):
        self.token_embedding_layer.build((None, None))
        if self.pos_embedding_layer is not None:
            self.pos_embedding_layer.build((None, None))
        if self.token_type_embedding_layer is not None:
            self.token_type_embedding_layer.build((None, None))
        if self.norm is not None:
            self.norm.build((None, None, self.embedding_dim))
        super().build(input_shape)
        
    def change_vocabulary(self, vocab, ** kwargs):
        self.vocab_size = len(vocab)
        self.hparams.vocab_size = len(vocab)
        self.token_embedding_layer.change_vocabulary(vocab, ** kwargs)
    
    @property
    def max_input_length(self):
        if not self.positional_offset: return self._max_input_length
        return self._max_input_length + self.positional_offset

    @property
    def use_token_type(self):
        return self.token_type_embedding_layer is not None

    @timer
    def linear(self, output):
        return K.matmul(output, K.transpose(self.token_embedding_layer.embeddings))

    @timer
    def embed_tokens(self, tokens):
        embeddings = self.token_embedding_layer(tokens)
        if self.scale_embeddings:
            embeddings = embeddings * K.cast(self.embedding_factor, embeddings.dtype)
        return embeddings
    
    @timer
    def embed_token_types(self, token_types, batch_size, seq_len):
        token_type_embedded = 0.
        if self.token_type_embedding_layer is not None:
            if token_types is None:
                token_types = K.zeros((batch_size, seq_len), dtype = 'int32')
            elif len(K.shape(token_types)) == 0:
                token_types = K.full((batch_size, seq_len), value = token_types, dtype = 'int32')
            token_type_embedded = self.token_type_embedding_layer(token_types)
        return token_type_embedded
    
    @timer
    def embed_positions(self, seq_len, offset = None):
        if self.pos_embedding_layer is None: return 0
        
        if self.repeat_position:
            position_ids = K.repeat(
                K.arange(seq_len // self.repeat_position + 1), self.repeat_position
            )[:seq_len]
        else:
            position_ids = K.arange(seq_len)

        position_ids = K.expand_dims(position_ids, axis = 0)
        if offset is not None:
            position_ids = position_ids + offset
        if self.positional_offset:
            position_ids = position_ids + self.positional_offset
        
        if logger.isEnabledFor(logging.DEBUG) and keras.backend.backend() == 'tensorflow':
            import tensorflow as tf
            tf.print("Position ids :", position_ids)
        
        return self.pos_embedding_layer(position_ids)
    
    def call(self,
             inputs,
             *,
             
             prefix = None,
             
             mask   = None,
             training   = False,
             
             offset     = None,
             token_types    = None
            ):
        if prefix is not None and offset is not None:
            raise RuntimeError('When `offset` is provided, `prefix` must be None')
        
        tokens = inputs
        
        if logger.isEnabledFor(logging.DEBUG) and keras.backend.backend() == 'tensorflow':
            import tensorflow as tf
            if tokens is not None:
                tf.print("Tokens shape :", tf.shape(tokens))
            tf.print("Positional offset :", tf.reshape(offset, [-1]))
        
        if tokens is not None:
            token_embedded = self.embed_tokens(tokens)

            if prefix is not None:
                token_embedded = K.concatenate([prefix, token_embedded], axis = 1)
        else:
            token_embedded = prefix
        
        # Embed token types (if necessary)
        token_type_embedded = self.embed_token_types(
            token_types, K.shape(token_embedded)[0], K.shape(token_embedded)[1]
        )
        
        # Embed positions 
        pos_embedded = self.embed_positions(
            K.shape(token_embedded)[1], offset = offset
        )
        
        if logger.isEnabledFor(logging.DEBUG) and keras.backend.backend() == 'tensorflow':
            import tensorflow as tf
            tf.print('token embed shape :', tf.shape(token_embedded), 'pos shape :', tf.shape(pos_embedded), 'type shape :', tf.shape(token_type_embedded))
        
        # Combine all embeddings
        embeddings  = token_embedded + pos_embedded + token_type_embedded

        if self.norm is not None:
            embeddings = self.norm(embeddings, training = training)
        if self.dropout is not None:
            embeddings = self.dropout(embeddings, training = training)

        if keras.backend.backend() != 'jax':
            embeddings._keras_mask = mask
        
        return embeddings

    def get_output_shape(self, inputs):
        return tuple(inputs) + (self.embedding_dim, )
    
    def get_config(self):
        return (self.hparams + super().get_config()).get_config()

@keras.saving.register_keras_serializable('transformers')
class TextTransformerBlock(TransformerBlock):
    """ Regular `TransformerBlock` with a `TransformerTokenEmbedding` layer applied on inputs """
    default_params  = HParamsTextTransformerBlock
    _attr_to_set    = TransformerBlock._attr_to_set + ['vocab_size', 'positional_offset']
    
    def __init__(self, vocab_size, embedding_dim, max_input_length, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim,
            max_input_length = max_input_length, ** kwargs
        )

        self.sos_token  = self.hparams.sos_token
        self.eos_token  = self.hparams.eos_token
        self.pad_token  = self.hparams.pad_token
    
    def _init_input_layers(self,
                           token_embedding = None,
                           positional_embedding = None,
                           ** kwargs
                          ):
        self.embeddings = TransformerTokenEmbedding(
            token_embedding     = token_embedding,
            positional_embedding    = positional_embedding,
            name    = 'embeddings',
            ** self.hparams
        )
    
    def change_vocabulary(self, vocab, ** kwargs):
        self.vocab_size = len(vocab)
        self.hparams.vocab_size = len(vocab)
        self.embeddings.change_vocabulary(vocab, ** kwargs)

    def build(self, input_shape):
        super(TextTransformerBlock, self).build((None, None, self.embedding_dim))
        self.embeddings.build((None, None))
        
    @property
    def max_input_length(self):
        return self.embeddings.max_input_length
    
    @property
    def use_token_type(self):
        return self.embeddings.use_token_type
    
    @property
    def pad_value(self):
        return self.pad_token
    
    def set_tokens(self, sos_token = None, eos_token = None, pad_token = None):
        if sos_token not in (-1, None): self.sos_token = sos_token
        if eos_token not in (-1, None): self.eos_token = eos_token
        if pad_token not in (-1, None): self.pad_token = pad_token

        self.hparams.update({
            'sos_token' : self.sos_token,
            'eos_token' : self.eos_token,
            'pad_token' : self.pad_token
        })

    def embed_tokens(self, tokens):
        return self.embeddings.embed_tokens(tokens)
    
    def prepare_input(self,
                      inputs,
                      *,
                      
                      lengths   = None,
                      token_types   = None,
                      initial_state = None,
                      additional_inputs = [],

                      mask  = None,
                      training  = False,
                      padding_mask  = None,
             
                      prefix     = None,
                      offset    = None,
                      
                      ** kwargs
                     ):
        tokens = inputs
        if len(additional_inputs) > 0:
            if not self.use_token_type:
                raise RuntimeError('Found additional inputs while `self.use_token_type = False`')
            
            token_types     = additional_inputs[0]
        
        if mask is None and padding_mask is not None: mask = padding_mask
        if tokens is not None:
            mask = build_padding_mask(
                tokens, mask = mask, pad_value = self.pad_token, dtype = 'bool'
            )

        if offset is None and initial_state: offset = lengths[:, None] - 1
        
        embedded = self.embeddings(
            tokens,
            token_types     = token_types,
            
            prefix  = prefix,
            offset  = offset,

            training    = training,
            mask    = mask
        )
        
        return embedded

    def compute_output(self, output, training = False, mask = None, prefix = None, ** kwargs):
        output = super(TextTransformerBlock, self).compute_output(
            output, training = training, mask = mask, ** kwargs
        )
        if prefix is not None:
            output = output[:, K.shape(prefix)[1] - 1 :]
        
        return output

    def infer(self, * args, ** kwargs):
        return infer_method(self, * args, is_transformer = True, ** kwargs)
    
    def transfer_weights(self, * args, ** kwargs):
        kwargs.setdefault('skip_layers', ('sos_token', 'eos_token', 'pad_token'))
        return super(TextTransformerBlock, self).transfer_weights(* args, ** kwargs)
    
    def get_output_shape(self, inputs, * args, ** kwargs):
        return super().get_output_shape(
            self.embeddings.get_output_shape(inputs), * args, ** kwargs
        )

@keras.saving.register_keras_serializable('transformers')
class TextTransformerEncoder(TextTransformerBlock):
    default_params = HParamsTextTransformerEncoder

@keras.saving.register_keras_serializable('transformers')
class TextTransformerDecoder(TextTransformerBlock):
    default_params = HParamsTextTransformerDecoder

@keras.saving.register_keras_serializable('transformers')
class TextTransformer(Transformer):
    encoder_class   = TextTransformerEncoder
    decoder_class   = TextTransformerDecoder
    _shared_keys    = Transformer._shared_keys + _shared_config
    
    def set_tokens(self, ** kwargs):
        """
            Call `set_tokens` on both the `encoder` and the `decoder`
            If you want to specify custom tokens for encoder / decoder, simply prefix its name by `encoder_` (or `decoder_`)
            For instance `set_tokens(encoder_pad_token = 0, pad_token = 1)` will set `pad_token` to 0 and 1 respectively for the encoder / decoder
        """
        if hasattr(self.encoder, 'set_tokens'):
            self.encoder.set_tokens(** {
                ** kwargs, ** {k[8:] : v for k, v in kwargs.items() if k.startswith('encoder_')}
            })
        if hasattr(self.decoder, 'set_tokens'):
            self.decoder.set_tokens(** {
                ** kwargs, ** {k[8:] : v for k, v in kwargs.items() if k.startswith('decoder_')}
            })

    def prepare_for_xla(self, inputs, * args, mask = None, padding_multiple = 256, ** kwargs):
        inputs = pad_to_multiple(
            inputs, padding_multiple, axis = 1,
            constant_values = getattr(self.encoder, 'pad_value', 0.)
        )
        if mask is not None:
            kwargs['mask'] = pad_to_multiple(
                mask, padding_multiple, axis = 1, constant_values = False
            )
        
        if hasattr(self.decoder, 'prepare_for_xla'):
            _, kwargs = self.decoder.prepare_for_xla(** kwargs)
        return (self, inputs) + args, kwargs
