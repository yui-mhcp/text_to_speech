import tensorflow as tf

from custom_layers import get_activation, MultiHeadAttention, HParamsMHA

from hparams.hparams import HParams
from utils.text.text_processing import create_look_ahead_mask

_base_layer_kwargs    = {
    'embedding_dim' : 512,
    'epsilon'   : 1e-12,
    'drop_rate' : 0.1,
    'ffn_dim'   : 1024,
    'ffn_activation'    : 'relu'
}

_base_enc_dec_kwargs    = {
    'num_layers'    : 4,
    'return_attention'  : True
}

HParamsTransformerEncoderLayer = HParams(
    ** HParamsMHA.get_config(add_prefix = 'mha'),
    ** _base_layer_kwargs
)
HParamsTransformerDecoderLayer = HParams(
    ** HParamsMHA.get_config(add_prefix = 'mha'),
    ** HParamsMHA.get_config(add_prefix = 'enc_mha'),
    ** _base_layer_kwargs
)

HParamsTransformerEncoder = HParamsTransformerEncoderLayer(
    ** _base_enc_dec_kwargs
)
HParamsTransformerDecoder = HParamsTransformerDecoderLayer(
    ** _base_enc_dec_kwargs
)

HParamsTransformer  = HParams(
    ** HParamsTransformerEncoder.get_config(add_prefix = 'encoder'),
    ** HParamsTransformerDecoder.get_config(add_prefix = 'decoder')
)

def FeedForwardNetwork(ffn_dim, ffn_activation, embedding_dim):
    act_layer = get_activation(ffn_activation)
    
    ffn = tf.keras.Sequential(name = "ffn")
    ffn.add(tf.keras.layers.Dense(ffn_dim, name = 'ffn_dense_1'))
    if act_layer is not None: ffn.add(act_layer)
    ffn.add(tf.keras.layers.Dense(embedding_dim, name = 'ffn_dense_2'))
    return ffn

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams    = HParamsTransformerEncoderLayer.extract(kwargs)
        self.hparams    = self.hparams(embedding_dim = embedding_dim, mha_attention_dim = embedding_dim)
        
        self.attention  = MultiHeadAttention(** self.hparams.get_config(prefix = 'mha'), name = 'mha')
        
        self.ffn = FeedForwardNetwork(self.hparams.ffn_dim, self.hparams.ffn_activation, embedding_dim)
        
        self.norm   = tf.keras.layers.LayerNormalization(epsilon = self.hparams.epsilon)
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate)
    
    def call(self, inputs, mask = None, training = False, return_attention = True):
        """
            Arguments :
                - inputs    : encoder inputs with shape [batch_size, seq_len, embedding_dim], embedded inputs
                - mask      : attention mask (padding mask based in inputs)
                - training  : whether it is training / inference phase
                - return_attention  : whether to return attention weights or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : self-attention weights for each head of the MHA
            
            Pipeline : 
            1) `inputs` is passed in the MHA layer (as query, key and value) : self-attention
            2) The attention output is passed in the point-wise feed-forward network (2 Dense layers)
            3) Normalization of the sum `attention_output + ffn_output`
        """
        attn_out, attn_weights = self.attention(
            inputs, inputs, inputs, mask = mask, training = training, return_attention = True
        )
        
        ffn_output  = self.ffn(attn_out, training = training)
        ffn_output  = self.dropout(ffn_output, training = training)
        
        output = self.norm(attn_out + ffn_output, training = training)
    
        return (output, attn_weights) if return_attention else output
    
    def get_config(self):
        config = super().get_config()
        return (self.hparams + config).get_config()
    
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams    = HParamsTransformerDecoderLayer.extract(kwargs)
        self.hparams    = self.hparams(
            embedding_dim       = embedding_dim,
            mha_attention_dim   = embedding_dim,
            enc_mha_attention_dim   = embedding_dim
        )
        
        self.attention  = MultiHeadAttention(** self.hparams.get_config(prefix = 'mha'), name = 'mha')
        self.enc_attention  = MultiHeadAttention(
            ** self.hparams.get_config(prefix = 'enc_mha'), name = 'enc_mha'
        )
        
        self.ffn = FeedForwardNetwork(self.hparams.ffn_dim, self.hparams.ffn_activation, embedding_dim)
        
        self.norm       = tf.keras.layers.LayerNormalization(epsilon = self.hparams.epsilon)
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate)
    
    def call(self,
             inputs,
             training   = False,
             enc_padding_mask   = None,
             look_ahead_mask    = None,
             return_attention   = True
            ):
        """
            Arguments :
                - inputs    : list of [encoder_output, decoder_input]
                    - encoder_output    : encoder output with shape [batch_size, seq_len_in, encoder_embedding_dim]
                    - decoder_input     : decoder input with shape [batch_size, seq_len_out, embedding_dim]
                - enc_padding_mask  : mask for encoder input padding (passed to the 2nd attention block)
                - look_ahead_mask   : mask for decoder_input look_ahead (+ padding (optionnal))
                - training  : whether it is training / inference phase
                - return_attention  : whether to return attention weights or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : self-attention weights for each head of the MHA
            
            Pipeline : 
            1) `decoder_input` is passed in the 1st MHA layer (as query, key and value) with the look_ahead_mask : self-attention
            2) Normalization of the sum of attention_output (1st block) + decoder_inputs
            3) 2nd attention block with `encoder_output` as key and value, and `normalized attention output` as query (with the `padding_mask`) : encoder attention
            4) Pass the output of the 2nd block to the FFN network (2 Dense layers)
            5) Normalize the sum of `enc_attn_output` and `ffn_output`
        """
        encoder_output, target = inputs
        
        if look_ahead_mask is None:
            look_ahead_mask = create_look_ahead_mask(
                tf.shape(target)[0], tf.shape(target)[1], tf.float32
            )
        
        target_attn, target_attn_weights  = self.attention(
            target, target, target, mask = look_ahead_mask, training = training
        )
        
        enc_attn, enc_attn_weights = self.enc_attention(
            target_attn, encoder_output, encoder_output, mask = enc_padding_mask, training = training
        )
        
        
        ffn_output  = self.ffn(enc_attn, training = training)
        ffn_output  = self.dropout(ffn_output, training = training)
        
        output = self.norm(enc_attn + ffn_output, training = training)
    
        return (output, target_attn_weights, enc_attn_weights) if return_attention else output
    
    def get_config(self):
        config = super().get_config()
        return (self.hparams + config).get_config()
    
class TransformerEncoder(tf.keras.Model):
    def __init__(self, embedding_dim, num_layers, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams    = HParamsTransformerEncoder.extract(kwargs)
        self.hparams    = self.hparams(embedding_dim = embedding_dim, num_layers = num_layers)
        
        self.encoder_layers = [
            TransformerEncoderLayer(name = 'layer_{}'.format(i+1), ** self.hparams)
            for i in range(self.hparams.num_layers)
        ]
        
    def call(self, inputs, mask = None, training = False, return_attention = True, return_states = False):
        """
            Arguments :
                - inputs    : encoder inputs with shape [batch_size, seq_len, embedding_dim], embedded inputs
                - mask      : attention mask (padding mask based in inputs)
                - training  : whether it is training / inference phase
                - return_attention  : whether to return attention weights or not
                - return_states     : whether to return intermediate representation or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : dict self-attention weights for each head of the MHA of each layer
        """
        attn_outputs    = {} if return_attention else None
        states_outputs  = {} if return_states else None
        
        output = inputs
        for i, layer in enumerate(self.encoder_layers):
            output, attn_weights = layer(
                output, mask = mask, training = training, return_attention = True
            )
            if return_attention:
                attn_outputs['attn_{}'.format(layer.name)] = attn_weights
            
            if return_states:
                states_outputs['state_{}'.format(layer.name)] = output
        
        if not return_attention and not return_states:
            return hidden_states
        
        output = (output,)
        if return_attention: output = output + (attn_outputs,)
        if return_states: output = output + (states_outputs,)
        
        return output
    
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

class TransformerDecoder(tf.keras.Model):
    def __init__(self, embedding_dim, num_layers, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams    = HParamsTransformerDecoder.extract(kwargs)
        self.hparams    = self.hparams(embedding_dim = embedding_dim, num_layers = num_layers)
        
        self.decoder_layers = [
            TransformerDecoderLayer(name = 'layer_{}'.format(i+1), ** self.hparams)
            for i in range(self.hparams.num_layers)
        ]
        
    def call(self,
             inputs,
             training   = False,
             enc_padding_mask   = None,
             look_ahead_mask    = None,
             return_attention   = True,
             return_states      = False
            ):
        """
            Arguments :
                - inputs    : list of [encoder_output, decoder_input]
                    - encoder_output    : encoder output with shape [batch_size, seq_len_in, encoder_embedding_dim]
                    - decoder_input     : decoder input with shape [batch_size, seq_len_out, embedding_dim]
                - padding_mask  : mask for encoder input padding (passed to the 2nd attention block)
                - look_ahead_mask   : mask for decoder_input look_ahead (+ padding (optionnal))
                - training  : whether it is training / inference phase
                - return_attention  : whether to return attention weights or not
                - return_states     : whether to return intermediate representation or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : dict self-attention weights + encoder attention weights for each head of the MHA of each layer
        """
        attn_outputs    = {} if return_attention else None
        states_outputs  = {} if return_states else None
        
        encoder_output, output = inputs
        
        if look_ahead_mask is None:
            look_ahead_mask = create_look_ahead_mask(
                tf.shape(output)[0], tf.shape(output)[1], tf.float32
            )
        
        for i, layer in enumerate(self.decoder_layers):
            output, attn_weights, enc_attn_weights = layer(
                [encoder_output, output], training = training, return_attention = True,
                enc_padding_mask = enc_padding_mask, look_ahead_mask = look_ahead_mask
            )
            if return_attention:
                attn_outputs['attn_{}'.format(layer.name)] = attn_weights
                attn_outputs['enc_attn_{}'.format(layer.name)] = enc_attn_weights
            
            if return_states:
                states_outputs['state_{}'.format(layer.name)] = output
        
        if not return_attention and not return_states:
            return hidden_states
        
        output = (output,)
        if return_attention: output = output + (attn_outputs,)
        if return_states: output = output + (states_outputs,)
        
        return output
    
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

class Transformer(tf.keras.Model):
    def __init__(self, embedding_dim = None, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams = HParamsTransformer.extract(kwargs)
        # Allow to have different embedding dim for encoder and decoder
        if embedding_dim is not None:
            self.hparams    = self.hparams(
                encoder_embedding_dim = embedding_dim, decoder_embedding_dim = embedding_dim
            )
        
        self.encoder    = TransformerEncoder(
            ** self.hparams.get_config(prefix = 'encoder'), name = 'encoder'
        )
        
        self.decoder    = TransformerDecoder(
            ** self.hparams.get_config(prefix = 'decoder'), name = 'decoder'
        )
    
    @tf.function(experimental_relax_shapes = True)
    def encode(self, encoder_input, padding_mask = None, training = False):
        return self.encoder(
            encoder_input, mask = padding_mask, training = training, return_attention = True
        )
    
    @tf.function(experimental_relax_shapes = True)
    def decode(self, encoder_out, decoder_input, training = False,
               enc_padding_mask = None, look_ahead_mask = None):
        return self.decoder(
            [encoder_out, decoder_input],
            enc_padding_mask    = enc_padding_mask,
            look_ahead_mask     = look_ahead_mask,
            training    = training,
            return_attention    = True
        )
        
    def call(self,
             inputs,
             training   = False,
             encoder_padding_mask   = None,
             look_ahead_mask    = None,
             return_attention   = True
            ):
        encoder_input, decoder_input = inputs
        
        encoder_out, encoder_attn = self.encode(
            encoder_input, padding_mask = encoder_padding_mask, training = training
        )
        
        decoder_out, decoder_attn = self.decode(
            encoder_out,
            decoder_input,
            enc_padding_mask    = encoder_padding_mask,
            look_ahead_mask     = look_ahead_mask,
            training    = training
        )
        
        return decoder_out if not return_attention else (decoder_out, encoder_attn, decoder_attn)
        
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

custom_functions    = {
    'FeedForwardNetwork'    : FeedForwardNetwork,
    'TransformerEncoder'    : TransformerEncoder,
    'TransformerDecoder'    : TransformerDecoder,
    'Transformer'       : Transformer
}
        
custom_objects  = {
    'MultiHeadAttention'        : MultiHeadAttention,
    
    'TransformerEncoderLayer'   : TransformerEncoderLayer,
    'TransformerDecoderLayer'   : TransformerDecoderLayer,
    'TransformerEncoder'    : TransformerEncoder,
    'TransformerDecoder'    : TransformerDecoder,
    'Transformer'       : Transformer
}
