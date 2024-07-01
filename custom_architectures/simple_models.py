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

from keras.layers import Input, Dense

from utils.hparams import HParams
from . import current_blocks
from .current_blocks import *
from .current_blocks import _get_var, _conv_layers
from custom_layers import get_activation

_conv_types = {k.lower() : k + 'BN' for k in _conv_layers.keys()}

HParamsDenseBN  = HParams(
    units       = 32, 
    use_bias    = True, 
    bnorm       = 'never', 
    kernel_initializer  = 'glorot_uniform',
    activation  = 'relu', 
    drop_rate   = 0.25
)

HParamsConvBN   = HParams(
    filters         = lambda i: 32 * 2 ** i,
    kernel_size     = 3,
    strides         = 1,
    padding         = 'same',
    use_bias        = True,
    dilation_rate   = 1,
    kernel_initializer  = 'glorot_uniform',
    bias_initializer    = 'zeros',
    use_manual_padding  = True,
    
    activation      = 'relu',
    drop_rate       = 0.25,
    bnorm           = 'after',
    
    use_mask    = None,
    pooling     = None,
    pool_size   = 2,
    pool_strides    = 2,
    residual    = False
)

HParamsPerceptron = HParams(
    ** HParamsDenseBN,
    pooling     = None,
    n_dense     = 2,
    dense_name  = 'dense_{}',
    
    final_bias          = False,
    final_activation    = None,
    final_name          = 'classification_layer',
    
    name     = 'multi_layer_perceptron'
)

HParamsCNN = HParams(
    ** HParamsConvBN,
    ** HParamsConvBN.get_config(add_prefix = 'final_conv'),
    ** HParamsDenseBN.get_config(add_prefix = 'dense'),
    
    add_mask_layer  = None,
    pad_value   = 0.,
    add_final_norm  = False,
    
    n_conv  = 4,
    conv_name   = 'conv_{}',
    norm_name   = 'norm_{}',
    n_final_conv    = 0,
    final_conv_name = 'final_conv_{}',
    final_norm_name = 'final_norm_{}',
    
    dense_as_final   = True,
    flatten          = True,
    flatten_type     = None,
    flatten_kwargs   = {},
    
    n_dense         = 0,
    dense_name      = 'dense_{}',
    
    final_bias   = False,
    final_activation = None,
    final_name   = 'classification_layer',
    
    conv_type    = 'conv2d',
    
    name = 'simple_cnn'
)

def perceptron(input_shape  = None,
               output_shape = None,
               
               inputs = None,
               return_output = False,
               
               ** kwargs
              ):
    assert (input_shape is not None or inputs is not None) and output_shape is not None
    hparams = HParamsPerceptron(** kwargs)

    use_sequential = not isinstance(output_shape, (list, tuple))
    if inputs is None:
        if isinstance(input_shape, int): input_shape = (input_shape, )
        
        inputs = Input(shape = input_shape, name = 'inputs')
        if use_sequential:
            inputs = keras.Sequential([inputs], name = hparams.name)
    else:
        use_sequential = False
        input_shape = tuple(inputs.shape)[1:]
    
    x = inputs
    if len(input_shape) > 1:
        x = add_layer(x, get_flatten_layer(
            hparams.pooling, '{}d'.format(len(input_shape) - 1)
        ))
    
    for i in range(hparams.n_dense):
        config = HParamsDenseBN.extract(hparams)
        config = {k : _get_var(v, i, key = k) for k, v in config.items()}
        config['name'] = hparams.dense_name.format(i)
        
        x = DenseBN(x, ** config)
    
    if not isinstance(output_shape, (list, tuple)):
        outputs = add_layer(x, Dense(
            output_shape,
            use_bias    = hparams.final_bias,
            activation  = get_activation(hparams.final_activation),
            name    = hparams.final_name
        ))
    else:
        outputs = [Dense(
            output_shape_i,
            use_bias    = _get_var(hparams.final_bias, i),
            activation  = get_activation(_get_var(hparams.final_activation, i)),
            name        = '{}_{}'.format(hparams.final_name, i)
        )(x) for i, output_shape_i in enumerate(output_shape)]
        
    if use_sequential or return_output: return outputs
    return keras.Model(inputs = inputs, outputs = outputs, name = hparams.name)

def simple_cnn(input_shape  = None,
               output_shape = None,
               
               inputs = None,
               return_output    = False,
               use_sequential   = None,

               ** kwargs
              ):
    assert (input_shape is not None or inputs is not None) and output_shape is not None
    hparams = HParamsCNN(** kwargs)
    
    if hparams.add_mask_layer is None:
        hparams.add_mask_layer = True if inputs is None else False
    
    if use_sequential is None:
        use_sequential = not isinstance(output_shape, (list, tuple)) and not hparams.residual and not hparams.final_conv_residual
    
    if inputs is None:
        if isinstance(input_shape, int): input_shape = (input_shape, input_shape, 3)
        
        inputs = Input(shape = input_shape, name = 'input_image')
        if use_sequential:
            inputs = keras.Sequential([inputs], name = hparams.name)
    else:
        use_sequential = False
        input_shape = tuple(inputs.shape)[1:]

    conv_type = hparams.conv_type.lower()
    if hparams.use_mask is None:
        hparams.use_mask = True if 'masked{}'.format(conv_type) in _conv_types else False
        logger.info('Set `use_mask = {}` by default as the masking operation is{} supported. To remove this message or change the behavior, explicitely pass `use_mask = ...`'.format(
            hparams.use_mask, '' if hparams.use_mask else ' not'
        ))
    
    if hparams.use_mask: conv_type = 'masked{}'.format(conv_type)
    assert conv_type in _conv_types, "Unknown convolution layer : {}".format(conv_type)

    conv_layer_fn = getattr(current_blocks, _conv_types[conv_type])
    
    ########################################
    #     Conv (with pooling / strides)    #
    ########################################
    
    x = inputs
    if hparams.use_mask and hparams.add_mask_layer:
        x = add_layer(x, keras.layers.Masking(mask_value = hparams.pad_value))

    for i in range(hparams.n_conv):
        last_layer = (i == hparams.n_conv -1) and hparams.n_final_conv == 0 and not hparams.dense_as_final and not hparams.flatten
        
        config  = HParamsConvBN.extract(hparams)
        config  = {k : _get_var(v, i, key = k) for k, v in config.items()}
        config.update({
            'name'  : hparams.conv_name.format(i + 1),
            'bn_name'   : hparams.norm_name.format(i + 1)
        })
        
        if last_layer:
            config.update({
                'filters'   : output_shape,
                'activation'    : hparams.final_activation,
                'drop_rate' : 0.
            })
            if not hparams.add_final_norm: config['bnorm'] = 'never'

        x = conv_layer_fn(x, ** config)
    
    ############################################################
    #      Final conv (normally without pooling / strides)     #
    ############################################################
    
    for i in range(hparams.n_final_conv):
        config  = hparams.get_config(prefix = 'final_conv')
        config  = {k : _get_var(v, i, key = k) for k, v in config.items()}
        config.update({
            'name'  : hparams.final_conv_name.format(i + 1),
            'bn_name'   : hparams.final_norm_name.format(i + 1)
        })
        
        x = conv_layer_fn(x, ** config)
        
    ##########
    #  Dense #
    ##########
    
    if hparams.flatten:
        if not hparams.dense_as_final: hparams.flatten_kwargs.setdefault('units', output_shape)
        dim = '1d' if '1d' in conv_type else '2d'
        x = add_layer(x, get_flatten_layer(
            hparams.flatten_type, dim, ** hparams.flatten_kwargs
        ))
        if not hparams.dense_as_final and hparams.final_activation:
            x = add_layer(x, get_activation(hparams.final_activation))
        
    elif len(input_shape) == 3 and hparams.dense_as_final:
        seq_len = -1 if input_shape[0] is None else x.shape[1]
        x = add_layer(x, Reshape((seq_len, x.shape[-2] * x.shape[-1])))
    
    outputs = x
    if hparams.dense_as_final:
        for i in range(hparams.n_dense):
            config = HParamsDenseBN.extract(hparams.get_config(prefix = 'dense'))
            config = {k : _get_var(v, i, key = k) for k, v in config.items()}
            
            x = DenseBN(x, ** config)

        ####################
        #      Outputs     #
        ####################
        if isinstance(output_shape, (list, tuple)):
            output_layer = [Dense(
                output_shape_i,
                use_bias    = _get_var(hparams.final_bias, i),
                activation  = _get_var(hparams.final_activation, i),
                name        = '{}_{}'.format(hparams.final_name, i)
            ) for i, output_shape_i in enumerate(output_shape)]
        else:
            output_layer = Dense(
                output_shape,
                use_bias    = hparams.final_bias,
                activation  = hparams.final_activation, 
                name        = hparams.final_name
            )
        
        outputs = add_layer(outputs, output_layer)
        if hparams.final_activation:
            if isinstance(outputs, (list, tuple)):
                outputs = [
                    get_activation(_get_var(hparams.final_activation, i) if isinstance(hparams.final_activation, (list, tuple)) else hparams.final_activation)(xi)
                    for i, xi in enumerate(outputs)
                ]
            else:
                outputs = add_layer(outputs, get_activation(hparams.final_activation))

    
    if use_sequential or return_output:
        return outputs
    else:
        return keras.Model(inputs = inputs, outputs = outputs, name = hparams.name)


def comparator(encoder_a,
               encoder_b,
               input_signature_a    = None,
               input_signature_b    = None,
               distance_metric      = 'euclidian',
               pred_probability     = True,
               final_name   = 'decision_layer',
               name     = 'Comparator',
               ** kwargs
              ):
    from custom_layers import SimilarityLayer
    
    def get_input(model, signature, name):
        if signature is None:
            input_shape, input_type = model.input_shape[1:], 'float32'
        elif not isinstance(signature, (list, tuple)):
            input_shape, input_type = signature.shape[1:], signature.dtype
        else:
            input_shape = [inp.shape[1:] for inp in signature]
            input_type  = [inp.dtype for inp in signature]
        
            return [
                Input(shape, dtype = dtype, name = name + '_{}'.format(i))
                for i, (shape, dtype) in enumerate(zip(input_shape, input_type))
            ]
        return Input(input_shape, dtype = input_type, name = name)
    
    assert isinstance(encoder_a, keras.Model) and isinstance(encoder_b, keras.Model)
    
    input_a = get_input(encoder_a, input_signature_a, 'input_a')
    input_b = get_input(encoder_b, input_signature_b, 'input_b')
    
    inputs  = [input_a, input_b]
    
    embedded_a = encoder_a(input_a)
    embedded_b = encoder_b(input_b)
    
    if isinstance(embedded_a, (list, tuple)): embedded_a = embedded_a[0]
    if isinstance(embedded_b, (list, tuple)): embedded_b = embedded_b[0]
    
    output = SimilarityLayer(
        distance_metric = distance_metric, pred_probability = pred_probability, name = final_name
    )([embedded_a, embedded_b])
    
    return keras.Model(inputs = inputs, outputs = output, name = name)

def siamese(model, input_signature = None, activation = 'sigmoid',
            name = 'SiameseNetwork', ** kwargs):
    """ Special case of `Comparator` where both `encoder_a` and `encoder_b` are the same model """
    kwargs.update({'input_signature_a' : input_signature, 'input_signature_b' : input_signature})
    
    return comparator(
        encoder_a = model, encoder_b = model, activation = activation, name = name, ** kwargs
    )

def classifier(feature_extractor,
               input_shape  = None,
               output_shape = None,
               
               include_top  = True,
               weights      = 'imagenet',
               pooling      = None,
               ** kwargs
              ):
    is_imagenet_model = False
    if callable(feature_extractor):
        is_imagenet_model = True
        feature_extractor = feature_extractor(
            input_shape = input_shape,
            include_top = include_top,
            weights     = weights,
            pooling     = pooling
        )
    elif isinstance(feature_extractor, str):
        assert feature_extractor != 'classifier'
        
        from custom_architectures import get_architecture
        feature_extractor = get_architecture(feature_extractor, input_shape = input_shape, ** kwargs)

    assert isinstance(feature_extractor, keras.Model), "Invalid feature_extractor type : {}".format(type(feature_extractor))
    
    if output_shape is not None:
        kwargs.setdefault('n_dense', 0)
        
        features = feature_extractor.layers[-2].output if is_imagenet_model and include_top else feature_extractor.output
        output = perceptron(
            inputs = features, output_shape = output_shape, return_output = True,
            pooling = pooling, ** kwargs
        )
        
        return keras.Model(feature_extractor.input, output, name = feature_extractor.name)
    
    return feature_extractor

_conv_layer_fn = {k.lower().replace('bn', '') : v for k, v in globals().items() if 'Conv' in k}