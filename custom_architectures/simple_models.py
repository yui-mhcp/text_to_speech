import tensorflow as tf

from tensorflow.keras.layers import Input, Subtract, Flatten, Dense

from custom_architectures.current_blocks import *
from custom_architectures.current_blocks import _get_var, _add_layer, _get_flatten_layer

from hparams import HParams

HParamsDenseBN  = HParams(
    units       = [32, 32], 
    use_bias    = True, 
    bnorm       = 'never', 
    kernel_initializer  = 'glorot_uniform',
    activation  = 'leaky', 
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
    
    activation      = 'relu',
    drop_rate       = 0.25,
    bnorm           = 'after',
    
    pooling         = None,
    pool_size       = 2,
    pool_strides    = 2,
    residual        = False
)

HParamsPerceptron = HParams(
    ** HParamsDenseBN,
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
    
    n_conv  = 4,
    conv_name   = 'conv_{}',
    n_final_conv    = 0,
    final_conv_name = 'final_conv_{}',
    
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

def perceptron(input_shape, output_shape, ** kwargs):
    if isinstance(input_shape, int): input_shape = (input_shape,)

    use_sequential = not isinstance(output_shape, (list, tuple))
    
    hparams = HParamsPerceptron(** kwargs)
    
    if use_sequential:
        inputs = tf.keras.Sequential(name = hparams.name)
    else:
        inputs = Input(shape = input_shape, name = "inputs")
    
    x = inputs
    if len(input_shape) > 1:
        x = _add_layer(x, Flatten())
            
    for i in range(hparams.n_dense):
        config = HParamsDenseBN.extract(hparams)
        config = {k : _get_var(v, i) for k, v in config.items()}
        config['name'] = hparams.dense_name.format(i)
        
        x = DenseBN(x, ** config)
    
    if use_sequential:
        x.add(Dense(
            output_shape,
            use_bias    = hparams.final_bias,
            activation  = hparams.final_activation, 
            name        = hparams.final_name
        ))
        
        x.build((None,) + input_shape)
        
        return x
    else:
        outputs = [Dense(
            output_shape_i,
            use_bias    = _get_var(hparams.final_bias, i),
            activation  = _get_var(hparams.final_activation, i),
            name        = '{}_{}'.format(hparams.final_name, i)
        )(x) for i, output_shape_i in enumerate(output_shape)]
    
        return tf.keras.Model(inputs = inputs, outputs = outputs, name = hparams.name)

def simple_cnn(input_shape, output_shape, **kwargs):
    hparams = HParamsCNN(** kwargs)

    conv_type = hparams.conv_type.lower()
    assert conv_type in _conv_layer_fn, "Unknown conv type : {}".format(conv_type)
    
    conv_layer_fn = _conv_layer_fn[conv_type]
    
    use_sequential = not isinstance(output_shape, (list, tuple)) and not hparams.residual and not hparams.final_conv_residual
    if use_sequential:
        inputs = tf.keras.Sequential(name = hparams.name)
    else:
        inputs = Input(shape = input_shape, name = "inputs")
    
    ########################################
    #     Conv (with pooling / strides)    #
    ########################################
    
    x = inputs
    for i in range(hparams.n_conv):
        last_layer = (i == hparams.n_conv -1) and hparams.n_final_conv == 0 and not hparams.dense_as_final
        
        config  = HParamsConvBN.extract(hparams)
        config  = {k : _get_var(v, i) for k, v in config.items()}
        config['name']  = hparams.conv_name.format(i+1)
        
        if last_layer:
            config.update({
                'activation'    : hparams.final_activation,
                'bnorm'     : 'never',
                'drop_rate' : 0.
            })
        
        x = conv_layer_fn(x, ** config)
    
    ############################################################
    #      Final conv (normally without pooling / strides)     #
    ############################################################
    
    for i in range(hparams.n_final_conv):
        config  = hparams.get_config(prefix = 'final_conv')
        config  = {k : _get_var(v, i) for k, v in config.items()}
        config['name']  = hparams.final_conv_name.format(i+1)
        
        x = conv_layer_fn(x, ** config)
        
    ##########
    #  Dense #
    ##########
    
    if hparams.flatten:
        if not hparams.dense_as_final: hparams.flatten_kwargs['units'] = output_shape
        dim = '1d' if '1d' in conv_type else '2d'
        x = _add_layer(x, _get_flatten_layer(
            hparams.flatten_type, dim, ** hparams.flatten_kwargs
        ))
        
    elif len(input_shape) == 3 and hparams.dense_as_final:
        seq_len = -1 if input_shape[0] is None else x.shape[1]
        x = _add_layer(x, Reshape((seq_len, x.shape[-2] * x.shape[-1])))
    
    outputs = x
    if hparams.dense_as_final:
        for i in range(hparams.n_dense):
            config = HParamsDenseBN.extract(hparams.get_config(prefix = 'dense'))
            config = {k : _get_var(v, i) for k, v in config.items()}
            
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
        
        outputs = _add_layer(outputs, output_layer)
    
    if use_sequential:
        outputs.build((None,) + input_shape)
        return outputs
    else:
        return tf.keras.Model(inputs = inputs, outputs = outputs, name = hparams.name)


def siamese(model, input_shape = None,
            distance_metric = 'euclidian', activation = 'sigmoid', 
            final_name = 'decision_layer', name = 'SiameseNet'):
    assert distance_metric in _distance_fn
    assert isinstance(model, tf.keras.Model)
    
    if input_shape is None: input_shape = model.input_shape
    
    input1 = Input(input_shape, name = 'input_1')
    input2 = Input(input_shape, name = 'input_2')
    inputs = [input1, input2]
    
    embedded_1 = model(input1)
    embedded_2 = model(input2)
    
    embedded_distance = Subtract()([embedded_1, embedded_2])
    embedded_distance = Lambda(
        _distance_fn[distance_metric], name = distance_metric
    )(embedded_distance)
    
    if len(embedded_distance.shape) > 1:
        embedded_distance = Flatten()(embedded_distance)
    
    output = Dense(1, activation = activation, name = final_name)(embedded_distance)
    
    return tf.keras.Model(inputs = inputs, outputs = output, name = name)
    
_distance_fn = {
    'l1'        : lambda x: tf.abs(x),
    'euclidian' : lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), axis = -1, keepdims = True))
}

_conv_layer_fn = {
    'conv1d'    : Conv1DBN,
    'conv2d'    : Conv2DBN,
    'separableconv1d'   : SeparableConv1DBN,
    'separableconv2d'   : SeparableConv2DBN,
    'conv1dtranspose'   : Conv1DTransposeBN,
    'conv2dtranspose'   : Conv2DTransposeBN
}

custom_functions   = {
    'perceptron'    : perceptron,
    'simple_cnn'    : simple_cnn,
    'siamese'       : siamese
}
