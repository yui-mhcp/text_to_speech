import tensorflow as tf

from tensorflow.keras.layers import *
from custom_layers import get_activation

_flatten_type   = (None, 'max', 'mean', 'avg', 'average', 'lstm', 'gru', 'bi_lstm', 'bilstm', 'bi_gru', 'bigru')
_pool_type      = (None, False, 'none', 'max', 'avg', 'average', 'up', 'upsampling')

_str_layers     = {
    'dense' : Dense, 'conv1d' : Conv1D, 'conv2d' : Conv2D, 'lstm' : LSTM, 'gru' : GRU,
    'bi_lstm' : lambda * args, ** kwargs: Bidirectional(LSTM(* args, ** kwargs)),
    'bi_gru'  : lambda * args, ** kwargs: Bidirectional(GRU(* args, ** kwargs))
}

def _get_var(_vars, i):
    if callable(_vars) and _vars.__class__.__name__ == 'function': return _vars(i)
    elif isinstance(_vars, list): return _vars[i]
    else: return _vars

def _add_layer(model, layer):
    """
        Add `layer` to `model` and return it
            If model is a Sequential    : add it with .add()
            else    : add it by calling the layer on the `model` (which is the output of the previous layer)
        
        If `layer` is a list, call _add_layer on all layer in the list
            If model is Sequential  : add them sequentially and return the model
            else : add all layer in parallel (all having `model` as input) and return the list of outputs
    """
    if isinstance(model, tf.keras.Sequential):
        if not isinstance(layer, (list, tuple)): layer = [layer]
        for l in layer: model.add(l)
        return model
    elif isinstance(layer, (list, tuple)):
        return [l(model) for l in layer]
    return layer(model)

def _get_flatten_layer(flatten_type, dim, ** kwargs):
    dim = dim.lower()
    assert flatten_type in _flatten_type, "Flatten type {} not a valid type {}".format(flatten_type, _flatten_type)
    assert dim in ('1d', '2d')
    
    if flatten_type is None:
        return Flatten()
    elif flatten_type == 'max':
        return GlobalMaxPooling1D() if dim == '1d' else GlobalMaxPooling2D()
    elif flatten_type in ('mean', 'avg', 'average'):
        return GlobalAveragePooling1D() if dim == '1d' else GlobalAveragePooling2D()
    else: # GRU / LSTM
        layer = LSTM(** kwargs) if 'lstm' in flatten_type else GRU(** kwargs)
        
        if 'bi' in flatten_type: layer = Bidirectional(layer)
        
        return layer

def _get_pooling_layer(pool_type, dim, * args, global_pooling = False, ** kwargs):
    if isinstance(pool_type, (list, tuple)):
        return [
            _get_pooling_layer(pool, dim, * args, global_pooling = global_pooling, ** kwargs)
            for pool in pool_type
        ]
    
    dim = dim.lower()
    assert pool_type in _pool_type, "Pool type {} not a valid type {}".format(pool_type, _pool_type)
    assert dim in ('1d', '2d')
    
    if pool_type == 'max':
        if global_pooling:
            pool_class = GlobalMaxPooling1D if dim == '1d' else GlobalMaxPooling2D
        else:
            pool_class = MaxPooling1D if dim == '1d' else MaxPooling2D
    elif pool_type in ('avg', 'average'):
        if global_pooling:
            pool_class = GlobalAveragePooling1D if dim == '1d' else GlobalAveragePooling2D
        else:
            pool_class = AveragePooling1D if dim == '1d' else AveragePooling2D
    elif pool_type in ('up', 'upsampling'):
        if global_pooling:
            raise ValueError("Upsampling does not exist in global pooling mode !")
        pool_class = UpSampling1D if dim == '1d' else UpSampling2D
    else:
        return None
    
    return pool_class(* args, ** kwargs)

def _get_layer(layer_name, * args, ** kwargs):
    if isinstance(layer_name, (list, tuple)):
        return [_get_layer(layer, * args, ** kwargs)for layer in layer_name]
    
    layer_name = layer_name.lower()
    if layer_name not in _str_layers:
        raise ValueError("Unknown layer type !\n  Accepted : {}\n  Got : {}".format(
            tuple(_str_layers.keys()), layer_name))
    
    return _str_layers[layer_name](* args, ** kwargs)

def _layer_bn(model, layer_type, n, * args, 
              
              bnorm     = 'after',
              momentum  = 0.99,
              epsilon   = 0.001,
              
              pooling       = None,
              pool_size     = 2,
              pool_strides  = 2,
              pool_padding  = 'valid',
              
              activation    = None,
              activation_kwargs = {},
              drop_rate     = 0.1,
              residual      = False,
              
              name  = None, 
              ** kwargs
             ):
    assert bnorm in ('before', 'after', 'never')
    assert pooling in _pool_type, '{} is not a valid pooling type'.format(pooling)
    
    x = model
    
    if bnorm == 'before':
        x = _add_layer(x, BatchNormalization(
            momentum = momentum, epsilon = epsilon
        ))
    
    for i in range(n):
        args_i = [_get_var(a, i) for a in args]
        kwargs_i = {k : _get_var(v, i) for k, v in kwargs.items()}
        kwargs_i['name'] = '{}_{}'.format(name, i+1) if name and n > 1 else name
        
        if i > 0: kwargs_i.pop('input_shape', None)

        x = _add_layer(x, layer_type(* args_i, ** kwargs_i))
        
        if activation and i < n - 1:
            x = _add_layer(x, get_activation(activation, ** activation_kwargs))
    
    if bnorm == 'after':
        x = _add_layer(x, BatchNormalization(
            momentum = momentum, epsilon = epsilon
        ))
    
    if residual and tuple(x.shape) == tuple(model.shape):
        x = Add()([x, inputs])
    elif residual:
        print("Skip connection failed : shape mismatch ({} vs {})".format(model.shape, x.shape))
    
    if activation:
        x = _add_layer(x, get_activation(activation, ** activation_kwargs))
    
    if pooling not in (None, False, 'none') and 'Conv' in layer_type.__name__:
        dim = '1d' if '1D' in layer_type.__name__ else '2d'
        x = _add_layer(x, _get_pooling_layer(
            pooling, dim, pool_size = pool_size,
            strides = pool_strides, padding = pool_padding
        ))
    
    if drop_rate > 0.: x = _add_layer(x, Dropout(drop_rate))
    
    return x

def DenseBN(x, units, * args, ** kwargs):
    n = len(units) if isinstance(units, (list, tuple)) else 1
    kwargs['units'] = units
    
    return _layer_bn(x, Dense, n, * args, ** kwargs)

def Conv2DBN(x, filters, *args, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    x = _layer_bn(x, Conv2D, n, * args, ** kwargs)
    
    return x

def Conv1DBN(x, filters, *args, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    x = _layer_bn(x, Conv1D, n, * args, ** kwargs)
    
    return x

def SeparableConv2DBN(x, filters, *args, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    x = _layer_bn(x, SeparableConv2D, n, * args, ** kwargs)
    
    return x

def SeparableConv1DBN(x, filters, *args, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    x = _layer_bn(x, SeparableConv1D, n, * args, ** kwargs)
    
    return x

def Conv2DTransposeBN(x, filters, *args, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    x = _layer_bn(x, Conv2DTranspose, n, * args, ** kwargs)
    
    return x

def Conv1DTransposeBN(x, filters, *args, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    x = _layer_bn(x, Conv1DTranspose, n, * args, ** kwargs)
    
    return x
