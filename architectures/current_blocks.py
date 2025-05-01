# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras
import inspect
import logging
import importlib

from keras import tree
from keras import layers

from .layers import get_activation

logger  = logging.getLogger(__name__)

_use_cudnn_lstm = False

_keras_layers   = {k : v for k, v in vars(keras.layers).items() if isinstance(v, type)}
logger.debug('Found {} layers in `keras.layers`'.format(len(_keras_layers)))

_keras_layers_lower = {k.lower() : v for k, v in _keras_layers.items()}

_conv_layers    = {k : v for k, v in _keras_layers.items() if 'Conv' in k}

_merging_layers = {k : v for k, v in _keras_layers_lower.items() if 'merging' in v.__module__}
_pooling_layers = {k : v for k, v in _keras_layers_lower.items() if 'pooling' in v.__module__}

_flatten_type   = {None, 'none', 'max', 'avg', 'average', 'lstm', 'gru', 'bilstm', 'bigru'}

def _load_masked_layers():
    for module in ('masked_1d', 'masked_2d', 'masked_3d'):
        try:
            module = importlib.import_module('architectures.layers.' + module)
        except:
            continue

        masked_layers = {k : v for k, v in vars(module).items() if isinstance(v, type)}
        
        logger.debug('Layers {} loaded from {}'.format(tuple(masked_layers.keys()), module))
        
        _keras_layers.update(masked_layers)
        _conv_layers.update({k : v for k, v in masked_layers.items() if 'Conv' in k})
        _pooling_layers.update({k.lower() : v for k, v in masked_layers.items() if 'Pooling' in k})
        
_load_masked_layers()

def _get_var(_vars, i, key = None):
    if callable(_vars) and _vars.__class__.__name__ == 'function':
        return _vars(i) if key is None or 'activation' not in key else _vars
    elif isinstance(_vars, list): return _vars[i]
    else: return _vars

def _get_layer_dim(layer_class_name):
    if '1D' in layer_class_name:    return '1D'
    elif '2D' in layer_class_name:  return '2D'
    elif '3D' in layer_class_name:  return '3D'
    raise ValueError('Unable to determine dimension for {}'.format(layer_class_name))

def add_layer(model, layer):
    """
        Add `layer` to `model` and return it
            If model is a `Sequential`  : calls the `model.add` method
            else    : add it by calling the layer on the `model` (which is expected to be the output of the previous layer)
        
        If `layer` is a list, call _add_layer on all layer in the list
            If model is `Sequential`  : add them sequentially and return the model
            else : add all layer in parallel (all having `model` as input) and return the list of outputs
    """
    if isinstance(model, keras.Sequential):
        if not isinstance(layer, (list, tuple)): layer = [layer]
        for l in layer: model.add(l)
        return model
    
    elif isinstance(layer, (list, tuple)): return [l(model) for l in layer]
    return layer(model)

def get_merging_layer(name, ** kwargs):
    if name is True: name = 'add'
    
    name = name.lower()
    if name == 'concat': name = 'concatenate'
    if name not in _merging_layers:
        raise ValueError('Unknown merging layer !\n  Accepted : {}\n  Got : {}'.format(
            tuple(_merging_layers.keys()), name
        ))
    
    return _merging_layers[name](** kwargs)

def get_pooling_layer(pool_type,
                      input_dim,
                      * args,
                      use_mask  = None,
                      global_pooling    = False,
                      ** kwargs
                     ):
    """ Returns a pooling layer : `{Max / Average}Pooling{dim}` or `UpSampling{dim}` """
    if isinstance(pool_type, (list, tuple)):
        return [
            get_pooling_layer(pool, input_dim, * args, global_pooling = global_pooling, ** kwargs)
            for pool in pool_type
        ]
    
    if pool_type in (None, False, 'none'): return None
    
    input_dim = input_dim.lower()
    assert input_dim in ('1d', '2d', '3d'), 'Invalid dimension {}'.format(input_dim)
    
    layer_name = pool_type.lower() + 'pooling' + input_dim
    if global_pooling: layer_name = 'global' + layer_name
    
    if use_mask:
        if 'masked' + layer_name not in _pooling_layers:
            logger.warning('The masked pooling layer {} does not exist, fallback to regular pooling'.format(layer_name))
        else:
            layer_name = 'masked' + layer_name
    
    if layer_name not in _pooling_layers:
        raise ValueError('Unknown pooling layer !\n  Accepted : {}\n  Got : {}'.format(
            tuple(_pooling_layers.keys()), layer_name
        ))
    
    layer  = _pooling_layers[layer_name]
    kwargs = {k : v for k, v in kwargs.items() if k in inspect.signature(layer).parameters}
    return layer(* args, ** kwargs)

def get_flatten_layer(flatten_type, dim, use_cudnn = _use_cudnn_lstm, ** kwargs):
    """
        Returns a global flattening layer
        
        Arguments :
            - flatten_type  : the type of the layer
                - None          : regular `Flatten` layer
                - max / average : `Global{flatten_type}Pooling{dim}D`
                - lstm / gru    : RNN layer
                - bi_{lstm / gru}   : RNN layer wrapped in a `Bidirectional` layer
            - dim   : 1d / 2d / 3d, the dimension for the global pooling layer
        Returns :
            - layer : the flattening layer
    """
    if isinstance(flatten_type, str):
        flatten_type = ''.join(c for c in flatten_type if c.isalnum())
    assert flatten_type in _flatten_type, 'Invalid flatten layers !\n  Accepted : {}\n  Got : {}'.format(
        _flatten_type, flatten_type
    )
    
    if flatten_type in (None, 'none'):
        return layers.Flatten(** kwargs)
    elif flatten_type in ('max', 'avg', 'average'):
        return get_pooling_layer(flatten_type, dim, global_pooling = True, ** kwargs)
    else: # GRU / LSTM
        layer = layers.LSTM(** kwargs) if 'lstm' in flatten_type else layers.GRU(** kwargs)
        if 'bi' in flatten_type:
            layer = layers.Bidirectional(layer)
        set_cudnn_lstm(layer, ** kwargs)
        
        return layer

def set_cudnn_lstm(layer, use_cudnn = _use_cudnn_lstm, ** _):
    if isinstance(layer, keras.Model):
        _layers = layer._flatten_layers()
    elif isinstance(layer, layers.Bidirectional):
        _layers = (layer.forward_layer, layer.backward_layer)
    else:
        _layers = layer
    
    for l in _layers:
        if hasattr(l, 'use_cudnn'):
            l.use_cudnn = use_cudnn
            if not use_cudnn: l.supports_jit = True

def get_padding_layer(kernel_size, dim, *, dilation_rate = 1, use_mask = None, ** kwargs):
    if isinstance(kernel_size, (list, tuple)):
        if len(kernel_size) > 1:
            raise NotImplementedError('not supported when `kernel_size` is a tuple')
        kernel_size = kernel_size[0]
    if dilation_rate > 1:
        raise NotImplementedError('not supported when `dilation_rate > 1`')
    if kernel_size % 2 == 0:
        raise NotImplementedError('not supported when `kernel_size % 2 == 0`')
    
    padding_half =  kernel_size // 2
    if padding_half < 1: return None
    
    padding = (padding_half, padding_half)
    
    dim  = int(dim[0])
    name = '{}ZeroPadding{}D'.format('Masked' if use_mask else '', dim)
    if name not in _keras_layers:
        raise NotImplementedError('the layer `{}` is not available'.format(name))
    
    if dim > 1: dim = tuple([padding] * dim)
    return _keras_layers[name](padding)

def _get_layer(layer_name, * args, ** kwargs):
    if isinstance(layer_name, (list, tuple)):
        return [_get_layer(layer, * args, ** kwargs)for layer in layer_name]
    
    layer_name = layer_name.lower()
    if layer_name not in _str_layers:
        raise ValueError("Unknown layer type !\n  Accepted : {}\n  Got : {}".format(
            tuple(_str_layers.keys()), layer_name))
    
    return _str_layers[layer_name](* args, ** kwargs)

def _layer_bn(layer_class,
              model,
              * args,
              
              n = 1,
              use_manual_padding    = False,
              
              bnorm     = 'after',
              momentum  = 0.99,
              epsilon   = 0.001,
              bn_axis   = -1,
              bn_name   = None,
              
              use_mask  = False,
              pooling   = None,
              pool_size = None,
              pool_strides  = None,
              pool_padding  = 'valid',
              
              activation    = None,
              activation_kwargs = {},
              drop_rate     = 0.1,
              
              residual      = False,
              residual_kwargs   = {},
              residual_tensor   = None,
              
              name  = None,
              ** kwargs
             ):
    """
        Adds `n` times `layer_class` to `model` with `kwargs`, and possibly adds additional layers
        
        Arguments :
            - model         : either `Tensor` (for the Functional API) either `Sequential` instance
            - layer_class    : the layer's class to add
            - n             : the number of `layer_class` to add consecutively
            - args / kwargs : configuration for the `layer_class`
            
            - use_mask      : whether to use `Masking` in the `ZeroPadding` / pooling layers
            - use_manual_padding    : whether to add a `ZeroPadding{n}D` to override the `padding = 'same'` argument of convolutional layers. This may be useful for masking (`use_mask = True`)
            
            - bnorm     : where to add the normalization layer ('before', 'after' or 'never')
            - momentum / epsilon / bn_{axis / name} : kwargs for the `BatchNormalization` layer
            
            - pooling   : the pooling type to use (if None / False, do not add padding layer)
            - pool_{size / strides / padding}   : kwargs for the pooling layer
            
            - activation    : the activation layer to use
            - activation_kwargs : the kwargs for the activation layer
            - drop_rate     : the argument for the `Dropout` layer (if 0, no Dropout is added)
            
            - residual      : the residual type (False means no residual)
                This feature is not available in the `Sequential API`
                If `residual = True`, the default layer is the `Add()` layer
            - residual_kwargs   : the kwargs for the residual layer
            
            - name  : the name to use for the `layer_class`
        Returns : the updated `model` (if Sequential API) or the new `Tensor`
    """
    assert bnorm in ('before', 'after_all', 'after', 'never'), 'Invalid `bnorm` : {}'.format(bnorm)
    
    if residual and isinstance(model, keras.Sequential):
        raise ValueError('Residual connections are not supported in `Sequential` models')
    
    if n > 1:
        if name and '{}' not in name: name = name + '_{}'
        if bn_name and bnorm == 'after_all' and '{}' not in bn_name: bn_name = bn_name + '_{}'

    
    x = model
    
    if bnorm == 'before':
        x = add_layer(x, layers.BatchNormalization(
            momentum = momentum, epsilon = epsilon, axis = bn_axis, name = bn_name
        ))

    for i in range(n):
        args_i      = [_get_var(a, i) for a in args]
        kwargs_i    = {k : _get_var(v, i) for k, v in kwargs.items()}
        kwargs_i['name'] = name.format(i + 1) if n > 1 else name
        
        if i > 0: kwargs_i.pop('input_shape', None)
        
        if use_manual_padding and kwargs_i.get('padding', None) == 'same':
            dim = _get_layer_dim(layer_class.__name__)
            try:
                pad_layer = get_padding_layer(
                    kwargs_i['kernel_size'],
                    dim,
                    use_mask = use_mask,
                    dilation_rate = kwargs_i.get('dilation_rate', 1)
                )
                if pad_layer is not None: x = add_layer(x, pad_layer)
                kwargs_i['padding'] = 'valid'
            except NotImplementedError as e:
                logger.warning('manual padding is not supported for {} (reason : {})'.format(
                    kwargs_i['name'], e
                ))
        
        x = add_layer(x, layer_class(* args_i, ** kwargs_i))
        
        if bnorm == 'after_all':
            bn_name_i = bn_name.format(i + 1) if n > 1 else bn_name
            x = add_layer(x, layers.BatchNormalization(
                momentum = momentum, epsilon = epsilon, axis = bn_axis, name = bn_name_i
            ))

        if activation and i < n - 1:
            x = add_layer(x, get_activation(activation, ** activation_kwargs))
    
    if bnorm == 'after':
        x = add_layer(x, layers.BatchNormalization(
            momentum = momentum, epsilon = epsilon, axis = bn_axis, name = bn_name
        ))
    
    if residual:
        if residual_tensor is None: residual_tensor = model
        if tuple(x.shape) == tuple(residual_tensor.shape):
            x = get_merging_layer(residual, ** residual_kwargs)([x, residual_tensor])
        else:
            logger.warning("Skip connection failed : shape mismatch ({} vs {})".format(
                model.shape, x.shape
            ))
    
    if activation:
        x = add_layer(x, get_activation(activation, ** activation_kwargs))
    
    if pooling and 'Conv' in layer_class.__name__:
        if pool_size is None and pool_strides is None: pool_size = 2
        if pool_size is None:       pool_size = pool_strides
        if pool_strides is None:    pool_strides = pool_size
        
        dim = _get_layer_dim(layer_class.__name__)
        x   = add_layer(x, get_pooling_layer(
            pooling,
            input_dim   = dim,
            pool_size   = pool_size,
            strides     = pool_strides,
            padding     = pool_padding,
            use_mask    = use_mask
        ))
    
    if drop_rate > 0.: x = add_layer(x, layers.Dropout(drop_rate))
    
    return x


def DenseBN(x, units, * args, ** kwargs):
    n = len(units) if isinstance(units, (list, tuple)) else 1
    return _layer_bn(layers.Dense, x, units, * args, n = n, ** kwargs)

def _build_conv_bn(layer):
    def conv_bn(x, filters, * args, ** kwargs):
        n = len(filters) if isinstance(filters, (list, tuple)) else 1
        return _layer_bn(layer, x, filters, * args, n = n, ** kwargs)
    return conv_bn

globals().update({
    '{}BN'.format(k) : _build_conv_bn(v) for k, v in _conv_layers.items()
})

