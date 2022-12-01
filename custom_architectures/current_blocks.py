
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

import logging
import tensorflow as tf

from tensorflow.keras.layers import *
from custom_layers import get_activation
from custom_layers.masked_1d import *

logger  = logging.getLogger(__name__)

_flatten_type   = (None, 'max', 'mean', 'avg', 'average', 'lstm', 'gru', 'bi_lstm', 'bilstm', 'bi_gru', 'bigru')
_pool_type      = (None, False, 'none', 'max', 'avg', 'average', 'up', 'upsampling')

_str_layers     = {
    'dense'     : Dense,
    'conv1d'    : Conv1D,
    'masked_conv1d' : MaskedConv1D,
    'conv2d'    : Conv2D,
    'lstm'      : LSTM,
    'gru'       : GRU,
    'bi_lstm'   : lambda * args, ** kwargs: Bidirectional(LSTM(* args, ** kwargs)),
    'bi_gru'    : lambda * args, ** kwargs: Bidirectional(GRU(* args, ** kwargs))
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

def _get_pooling_layer(pool_type, dim, * args, global_pooling = False, use_mask = None, ** kwargs):
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
            if dim == '2d': pool_class = MaxPooling2D
            else: pool_class = MaxPooling1D if use_mask is False else MaskedMaxPooling1D
    elif pool_type in ('avg', 'average'):
        if global_pooling:
            pool_class = GlobalAveragePooling1D if dim == '1d' else GlobalAveragePooling2D
        else:
            if dim == '2d': pool_class = AveragePooling2D
            else: pool_class = AveragePooling1D if use_mask is False else MaskedAveragePooling1D
    elif pool_type in ('up', 'upsampling'):
        if global_pooling:
            raise ValueError("Upsampling does not exist in global pooling mode !")
        pool_class = UpSampling1D if dim == '1d' else UpSampling2D
    else:
        return None
    
    return pool_class(* args, ** kwargs)

def _get_padding_layer(kernel_size, dim, * args, use_mask = None, ** kwargs):
    if isinstance(kernel_size, (list, tuple)): kernel_size = kernel_size[0]
    kernel_half = kernel_size // 2
    if kernel_size <= 1: return None
    
    padding = (kernel_half, kernel_half)
    
    dim = dim.lower()
    assert dim in ('1d', '2d')
    
    if dim == '2d':
        raise NotImplementedError()
    else:
        return ZeroPadding1D(padding) if not use_mask else MaskedZeroPadding1D(padding)

def _get_layer(layer_name, * args, ** kwargs):
    if isinstance(layer_name, (list, tuple)):
        return [_get_layer(layer, * args, ** kwargs)for layer in layer_name]
    
    layer_name = layer_name.lower()
    if layer_name not in _str_layers:
        raise ValueError("Unknown layer type !\n  Accepted : {}\n  Got : {}".format(
            tuple(_str_layers.keys()), layer_name))
    
    return _str_layers[layer_name](* args, ** kwargs)

def _layer_bn(model, layer_type, n, * args, 
              use_manual_padding    = False,
              
              bnorm     = 'after',
              momentum  = 0.99,
              epsilon   = 0.001,
              bn_axis   = -1,
              bn_name   = None,
              
              use_mask  = False,
              pooling   = None,
              pool_size = 2,
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
            momentum = momentum, epsilon = epsilon, axis = bn_axis, name = bn_name
        ))
    
    for i in range(n):
        args_i = [_get_var(a, i) for a in args]
        kwargs_i = {k : _get_var(v, i) for k, v in kwargs.items()}
        kwargs_i['name'] = '{}_{}'.format(name, i+1) if name and n > 1 else name
        
        if i > 0: kwargs_i.pop('input_shape', None)
        
        if use_manual_padding and kwargs_i.get('padding', None) == 'same':
            dim = '1d' if '1D' in layer_type.__name__ else '2d'
            try:
                pad_layer = _get_padding_layer(kwargs_i['kernel_size'], dim, use_mask = use_mask)
                if pad_layer is not None: x = _add_layer(x, pad_layer)
                kwargs_i['padding'] = 'valid'
            except NotImplementedError:
                logger.warning('manual padding is not supported for this dimension ({}) yet'.format(dim))
        
        x = _add_layer(x, layer_type(* args_i, ** kwargs_i))
        
        if activation and i < n - 1:
            x = _add_layer(x, get_activation(activation, ** activation_kwargs))
    
    if bnorm == 'after':
        x = _add_layer(x, BatchNormalization(
            momentum = momentum, epsilon = epsilon, axis = bn_axis, name = bn_name
        ))
    
    if residual and tuple(x.shape) == tuple(model.shape):
        x = Add()([x, model])
    elif residual:
        logger.info("Skip connection failed : shape mismatch ({} vs {})".format(
            model.shape, x.shape
        ))
    
    if activation:
        x = _add_layer(x, get_activation(activation, ** activation_kwargs))
    
    if pooling not in (None, False, 'none') and 'Conv' in layer_type.__name__:
        dim = '1d' if '1D' in layer_type.__name__ else '2d'
        x = _add_layer(x, _get_pooling_layer(
            pooling, dim, pool_size = pool_size,
            strides = pool_strides, padding = pool_padding, use_mask = use_mask
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

def MaskedConv1DBN(x, filters, *args, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    x = _layer_bn(x, MaskedConv1D, n, * args, ** kwargs)
    
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
