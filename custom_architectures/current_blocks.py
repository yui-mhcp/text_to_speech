
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
_concat_type    = (True, 'add', 'mul', 'concat')
_pool_type      = (None, False, 'none', 'max', 'avg', 'average', 'up', 'upsampling')

_str_layers     = {
    'dense'     : Dense,
    'conv1d'    : Conv1D,
    'masked_conv1d' : MaskedConv1D,
    'conv2d'    : Conv2D,
    'conv3d'    : Conv3D,
    'lstm'      : LSTM,
    'gru'       : GRU,
    'bi_lstm'   : lambda * args, ** kwargs: Bidirectional(LSTM(* args, ** kwargs)),
    'bi_gru'    : lambda * args, ** kwargs: Bidirectional(GRU(* args, ** kwargs))
}

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

def _get_concat_layer(concat_type, ** kwargs):
    assert concat_type in _concat_type, 'Invalid concat layer !\n  Accepted : {}\n  Got {}'.format(
        _concat_type, concat_type
    )
    
    if concat_type in (True, 'add'):  return Add(** kwargs)
    elif concat_type == 'mul':     return Multiply(** kwargs)
    elif concat_type == 'concat':     return Concatenate(** kwargs)

def _get_flatten_layer(flatten_type, dim, ** kwargs):
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
    dim = dim.lower()
    assert dim in ('1d', '2d', '3d'), 'Invalid dimension {}'.format(dim)
    assert flatten_type in _flatten_type, 'Invalid flatten layers !\n  Accepted : {}\n  Got {}'.format(
        _flatten_type, flatten_type
    )
    
    if flatten_type in (None, 'none'):  return Flatten(** kwargs)
    elif flatten_type == 'max':
        if dim == '1d':     return GlobalMaxPooling1D(** kwargs)
        elif dim == '2d':   return GlobalMaxPooling2D(** kwargs)
        elif dim == '3d':   return GlobalMaxPooling3D(** kwargs)
    elif flatten_type in ('mean', 'avg', 'average'):
        if dim == '1d':     return GlobalAveragePooling1D(** kwargs)
        elif dim == '2d':   return GlobalAveragePooling2D(** kwargs)
        elif dim == '3d':   return GlobalAveragePooling3D(** kwargs)
    else: # GRU / LSTM
        layer = LSTM(** kwargs) if 'lstm' in flatten_type else GRU(** kwargs)
        
        if 'bi' in flatten_type: layer = Bidirectional(layer)
        
        return layer

def _get_pooling_layer(pool_type, dim, * args, global_pooling = False, use_mask = None, ** kwargs):
    """ Returns a pooling layer : `{Max / Average}Pooling{dim}` or `UpSampling{dim}` """
    if isinstance(pool_type, (list, tuple)):
        return [
            _get_pooling_layer(pool, dim, * args, global_pooling = global_pooling, ** kwargs)
            for pool in pool_type
        ]
    
    dim = dim.lower()
    assert dim in ('1d', '2d', '3d'), 'Invalid dimension {}'.format(dim)
    assert pool_type in _pool_type, 'Invalid pooling layers !\n  Accepted : {}\n  Got {}'.format(
        _pool_type, pool_type
    )
    
    if pool_type in (None, False, 'none'): return None
    elif pool_type == 'max':
        if global_pooling:
            if dim == '1d':     pool_class = GlobalMaxPooling1D
            elif dim == '2d':   pool_class = GlobalMaxPooling2D
            elif dim == '3d':   pool_class = GlobalMaxPooling3D
        else:
            if dim == '1d':     pool_class = MaxPooling1D if not use_mask else MaskedMaxPooling1D
            elif dim == '2d':   pool_class = MaxPooling2D
            elif dim == '3d':   pool_class = MaxPooling3D
    
    elif pool_type in ('mean', 'avg', 'average'):
        if global_pooling:
            if dim == '1d':     pool_class = GlobalAveragePooling1D
            elif dim == '2d':   pool_class = GlobalAveragePooling2D
            elif dim == '3d':   pool_class = GlobalAveragePooling3D
        else:
            if dim == '1d':     pool_class = AveragePooling1D if not use_mask else MaskedAveragePooling1D
            elif dim == '2d':   pool_class = AveragePooling2D
            elif dim == '3d':   pool_class = AveragePooling3D
    
    elif pool_type in ('up', 'upsampling'):
        if global_pooling: raise ValueError("Upsampling does not exist in global pooling mode !")
        
        if dim == '1d':     pool_class = UpSampling1D
        elif dim == '2d':   pool_class = UpSampling2D
        elif dim == '3d':   pool_class = UpSampling3D
    
    return pool_class(* args, ** kwargs)

def _get_padding_layer(kernel_size, dim, * args, dilation_rate = 1, use_mask = None, ** kwargs):
    if isinstance(kernel_size, (list, tuple)): kernel_size = kernel_size[0]
    if dilation_rate > 1:
        raise NotImplementedError('not supported when `dilation_rate > 1`')
    if kernel_size % 2 == 0:
        raise NotImplementedError('not supported when `kernel_size % 2 == 0`')
    
    padding_half =  kernel_size // 2   
    if padding_half < 1: return None
    
    padding = (padding_half, padding_half)
    
    dim = dim.lower()
    assert dim in ('1d', '2d')
    
    if dim == '2d':
        raise NotImplementedError('not supported for 2D yet')
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
              pool_size = None,
              pool_strides  = None,
              pool_padding  = 'valid',
              
              activation    = None,
              activation_kwargs = {},
              drop_rate     = 0.1,
              
              residual      = False,
              residual_kwargs   = {},
              
              name  = None, 
              ** kwargs
             ):
    """
        Adds `n` times `layer_type` to `model` with `kwargs`, and possibly adds additional layers
        
        Arguments :
            - model         : either `tf.Tensor` (for the Functional API) either `Sequential` instance
            - layer_type    : the layer's class to add
            - n             : the number of `layer_type` to add consecutively
            - args / kwargs : configuration for the `layer_type`
            
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
            
            - name  : the name to use for the `layer_type`
        Returns : the updated `model` (if Sequential API) or the new `tf.Tensor`
    """
    assert bnorm in ('before', 'after', 'never'), 'Invalid `bnorm` : {}'.format(bnorm)
    assert pooling in _pool_type, '{} is not a valid pooling type'.format(pooling)
    
    if residual and isinstance(model, tf.keras.Sequential):
        raise ValueError('When `residual = True` `model` cannot be a `Sequential` !')
    
    x = model
    
    if bnorm == 'before':
        x = _add_layer(x, BatchNormalization(
            momentum = momentum, epsilon = epsilon, axis = bn_axis, name = bn_name
        ))
    
    for i in range(n):
        args_i      = [_get_var(a, i) for a in args]
        kwargs_i    = {k : _get_var(v, i) for k, v in kwargs.items()}
        kwargs_i['name'] = '{}_{}'.format(name, i + 1) if name and n > 1 else name
        
        if i > 0: kwargs_i.pop('input_shape', None)
        
        if use_manual_padding and kwargs_i.get('padding', None) == 'same':
            dim = _get_layer_dim(layer_type.__name__)
            try:
                pad_layer = _get_padding_layer(
                    kwargs_i['kernel_size'], dim, use_mask = use_mask,
                    dilation_rate = kwargs_i.get('dilation_rate', 1)
                )
                if pad_layer is not None: x = _add_layer(x, pad_layer)
                kwargs_i['padding'] = 'valid'
            except NotImplementedError as e:
                logger.warning('manual padding is not supported for {} (reason : {})'.format(
                    kwargs_i['name'], e
                ))
        
        x = _add_layer(x, layer_type(* args_i, ** kwargs_i))
        
        if activation and i < n - 1:
            x = _add_layer(x, get_activation(activation, ** activation_kwargs))
    
    if bnorm == 'after':
        x = _add_layer(x, BatchNormalization(
            momentum = momentum, epsilon = epsilon, axis = bn_axis, name = bn_name
        ))
    
    if residual and tuple(x.shape) == tuple(model.shape):
        x = _get_concat_layer(residual, ** residual_kwargs)([x, model])
    elif residual:
        logger.warning("Skip connection failed : shape mismatch ({} vs {})".format(
            model.shape, x.shape
        ))
    
    if activation:
        x = _add_layer(x, get_activation(activation, ** activation_kwargs))
    
    if pooling not in (None, False, 'none') and 'Conv' in layer_type.__name__:
        if pool_size is None and pool_strides is None: pool_size = 2
        if pool_size is None:       pool_size = pool_strides
        if pool_strides is None:    pool_strides = pool_size
        
        dim = _get_layer_dim(layer_type.__name__)
        x   = _add_layer(x, _get_pooling_layer(
            pooling, dim, pool_size = pool_size, strides = pool_strides,
            padding = pool_padding, use_mask = use_mask
        ))
    
    if drop_rate > 0.: x = _add_layer(x, Dropout(drop_rate))
    
    return x

def DenseBN(x, units, * args, ** kwargs):
    return _layer_bn(
        x, Dense, len(units) if isinstance(units, (list, tuple)) else 1, * args, units = units, ** kwargs
    )

def Conv1DBN(x, filters, * args, ** kwargs):
    return _layer_bn(
        x, Conv1D, len(filters) if isinstance(filters, (list, tuple)) else 1, * args,
        filters = filters, ** kwargs
    )

def Conv2DBN(x, filters, * args, ** kwargs):
    return _layer_bn(
        x, Conv2D, len(filters) if isinstance(filters, (list, tuple)) else 1, * args,
        filters = filters, ** kwargs
    )

def Conv3DBN(x, filters, * args, ** kwargs):
    return _layer_bn(
        x, Conv3D, len(filters) if isinstance(filters, (list, tuple)) else 1, * args,
        filters = filters, ** kwargs
    )

def MaskedConv1DBN(x, filters, * args, ** kwargs):
    kwargs.setdefault('use_mask', True)
    return _layer_bn(
        x, MaskedConv1D, len(filters) if isinstance(filters, (list, tuple)) else 1, * args,
        filters = filters, ** kwargs
    )

def SeparableConv1DBN(x, filters, * args, ** kwargs):
    return _layer_bn(
        x, SeparableConv1D, len(filters) if isinstance(filters, (list, tuple)) else 1, * args,
        filters = filters, ** kwargs
    )

def SeparableConv2DBN(x, filters, * args, ** kwargs):
    return _layer_bn(
        x, SeparableConv2D, len(filters) if isinstance(filters, (list, tuple)) else 1, * args,
        filters = filters, ** kwargs
    )

def SeparableConv3DBN(x, filters, * args, ** kwargs):
    return _layer_bn(
        x, SeparableConv3D, len(filters) if isinstance(filters, (list, tuple)) else 1, * args,
        filters = filters, ** kwargs
    )


def Conv1DTransposeBN(x, filters, * args, ** kwargs):
    return _layer_bn(
        x, Conv1DTranspose, len(filters) if isinstance(filters, (list, tuple)) else 1, * args,
        filters = filters, ** kwargs
    )

def Conv2DTransposeBN(x, filters, * args, ** kwargs):
    return _layer_bn(
        x, Conv2DTranspose, len(filters) if isinstance(filters, (list, tuple)) else 1, * args,
        filters = filters, ** kwargs
    )

def Conv3DTransposeBN(x, filters, * args, ** kwargs):
    return _layer_bn(
        x, Conv3DTranspose, len(filters) if isinstance(filters, (list, tuple)) else 1, * args,
        filters = filters, ** kwargs
    )
