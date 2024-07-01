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

import numpy as np

def get_signature(shape):
    import tensorflow as tf
    if isinstance(shape, list):
        return [tf.TensorSpec(shape = s[1:], dtype = 'float32') for s in shape]
    return tf.TensorSpec(shape = shape[1:], dtype = 'float32')

def describe_model(model, with_compile = True):
    des = ''
    try:
        des += "- Inputs \t: {}\n".format(model.input_shape)
        des += "- Outputs \t: {}\n".format(model.output_shape)
    except AttributeError:
        des += "- Inputs \t: unknown\n"
        des += "- Outputs \t: unknown\n"
    des += "- Number of layers \t: {}\n".format(len(model.layers))
    des += "- Number of parameters \t: "
    if model.count_params() >= 1000000:
        des += '{:,.3f} Millions\n'.format(model.count_params() / 1000000)
    else:
        des += '{}\n'.format(model.count_params())
    
    if model.compiled:
        if with_compile:
            des += optimizer_to_str(getattr(model, 'optimizer', None))
            des += loss_to_str(getattr(model, 'loss', None))
            des += metrics_to_str(getattr(model, 'compile_metrics', None))
    else:
        des += "- Model not compiled yet\n"

    return des

def optimizer_to_str(optimizer):
    return keras_object_to_str(optimizer, name = 'Optimizer')

def loss_to_str(loss):
    return keras_object_to_str(loss, name = 'Loss')

def metrics_to_str(metrics):
    return keras_object_to_str(metrics, name = 'Metric')

def keras_object_to_str(obj, name = None, format = '- {name} \t: {value}\n'):
    if isinstance(obj, list):
        if len(obj) == 1:
            return keras_object_to_str(obj[0] if len(obj) == 1 else None, name, format)
        obj = {'{} #{}'.format(name, i) : obj_i for i, obj_i in enumerate(obj)}
    
    if isinstance(obj, dict):
        return ''.join([
            keras_object_to_str(obj = v, name = k, format = format) for k, v in obj.items()
        ])
    
    if obj is None: return ''
    if isinstance(obj, str):    return format.format(name = name, value = obj)
    if hasattr(obj, 'get_config'):    return format.format(name = name, value = obj.get_config())
    if callable(obj):   return format.format(name = name, value = obj.__name__)

    raise ValueError('Unknown data type ({}) : {}'.format(type(obj), obj))

def infer_downsampling_factor(model):
    """ Based on a sequential model, computes an estimation of the downsampling factor """
    from keras.layers import (
        Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, MaxPooling3D,
        AveragePooling1D, AveragePooling2D, AveragePooling3D
    )
    _downsampling_types = [
        Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, MaxPooling3D,
        AveragePooling1D, AveragePooling2D, AveragePooling3D
    ]
    try:
        from custom_layers import MaskedConv1D, MaskedMaxPooling1D, MaskedAveragePooling1D
        _downsampling_types.extend([MaskedConv1D, MaskedMaxPooling1D, MaskedAveragePooling1D])
    except Exception as e:
        pass
    
    def _get_factor(model):
        factor = 1
        for l in model.layers:
            if type(l) in _downsampling_types:
                factor = factor * np.array(l.strides)
            elif hasattr(l, 'layers'):
                factor = factor * _get_factor(l)
        
        return factor
    return _get_factor(model)

def infer_upsampling_factor(model):
    """ Based on a sequential model, computes an estimation of the upsampling factor """
    from keras.layers import (
        UpSampling1D, UpSampling2D, UpSampling3D,
        Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
    )
    _downsampling_types = [
        UpSampling1D, UpSampling2D, UpSampling3D,
        Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
    ]
    try:
        from custom_architectures.east_arch import UpSampling2DWithAlignedCorners
        _downsampling_types.append(UpSampling2DWithAlignedCorners)
    except Exception as e:
        pass
    
    def _get_factor(model):
        factor = 1
        for l in model.layers:
            if type(l) in _downsampling_types:
                if hasattr(l, 'strides'):
                    strides = l.strides
                elif hasattr(l, 'size'):
                    strides = l.size
                elif hasattr(l, 'scale_factor'):
                    strides = l.scale_factor
                factor = factor * np.array(strides)
            elif hasattr(l, 'layers'):
                factor = factor * _get_factor(l)
        
        return factor
    return _get_factor(model)

def _get_tracked_type(value, types):
    if isinstance(value, (list, tuple)) and len(value) > 0: value = value[0]
    for t in types:
        if isinstance(value, t): return t
    return None