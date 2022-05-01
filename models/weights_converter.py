
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
import numpy as np

def print_vars(model):
    variables = model.variables if not isinstance(model, list) else model
    print("# variables : {}".format(len(variables)))
    for v in variables:
        print("Name : {}\t- Shape : {}".format(v.name, v.shape))
    print('\n\n')

def transpose_weights(weights):
    if len(weights.shape) <= 1:
        return weights
    if len(weights.shape) == 2:
        return weights.T
    if len(weights.shape) == 3:
        return np.transpose(weights, [2, 1, 0])
    else:
        raise ValueError("Unknown weights shape : {}".format(weights.shape))

""" Pytorch to Tensorflow convertion """

def get_pt_layers(pt_model):
    layers = {}
    state_dict = pt_model.state_dict() if not isinstance(pt_model, dict) else pt_model
    for k, v in state_dict.items():
        layer_name = '.'.join(k.split('.')[:-1])
        if layer_name not in layers: layers[layer_name] = []
        if hasattr(v, 'cpu'): v = v.cpu()
        layers[layer_name].append(v.numpy())
    return layers

def pt_convert_layer_weights(layer_weights):
    new_weights = []
    if len(layer_weights) == 2:
        new_weights = sorted(layer_weights, key = lambda w: len(w.shape), reverse = True)
    elif len(layer_weights) < 4:
        new_weights = layer_weights
    elif len(layer_weights) == 4:
        new_weights = layer_weights[:2] + [layer_weights[2] + layer_weights[3]]
    elif len(layer_weights) == 5:
        new_weights = layer_weights[:4]
    elif len(layer_weights) == 8:
        new_weights = layer_weights[:2] + [layer_weights[2] + layer_weights[3]]
        new_weights += layer_weights[4:6] + [layer_weights[6] + layer_weights[7]]
    else:
        raise ValueError("Unknown weights length : {}\n  Shapes : {}".format(len(layer_weights), [tuple(v.shape) for v in layer_weights]))
    
    return [transpose_weights(w) for w in new_weights]

def get_pt_variables(pt_model, verbose = False):
    pt_layers = get_pt_layers(pt_model)
    converted_weights = []
    for layer_name, layer_variables in pt_layers.items():
        converted_variables = pt_convert_layer_weights(layer_variables) if 'embedding' not in layer_name else layer_variables
        converted_weights += converted_variables
        
        logging.log(logging.INFO if verbose else logging.DEBUG, "Layer : {} \t {} \t {}".format(
            layer_name, 
            [tuple(v.shape) for v in layer_variables],
            [tuple(v.shape) for v in converted_variables],
        ))
    return converted_weights

def pt_convert_model_weights(pt_model, tf_model, verbose = False):
    converted_weights = get_pt_variables(pt_model, verbose = verbose)
    
    partial_transfer_learning(tf_model, converted_weights, verbose = verbose)
    logging.info("Weights converted successfully !")
    
    
""" Tensorflow to Pytorch converter """

def get_tf_layers(tf_model):
    layers = {}
    variables = tf_model.variables if not isinstance(tf_model, list) else tf_model
    for v in variables:
        layer_name = '/'.join(v.name.split('/')[:-1])
        if layer_name not in layers: layers[layer_name] = []
        layers[layer_name].append(v.numpy())
    return layers

def tf_convert_layer_weights(layer_weights):
    new_weights = []
    if len(layer_weights) < 3 or len(layer_weights) == 4:
        new_weights = layer_weights
    elif len(layer_weights) == 3:
        new_weights = layer_weights[:2] + [layer_weights[2] / 2., layer_weights[2] / 2.]
    else:
        raise ValueError("Unknown weights length : {}\n  Shapes : {}".format(len(layer_weights), [tuple(v.shape) for v in layer_weights]))
    
    return [transpose_weights(w) for w in new_weights]


def tf_convert_model_weights(tf_model, pt_model, verbose = False):
    import torch
    
    pt_layers = pt_model.state_dict()
    tf_layers = get_tf_layers(tf_model)
    converted_weights = []
    for layer_name, layer_variables in tf_layers.items():
        converted_variables = tf_convert_layer_weights(layer_variables) if 'embedding' not in layer_name else layer_variables
        converted_weights += converted_variables
        
        logging.log(logging.INFO if verbose else logging.DEBUG, "Layer : {} \t {} \t {}".format(
            layer_name, 
            [tuple(v.shape) for v in layer_variables],
            [tuple(v.shape) for v in converted_variables],
        ))
    
    tf_idx = 0
    for i, (pt_name, pt_weights) in enumerate(pt_layers.items()):
        if len(pt_weights.shape) == 0: continue
        
        pt_weights.data = torch.from_numpy(converted_weights[tf_idx])
        tf_idx += 1
    
    pt_model.load_state_dict(pt_layers)
    logging.info("Weights converted successfully !")

""" Partial transfer learning """

def partial_transfer_learning(target_model, 
                              pretrained_model, 
                              partial_transfer      = True,
                              partial_initializer   = 'zeros',
                              verbose       = False
                             ):
    """
        Make transfer learning on model with either : 
            - different number of layers (and same shapes for some layers)
            - different shapes (and same number of layers)
            
        Arguments : 
            - target_model  : tf.keras.Model instance (model where weights will be transfered to)
            - pretrained_model  : tf.keras.Model or list of weights (pretrained)
            - partial_transfer : whether to do partial transfer for layers with different shapes (only relevant if 2 models have same number of layers)
    """
    assert partial_initializer in (None, 'zeros', 'ones', 'normal', 'normal_conditionned')
    def partial_weight_transfer(target, pretrained_v):
        v = target
        if partial_initializer == 'zeros':
            v = np.zeros_like(target)
        elif partial_initializer == 'ones':
            v = np.ones_like(target)
        elif partial_initializer == 'normal_conditionned':
            v = np.random.normal(loc = np.mean(pretrained_v), scale = np.std(pretrained_v), size = target.shape)
        elif partial_initializer == 'normal':
            v = np.random.normal(size = target.shape)

        
        if v.ndim == 1:
            max_0 = min(v.shape[0], pretrained_v.shape[0])
            v[:max_0] = pretrained_v[:max_0]
        elif v.ndim == 2:
            max_0 = min(v.shape[0], pretrained_v.shape[0])
            max_1 = min(v.shape[1], pretrained_v.shape[1])
            v[:max_0, :max_1] = pretrained_v[:max_0, :max_1]
        elif v.ndim == 3:
            max_0 = min(v.shape[0], pretrained_v.shape[0])
            max_1 = min(v.shape[1], pretrained_v.shape[1])
            max_2 = min(v.shape[2], pretrained_v.shape[2])
            v[:max_0, :max_1, :max_2] = pretrained_v[:max_0, :max_1, :max_2]
        elif v.ndim == 4:
            max_0 = min(v.shape[0], pretrained_v.shape[0])
            max_1 = min(v.shape[1], pretrained_v.shape[1])
            max_2 = min(v.shape[2], pretrained_v.shape[2])
            max_3 = min(v.shape[3], pretrained_v.shape[3])
            v[:max_0, :max_1, :max_2, :max_3] = pretrained_v[:max_0, :max_1, :max_2, :max_3]
        else:
            raise ValueError("Variable dims > 4 non géré !")
        
        return v
    
    target_variables = target_model.variables
    pretrained_variables = pretrained_model.variables if not isinstance(pretrained_model, list) else pretrained_model
    
    skip_layer = len(target_variables) != len(pretrained_variables)
    skip_from_a = None
    if skip_layer:
        skip_from_a = (len(target_variables) > len(pretrained_variables))
    
    new_weights = []
    idx_a, idx_b = 0, 0
    while idx_a < len(target_variables):
        v = target_variables[idx_a]
        if hasattr(v, 'numpy'): v = v.numpy()
        if idx_b == len(pretrained_variables):
            if not skip_layer or not skip_from_a: break
            idx_a += 1
            new_weights.append(v)
            continue
        else:
            pretrained_v = pretrained_variables[idx_b]
            if not isinstance(pretrained_v, np.ndarray): pretrained_v = pretrained_v.numpy()

        logging.log(
            logging.INFO if verbose else logging.DEBUG,
            "Target[{}] shape     : {}\nPretrained[{}] shape : {}".format(
                idx_a, v.shape, idx_b, pretrained_v.shape
            )
        )
            
        if v.shape != pretrained_v.shape and skip_layer:
            if skip_from_a: 
                idx_a += 1
                new_weights.append(v)
            else: idx_b += 1
            continue
        
        if len(v.shape) != len(pretrained_v.shape):
            raise ValueError("Number of dimension for variables {} differs !\n  Target shape : {}\n  Pretrained shape : {}".format(idx_a, v.shape, pretrained_v.shape))
                        
        new_v = None
        if v.shape == pretrained_v.shape:
            new_v = pretrained_v
        elif not partial_transfer:
            logging.info("Variables {} shapes mismatch ({} vs {}), skipping it".format(idx_a, v.shape, pretrained_v.shape))
            
            new_v = v
        else:            
            logging.info("Variables {} shapes mismatch ({} vs {}), making partial transfer".format(idx_a, v.shape, pretrained_v.shape))
            
            new_v = partial_weight_transfer(v, pretrained_v)

        new_weights.append(new_v)
        idx_a, idx_b = idx_a + 1, idx_b + 1
    
    if idx_a != len(target_variables) and idx_b == len(pretrained_variables):
        logging.warning('All variables of pretrained model have been consumed but some variables remain in the new model !\n  Model A : length : {} - variables consummed : {}\n  Model B (pretrained) : length : {} - variables consummed : {}'.format(len(target_variables), idx_a, len(pretrained_variables), idx_b))
        new_weights.extend([
            v.numpy() if hasattr(v, 'numpy') else v for v in target_variables[idx_a:]
        ])
    elif idx_a != len(target_variables) and idx_b != len(pretrained_variables):
        raise ValueError("All variables of a model have not been consummed\n  Model A : length : {} - variables consummed : {}\n  Model B (pretrained) : length : {} - variables consummed : {}".format(len(target_variables), idx_a, len(pretrained_variables), idx_b))
    
    target_model.set_weights(new_weights)
    logging.info("Weights transfered successfully !")
        