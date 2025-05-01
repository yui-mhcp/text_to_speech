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

import os
import re
import sys
import enum
import logging
import warnings
import collections
import numpy as np

class PartialInitializer(enum.IntEnum):
    NONE    = -1
    ZEROS   = 0
    ONES    = 1
    NORMAL  = 2
    NORMAL_CONDITIONNED = 3
    UNIFORM = 4
    UNIFORM_CONDITIONNED    = 5

class PartialSampling(enum.IntEnum):
    NONE    = 0
    RANDOM  = 1
    MAX_STD = 2
    MAX_MEAN    = 3

logger = logging.getLogger(__name__)

def _split_attn(values, keys = ['query', 'key', 'value']):
    axis = 0 if values[0].shape[0] > values[0].shape[1] else 1
    splitted = {
        k : [w] for k, w in zip(keys, np.split(values[0], len(keys), axis = axis))
    }
    if len(values) == 2:
        for k, b in zip(keys, np.split(np.squeeze(values[1]), len(keys))):
            splitted[k].append(b)
    return splitted

_attn_patterns = {'/v_' : '/value_', '/q_' : '/query_', '/k_' : '/key_', '/c_' : '/output_'}
_transformer_patterns = {
    '/mlp' : '/ffn', '/(layers|resblocks)/' : '/layer_', '/(self_)?attn' : '/mha',
    'c_fc$' : 'dense_1', 'c_proj$' : 'dense_2', '_projection' : '_layer', '_proj' : '_layer',
    '/(ln\_|norm)1' : '/norm_input', '/(ln|LayerNorm)_' : '/norm_', '/proj$' : 'output_layer',
    # for BART
    '(?<=\d)/final_layer_norm' : '/norm',
    'encoder_(attn|mha)' : 'enc_mha',
    'mha_layer_norm$' : 'mha/norm_output',
    '/model(\.|/)shared' : '/encoder/token_embedding',
    # BERT
    '/word_'    : '/token_',
    '/attention(/self)?'    : '/mha',
    '/intermediate/dense'   : '/ffn/dense_1',
    '(?<=\d)/output/dense$' : '/ffn/dense_2',
    'output/dense$'         : 'output_layer',
    # GPT-2
    '(/h_\._|/h/)' : '/layer_',
    'wte$'  : 'token_embedding',
    'wpe$'  : 'position_embedding',
    'mha/dense_.*' : 'mha/output_layer',
    'mha/c_attn$'  : 'mha',
    # ClipCap
    'to_queries'   : 'query_layer',
    'linear$'      : 'prefix_layer'
}

_attn_split     = {
    '(mha$|in_proj|query_key_value)' : lambda key, values: {
        key.replace('mha' if 'mha' in key else 'query_key_value', 'mha/{}_layer'.format(qkv)) : v
        for qkv, v in _split_attn(values, keys = ['query', 'key', 'value']).items()
    },
    'to_keys_values$' : lambda key, values: {
        key.replace('to_keys_values', '/{}_layer'.format(kv)) : v
        for kv, v in _split_attn(values, keys = ['key', 'value']).items()
    }
}

_int_distance = {
    str(i) : {str(j) : abs(i - j) / 10. for j in range(10)}
    for i in range(10)
}

def _is_torch_module(x):
    """ Returns whether `x` is a `nn.Module` but not a `keras.Model` """
    if 'torch' not in sys.modules: return False
    import keras
    import torch.nn
    if isinstance(x, dict):
        return all(isinstance(v, torch.Tensor) for v in x.values())
    return isinstance(x, torch.nn.Module) and not isinstance(x, (keras.Model, keras.layers.Layer))

def load_saved_model_variables(path):
    """ Returns a mapping `{var_name : variable}` from the given `tf.saved_model` `path` """
    import tensorflow as tf
    
    if os.path.isdir(path): path = tf.train.latest_checkpoint(path)
    names   = tf.train.list_variables(path)
    weights = {n : tf.train.load_variable(path, n) for n, _ in names}
    weights = {k : v for k, v in weights.items() if isinstance(v, np.ndarray) and v.ndim}
    logger.info('Loaded {} variables from {}'.format(len(weights), path))
    return {
        k.replace('/.ATTRIBUTES/VARIABLE_VALUE', '') : v for k, v in weights.items()
    }

def get_model_name(variables, model = None, threshold = 0.6):
    """
        Infers the model's name based on its variables. If `model` is provided, return `model.name`
    """
    if model is not None and hasattr(model, 'name'): return model.name
    
    parts = {}
    for name in variables:
        candidate = name.split('/')[0]
        parts.setdefault(candidate, 0)
        parts[candidate] += 1
    
    for cand, n in parts.items():
        if n > len(variables) * threshold: return cand
    warnings.warn('Unable to determine the root based on candidates : {}'.format(parts))
    return ''

def remove_model_name(variables, ** kwargs):
    """ Updates `variables` to remove the model's names from the start of variable names """
    model_name = get_model_name(variables, ** kwargs)
    if model_name:
        variables = {
            k[len(model_name) + 1 :] if k.startswith(model_name) else k : v
            for k, v in variables.items()
        }
    
    return variables

def variable_to_numpy(var):
    """ Converts `var` to a `np.ndarray` by calling `detach, cpu, numpy` sequentially (if defined) """
    if isinstance(var, dict): return {k : variable_to_numpy(v) for k, v in var.items()}
    if isinstance(var, list): return [variable_to_numpy(v) for v in var]
    
    for attr in ('detach', 'cpu', 'numpy'):
        if hasattr(var, attr): var = getattr(var, attr)()
    return var

def normalize_var_name(name):
    """ Replaces '.' and '-' by '/' """
    if isinstance(name, dict): return {normalize_var_name(k) : v for k, v in name.items()}
    return name.replace('.', '/').replace('-', '/')

def get_var_name(var):
    """ Returns the variable name (either `path` or `name` attribute) """
    return var.path if hasattr(var, 'path') else var.name

def get_var_mapping(model):
    """ Returns a dict `{var_name : var}` """
    if hasattr(model, 'state_dict'):    model = model.state_dict()
    elif hasattr(model, 'weights'):     model = model.weights
    if isinstance(model, dict):
        return model

    mapping = collections.OrderedDict()
    for v in model: mapping[get_var_name(v)] = v
    return mapping

def get_variables(model):
    """ Returns a `list` of model variables """
    return list(get_var_mapping(model).values())

def get_weights(model):
    """ Returns a `list` of model variables (as `np.ndarray`) """
    if hasattr(model, 'get_weights'): return model.get_weights()
    return variable_to_numpy(get_variables(model))

def get_layer_name(name):
    """ Returns the layer's name based on variable's name `name` """
    if '/' not in name: return name
    return '/'.join(name.split('/')[:-1])

def get_layers_mapping(model,
                       *,
                       
                       transpose    = False,
                       to_numpy     = True,
                       skip_root    = False,
                       
                       source   = None,
                       ** _
                      ):
    """
        Returns a dict `{layer_name : list_of_vars}`
        A layer is identified by removing the last part from its name
        Layer's name parts are identified by splitting the name by '/' after normalizing the names
        
        Arguments :
            - model : a valid type for `get_var_mapping` (`dict, list, Model, torch.nn.Module`)
            
            - transpose : whether to transpose the weights
            - to_numpy  : whether to convert weights to `np.ndarray`
            - skip_root : whether to remove model's name for matching
            
            - source    : the source format (see `arrange_weights` for more information)
        Return :
            - mapping   : a `dict` of `{layer_name : list_of_layer_variables}`
    """
    layers = model
    if not isinstance(model, dict) or not isinstance(list(model.values())[0], list):
        variables   = get_var_mapping(model)
        variables   = normalize_var_name(variables)
        if skip_root:   variables = remove_model_name(variables, model = model)
        if to_numpy:    variables = variable_to_numpy(variables)
        
        layers  = collections.OrderedDict()
        for name, var in variables.items():
            key = get_layer_name(name)
            layers.setdefault(key, []).append(var)
    
    if source:      layers = arrange_weights(layers, source, target = 'keras')
    if transpose:   layers = transpose_weights(layers)
    
    return layers

get_layers = get_layers_mapping

def print_vars(model, ** kwargs):
    """ Displays all variables of `model` (name with shape) """
    variables = get_var_mapping(model, ** kwargs)

    msg = '# variables : {}'.format(len(variables))
    for name, var in variables.items():
        msg += '\nName : {}\t- Shape : {}'.format(name, tuple(var.shape))
    msg += '\n\n'
    print(msg)

def print_layers(model, ** kwargs):
    """ Displays all layers of `model` (name with shape of each variable) """
    layers = get_layers_mapping(model, to_numpy = False, ** kwargs)

    msg = '# layers : {}'.format(len(layers))
    for name, list_vars in layers.items():
        msg += '\nName : {}\t- Shape : {}'.format(
            name, [tuple(var.shape) for var in list_vars]
        )
    msg += '\n\n'
    print(msg)


def transpose_weights(weights):
    """ Returns the transposed version of `weights` (for `torch` to `keras` convertion) """
    if isinstance(weights, dict):
        return {k : transpose_weights(v) for k, v in weights.items()}
    
    if isinstance(weights, list):
        return [transpose_weights(w) for w in weights]
    
    if len(weights.shape) <= 1:
        return weights
    elif len(weights.shape) == 2:   # Dense weights
        return weights.T
    elif len(weights.shape) == 3:   # Conv1D weights
        return np.transpose(weights, [2, 1, 0])
    elif len(weights.shape) == 4:   # Conv2D weights
        return np.transpose(weights, [2, 3, 1, 0])
    elif len(weights.shape) == 5:   # Conv3D weights
        return np.transpose(weights, [2, 3, 4, 1, 0])
    else:
        raise ValueError("Unknown weights shape : {}".format(weights.shape))

def arrange_weights(weights, source, target = 'keras'):
    """ Rearrange `weights` from `source` format to `target` format """
    if source == target: return weights
    
    if isinstance(weights, dict):
        rearranged = {}
        for k, w in weights.items():
            new_v = arrange_weights(w, source, target)
            if not isinstance(new_v, dict):
                rearranged[k] = new_v
            else:
                rearranged.update({'{}-{}'.format(k, ki) : vi for ki, vi in new_v.items()})

        return rearranged
    
    if target == 'torch':
        if source == 'saved_model':
            weights = arrange_saved_model_weights(weights)
        return arrange_keras_weights(weights)
    elif target == 'keras':
        if source == 'torch':
            return arrange_torch_weights(weights)
        elif source == 'saved_model':
            return arrange_saved_model_weights(weights)
    else:
        raise ValueError('Unsupported target format : {}'.format(target))

def arrange_torch_weights(weights, expand_bidirectional = True):
    if len(weights) == 2:   # In keras, the bias should be the 2nd variable
        weights = sorted(weights, key = lambda w: len(w.shape), reverse = True)
    elif len(weights) < 4:
        pass
    elif len(weights) == 4: # LSTM layer
        weights = weights[:2] + [weights[2] + weights[3]]
    elif len(weights) == 5: # BatchNormalization layer
        weights = weights[:4]
    elif len(weights) == 8: # Bidirectional layer
        if expand_bidirectional:
            return {
                'forward'   : weights[:2] + [weights[2] + weights[3]],
                'backward'  : weights[4:6] + [weights[6] + weights[7]]
            }
        else:
            weights = weights[:2] + [weights[2] + weights[3]] + weights[4:6] + [weights[6] + weights[7]]
    else:
        raise ValueError("Unknown weights length : {}\n  Shapes : {}".format(
            len(weights), [tuple(v.shape) for v in weights]
        ))
    
    return weights

def arrange_keras_weights(weights):
    if len(weights) < 3 or len(weights) == 4:
        pass
    elif len(weights) == 3: # LSTM layer
        weights = weights[:2] + [weights[2] / 2., weights[2] / 2.]
    else:
        raise ValueError("Unknown weights length : {}\n  Shapes : {}".format(
            len(weights), [tuple(v.shape) for v in weights]
        ))
    
    return weights

def arrange_saved_model_weights(weights):
    if len(weights) < 4:
        return sorted(weights, key = lambda w: len(w.shape), reverse = True)
    elif len(weights) == 4: # Normalization layer
        return [weights[1], weights[0]] + weights[2:]
    raise ValueError("Unknown weights length : {}\n  Shapes : {}".format(
        len(weights), [tuple(v.shape) for v in weights]
    ))

def find_layers_mapping(model,
                        pretrained,
                        *,
                        
                        patterns    = {},
                        transforms  = {},
                        skip_layers = None,
                        
                        source  = None,
                        transpose   = False,
                        partial     = False,
                        
                        replacement_cost    = _int_distance,
                        default_replace_cost    = 1.5,
                        
                        tqdm = lambda x: x,
                        ** kwargs
                       ):
    """
        Maps 2 dict together with the keys' edit_distance as mapping criterion. 
        Each key from `tf_layers` is mapped to the key(s) with the best edit_distance from `pt_layers` and with valid weights' shapes*
        * a valid weight shape is either defined by the number of dimensions (if `partial = True`) or by the number of dimensions and shapes
        
        Arguments : 
            - tf_model : the target model
            - pt_model : the pretrained model
            - patterns : a mapping where the keys are regex pattern (or simply substrings) and the
                         value is the replacement value (i.e. the value to replace the key pattern)
            - transforms : a key-value mapping where keys are regex pattern (or simply substrings) and the 
                           value is a callable returning a new dict of layers' mapping (key is layer's name and values are the new weights)
                           This can be used to apply transformation (such as splitting) some keys in the pretrained model's mapping
            - skip_layers   : list of layers' name to skip (in the target model)
            - skip_root     : whether to skip root's name in `pt_model` layer's names
            - convert_to    : whether to convert layer's weights to another library format
            - default_replace_cost  : used to compute `edit_distance`'s score
        Returns : (mapping, pt_layers)
            - mapping   : `dict` where keys are keys from `tf_model` and values are a list of possible matching keys from `pt_layers`
            - pt_layers : the modified version of `pt_layers` (after `patterns` and `transforms`) to make the mapping between `mapping`'s values and `pt_model`'s variables (weights)
        
        Note : both `patterns` is applied after converting '.' to '/' in `pt_layers`
        Note 2 : `transforms` is applied after replacing `patterns` so make sure to use modified names in `transforms`
        Note 3 : In theory this function has been designed to map tensorflow's weights (tf_model) to pytorch's weights (pt_model)
                 but, in practice, both can be either tensorflow or pytorch models
    """
    from utils.text import edit_distance
    
    shape_fn  = sorted if not partial else len
    
    model_layers    = get_layers(model, to_numpy = False)
    model_layers    = {
        k : [shape_fn(tuple([s for s in vi.shape if s > 1])) for vi in v]
        for k, v in model_layers.items()
    }
    logger.debug('# model layers      : {}'.format(len(model_layers)))
    pretrained_layers = get_layers(
        pretrained, transpose = transpose, source = source, ** kwargs
    ).copy()
    logger.debug('# pretrained layers : {}'.format(len(pretrained_layers)))

    if patterns:
        for pat, repl in patterns.items():
            pretrained_layers = {re.sub(pat, repl, k) : v for k, v in pretrained_layers.items()}
    
    if transforms:
        for pat, trans in transforms.items():
            for k in list(pretrained_layers.keys()):
                if re.search(pat, k):
                    logger.debug('Applying transform {} on {} (shapes : {})'.format(
                        pat, k, [tuple(vi.shape) for vi in pretrained_layers[k]]
                    ))
                    pretrained_layers.update(trans(k, pretrained_layers.pop(k)))

    not_mapped = set(pretrained_layers.keys())
    pretrained_shapes  = {
        k : [shape_fn(np.squeeze(vi).shape) for vi in v if len(vi.shape)]
        for k, v in pretrained_layers.items()
    }

    def _remove_candidate(name):
        not_mapped.remove(name)
        for l in _layers[:i]:
            if len(mapping.get(l, [])) > 1 and name in mapping[l]:
                mapping[l].remove(name)
                if len(mapping[l]) == 1: _remove_candidate(mapping[l][0])
    
    _layers = list(model_layers.keys())
    mapping = {}
    for i, l1 in enumerate(tqdm(_layers)):
        shape = model_layers[l1]
        if skip_layers and any(re.search(s, l1) is not None for s in skip_layers):
            continue
        
        if l1 in not_mapped:
            mapping[l1] = [l1]
            _remove_candidate(l1)
            continue

        bests, score = [], float('inf')
        
        for l2 in not_mapped:
            if shape != pretrained_shapes[l2]: continue

            s = edit_distance(
                l1, l2,
                normalize   = False,
                replacement_cost    = replacement_cost,
                default_replace_cost    = default_replace_cost,
                ** kwargs
            )
            if s == score:
                bests.append(l2)
            elif s < score:
                bests = [l2]
                score = s
        
        mapping[l1] = bests
        if len(bests) == 1: _remove_candidate(bests[0])

    return mapping, pretrained_layers

def name_based_partial_transfer_learning(target_model,
                                         pretrained_model,
                                         
                                         source = None,
                                         transpose  = 'auto',
                                         
                                         partial_transfer      = True,
                                         partial_initializer   = 'zeros',
                                         
                                         sampling_mode  = None,
                                         
                                         tqdm   = lambda x: x,
                                         verbose    = False,
                                         ** kwargs
                                        ):
    """
        Make transfer learning on model with either : 
            - different number of layers (and same shapes for some layers)
            - different shapes (and same number of layers)
            
        Arguments : 
            - target_model  : Model instance (model where weights will be transfered to)
            - pretrained_model  : pretrained model to transfer weights from
            - partial_transfer  : whether to perform partial transfer for layers with different shapes
            - partial_initializer   : how to initialize weights when shapes differ
            - kwargs    : forwarded to `find_layers_mapping`
        
        Note : see `help(find_layers_mapping)` for more information about mappings' creation
    """
    import pandas as pd
    
    from utils.generic_utils import get_enum_item
    
    def partial_weight_transfer(target, pretrained_v):
        if target.shape == pretrained_v.shape:
            return pretrained_v
        elif target.shape == np.squeeze(pretrained_v).shape:
            return np.squeeze(pretrained_v), {'transform' : 'squeeze'}
        elif pretrained_v.shape == np.squeeze(target).shape:
            return np.broadcast_to(pretrained_v, target.shape), {'transform' : 'expand_dims'}
        elif len(target.shape) == 2 and target.shape == pretrained_v.T.shape:
            return pretrained_v.T, {'transform' : 'T'}
        elif not partial_transfer:
            logger.warning('Variable shapes missmatch ({} vs {}) and `partial_transfer = False`, leaving the variable as is'.format(target.shape, pretrained_v.shape))
            return target
        
        logger.log(
            logging.INFO if verbose else logging.DEBUG,
            "Shapes missmatch ({} vs {}), making partial transfer !".format(
                target.shape, pretrained_v.shape
            )
        )
        
        v = target
        if partial_initializer == PartialInitializer.ZEROS:
            v = np.zeros_like(target)
        elif partial_initializer == PartialInitializer.ONES:
            v = np.ones_like(target)
        elif partial_initializer == PartialInitializer.NORMAL:
            v = np.random.normal(size = target.shape)
        elif partial_initializer == PartialInitializer.NORMAL_CONDITIONNED:
            v = np.random.normal(
                loc = np.mean(pretrained_v), scale = np.std(pretrained_v), size = target.shape
            )
        elif partial_initializer == PartialInitializer.UNIFORM:
            v = np.random.uniform(size = target.shape)
        elif partial_initializer == PartialInitializer.UNIFORM_CONDITIONNED:
            v = np.random.uniform(
                minval = np.min(pretrained_v), maxval = np.max(pretrained_v), size = target.shape
            )
        
        if target.shape[0] < pretrained_v.shape[0] and sampling_mode != PartialSampling.NONE:
            if sampling_mode == PartialSampling.RANDOM:
                np.random.shuffle(pretrained_v)
            else:
                flat = pretrained_v if pretrained_v.ndim == 2 else np.reshape(
                    pretrained_v, [len(pretrained_v), -1]
                )
                if sampling_mode == PartialSampling.MAX_STD:
                    indexes = np.argsort(np.std(flat, axis = -1))[::-1]
                elif sampling_mode == PartialSampling.MAX_MEAN:
                    indexes = np.argsort(np.mean(flat, axis = -1))[::-1]
                
                indexes[: target.shape[0]] = np.sort(indexes[: target.shape[0]])
                pretrained_v = pretrained_v[indexes]
                
        slices = tuple([
            slice(0, min(t_s, p_s)) for t_s, p_s in zip(target.shape, pretrained_v.shape)
        ])
        v[slices] = pretrained_v[slices]
        
        return v, {'transform' : 'partial'}
    
    sampling_mode   = get_enum_item(str(sampling_mode), PartialSampling)
    partial_initializer = get_enum_item(str(partial_initializer), PartialInitializer)
    
    if source is None:      source = 'torch' if _is_torch_module(pretrained_model) else 'keras'
    if transpose == 'auto': transpose  = source == 'torch'
    
    mapping, pretrained_layers = find_layers_mapping(
        target_model,
        pretrained_model,
        source  = source,
        transpose   = transpose,
        partial     = True,
        tqdm    = tqdm,
        ** kwargs
    )
    pretrained_layers   = {k : v for k, v in pretrained_layers.items() if len(v) > 0}
    layer_var_idx = {k : 0 for k in pretrained_layers.keys()}

    no_map      = [k for k, v in mapping.items() if len(v) == 0]
    multi_map   = [k for k, v in mapping.items() if len(v) > 1]

    if no_map or multi_map:
        if no_map and not partial_transfer:
            raise ValueError('Some layers do not have any mapping !\n  Layers : {}'.format(no_map))
        if multi_map:
            raise ValueError('Some layers have multiple mapping, please try to remove ambiguity !\n  Layers :\n{}\n\n  Mapping :\n{}\n\n  All layers :\n{}'.format(
                '\n'.join('- {} : {}'.format(k, mapping[k]) for k in multi_map),
                '\n'.join('- {} : {}'.format(k, v) for k, v in mapping.items() if len(v) <= 1),
                '\n'.join('- {} : {}'.format(k, [vi.shape for vi in v]) for k, v in pretrained_layers.items())
            ))
    
    target_variables    = get_var_mapping(target_model)
    all_var_names       = list(target_variables.keys())
    
    mapping_infos   = {}
    new_weights     = []
    for i, (name, v) in enumerate(target_variables.items()):
        var_layer   = normalize_var_name(get_layer_name(name))

        mapping_infos[name] = {'layer' : var_layer, 'shape' : tuple(v.shape)}

        map_layer   = mapping.get(var_layer, mapping.get(name, []))
        if len(map_layer) == 0:
            logger.info('Variable {} from layer {} does not have any mapping : re-using its current weights'.format(name, var_layer))
            new_weights.append(variable_to_numpy(v))
            continue
        
        map_layer   = map_layer[0]
        
        if layer_var_idx[map_layer] >= len(pretrained_layers[map_layer]):
            logger.warning('Try to get variable {} from layer {} for variable {} !'.format(
                layer_var_idx[map_layer], map_layer, var_layer
            ))
            layer_var_idx[map_layer] -= 1
        
        map_weight  = pretrained_layers[map_layer][layer_var_idx[map_layer]]
        layer_var_idx[map_layer] += 1
        
        mapping_infos[name].update({
            'Map layer' : map_layer, 'Map shape' : tuple(map_weight.shape)
        })
        
        new_weight  = partial_weight_transfer(v, map_weight)
        if isinstance(new_weight, tuple):
            new_weight, info = new_weight
            mapping_infos[name].update(info)
        new_weights.append(new_weight)
    
    if any(w_idx not in (0, len(pretrained_layers[k])) for k, w_idx in layer_var_idx.items()):
        logger.warning('Some layers have unused weights !\n{}'.format(
            '\n'.join([
                '- {}\t{} used out of {} weights'.format(k, w_idx, len(pretrained_layers[k]))
                for k, w_idx in layer_var_idx.items() if w_idx not in (0, len(pretrained_layers[k]))
            ])
        ))
    
    mapping_infos = pd.DataFrame(mapping_infos).T
    
    logger.log(logging.INFO if verbose else logging.DEBUG, mapping_infos)
    
    target_model.set_weights(new_weights)
    logger.info("Weights transfered successfully !")
    return mapping_infos

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
            - target_model  : Model instance (model where weights will be transfered to)
            - pretrained_model  : Model or list of weights (pretrained)
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

        logger.log(
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
            logger.info("Variables {} shapes mismatch ({} vs {}), skipping it".format(idx_a, v.shape, pretrained_v.shape))
            
            new_v = v
        else:            
            logger.info("Variables {} shapes mismatch ({} vs {}), making partial transfer".format(idx_a, v.shape, pretrained_v.shape))
            
            new_v = partial_weight_transfer(v, pretrained_v)

        new_weights.append(new_v)
        idx_a, idx_b = idx_a + 1, idx_b + 1
    
    if idx_a != len(target_variables) and idx_b == len(pretrained_variables):
        logger.warning('All variables of pretrained model have been consumed but some variables remain in the new model !\n  Model A : length : {} - variables consummed : {}\n  Model B (pretrained) : length : {} - variables consummed : {}'.format(len(target_variables), idx_a, len(pretrained_variables), idx_b))
        new_weights.extend([
            v.numpy() if hasattr(v, 'numpy') else v for v in target_variables[idx_a:]
        ])
    elif idx_a != len(target_variables) and idx_b != len(pretrained_variables):
        raise ValueError("All variables of a model have not been consummed\n  Model A : length : {} - variables consummed : {}\n  Model B (pretrained) : length : {} - variables consummed : {}".format(len(target_variables), idx_a, len(pretrained_variables), idx_b))
    
    target_model.set_weights(new_weights)
    logger.info("Weights transfered successfully !")
