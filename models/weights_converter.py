
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

import re
import enum
import logging
import warnings
import collections
import numpy as np
import pandas as pd
import tensorflow as tf

class PartialInitializer(enum.IntEnum):
    NONE    = -1
    ZEROS   = 0
    ONES    = 1
    NORMAL  = 2
    NORMAL_CONDITIONNED = 3
    UNIFORM = 4
    UNIFORM_CONDITIONNED    = 5

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
    '(mha$|in_proj)' : lambda key, values: {
        key.replace('mha', 'mha/{}_layer'.format(qkv)) : v
        for qkv, v in _split_attn(values, keys = ['query', 'key', 'value']).items()
    },
    'to_keys_values$' : lambda key, values: {
        key.replace('to_keys_values', '/{}_layer'.format(kv)) : v
        for kv, v in _split_attn(values, keys = ['key', 'value']).items()
    }
}

def _get_layer_name_sep(vars_mapping):
    """ Get the layer's names separator ('.' for pytorch and '/' for tensorflow) """
    for name in vars_mapping:
        if '.' in name and '/' not in name:
            return '.'
        if '/' in name and '.' not in name:
            return '/'
    raise RuntimeError('Unable to determine separator for layer\'s names :\n{}'.format('\n'.join(vars_mapping.keys())))

def _get_root_name(vars_mapping, model = None, sep = None, threshold = 0.9):
    if model is not None and hasattr(model, 'name'): return model.name
    if not sep: sep = _get_layer_name_sep(vars_mapping)
    
    parts = {}
    for name in vars_mapping:
        candidate = name.split(sep)[0]
        parts.setdefault(candidate, 0)
        parts[candidate] += 1
    
    for cand, n in parts.items():
        if n > len(vars_mapping) * threshold: return cand
    warnings.warn('Unable to determine the root based on candidates : {}'.format(parts))
    return ''

def _get_layer_name(name, vars_mapping, skip_root = False, sep = None, root = None,
                    model = None, ** kwargs):
    """ Returns the layer's name based on variable's name `name` """
    def _is_root(part, root):
        if not root: root = _get_root_name(vars_mapping, model = model)
        return part == root
    
    if not sep: sep = _get_layer_name_sep(vars_mapping)
    name_parts  = name.split(sep)
    
    if len(name_parts) > 1:
        is_part0_root = _is_root(name_parts[0], root)
        if len(name_parts) > 2 or not is_part0_root: name_parts = name_parts[:-1]
        if skip_root and is_part0_root: name_parts = name_parts[1:]
    
    return sep.join(name_parts)

def print_vars(model, ** kwargs):
    """ Displays all variables of `model` (name with shape) """
    variables = get_var_mapping(model, ** kwargs)

    print("# variables : {}".format(len(variables)))
    for name, var in variables.items():
        print('Name : {}\t- Shape : {}'.format(name, tuple(var.shape)))
    print('\n\n')

def transpose_weights(weights):
    """ Returns the transposed version of `weights` (for pt / tf convertion) """
    if len(weights.shape) <= 1:
        return weights
    elif len(weights.shape) == 2:
        return weights.T
    elif len(weights.shape) == 3:
        return np.transpose(weights, [2, 1, 0])
    elif len(weights.shape) == 4:
        return np.transpose(weights, [2, 3, 1, 0])
    elif len(weights.shape) == 5:
        return np.transpose(weights, [2, 3, 4, 1, 0])
    else:
        raise ValueError("Unknown weights shape : {}".format(weights.shape))

def get_layers(model, ** kwargs):
    """ Equivalent to `get_layers_mapping` """
    return get_layers_mapping(model, ** kwargs)

def get_var_mapping(model):
    """ Returns a dict {var_name : var} """
    if hasattr(model, 'state_dict'):    model = model.state_dict()
    if hasattr(model, 'variables'):     model = model.variables
    return {v.name : v for v in model} if isinstance(model, list) else model

def get_layers_mapping(model, sep = None, convert_to = None, ** kwargs):
    """
        Returns a dict {layer_name : list_of_vars}
        A layer is identified by removing the last part from its name
        Layer's name parts are identified by splitting the name by `sep` ('/' inf tensorflow and '.' in pytorch's models)
        
        Arguments :
            - model : a valid type for `get_var_mapping` (dict, list, tf.keras.Model, torch.nn.Module)
            - sep   : either '/' (tensorflow) or '.' (pytorch) (determined based on the 1st variable's name if not provided)
            - kwargs    : forwarded to `_get_layer_name` to determine the layer's key in the mapping
    """
    assert convert_to in (None, 'tf', 'tensorflow', 'pt', 'pytorch')
    
    layers = model
    if not isinstance(model, dict) or not isinstance(list(model.values())[0], list):
        variables   = get_var_mapping(model)
        if not sep: sep = _get_layer_name_sep(variables)
        
        layers  = collections.OrderedDict()
        for name, var in variables.items():
            key = _get_layer_name(name, variables, sep = sep, model = model, ** kwargs)

            if hasattr(var, 'cpu'): var = var.cpu()
            layers.setdefault(key, []).append(var.numpy())
    elif not sep:
        sep = _get_layer_name_sep(layers)
    
    if convert_to in ('tf', 'tensorflow') and sep == '.':
        layers = {
            k.replace('.', '/') : pt_convert_layer_weights(w, name = k)
            for k, w in layers.items()
        }
    elif convert_to in ('pt', 'pytorch') and sep == '/':
        layers = {
            k.replace('/', '.') : tf_convert_layer_weights(w, name = k)
            for k, w in layers.items()
        }
        
    return layers

def find_layers_mapping(tf_model,
                        pt_model,
                        patterns    = {},
                        transforms  = {},
                        skip_layers = None,
                        
                        partial     = False,
                        
                        skip_root   = True,
                        convert_to  = 'tf',
                        
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
    from utils.distance.distance_method import edit_distance
    
    shape_fn  = sorted if not partial else len
    
    tf_layers = get_layers(tf_model).copy()
    pt_layers = get_layers(pt_model, convert_to = convert_to, skip_root = skip_root).copy()
    
    if patterns:
        for pat, repl in patterns.items():
            pt_layers = {re.sub(pat, repl, k) : v for k, v in pt_layers.items()}
    
    if transforms:
        for pat, trans in transforms.items():
            for k in list(pt_layers.keys()):
                if re.search(pat, k):
                    logger.debug('Applying transform {} on {} (shapes : {})'.format(
                        pat, k, [tuple(vi.shape) for vi in pt_layers[k]]
                    ))
                    pt_layers.update(trans(k, pt_layers.pop(k)))
    
    not_mapped = set(pt_layers.keys())
    pt_shapes  = {
        k : [shape_fn(np.squeeze(vi).shape) for vi in v if len(vi.shape)]
        for k, v in pt_layers.items()
    }

    mapping = {}
    for l1, l1_weights in tqdm(tf_layers.items()):
        shape = [shape_fn(np.squeeze(vi).shape) for vi in l1_weights if len(vi.shape)]
        
        bests, score = [], float('inf')
        if not skip_layers or all(re.search(s, l1) is None for s in skip_layers):
            for l2 in not_mapped:
                if shape != pt_shapes[l2]: continue

                s = edit_distance(l1, l2, normalize = False, default_replace_cost = default_replace_cost)
                if s == score:
                    bests.append(l2)
                    score = s
                elif s < score:
                    bests = [l2]
                    score = s
        
        mapping[l1] = bests
        if len(bests) == 1:
            not_mapped.remove(bests[0])
    
    return mapping, pt_layers

""" Pytorch to Tensorflow convertion """

def get_pt_layers(pt_model, ** kwargs):
    return get_layers_mapping(pt_model, sep = '.', ** kwargs)

def pt_convert_layer_weights(layer_weights, name):
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
        raise ValueError("Unknown weights length for variable {} : {}\n  Shapes : {}".format(
            name, len(layer_weights), [tuple(v.shape) for v in layer_weights]
        ))
    
    return [transpose_weights(w) for w in new_weights]

def get_pt_variables(pt_model, verbose = False):
    pt_layers = get_pt_layers(pt_model) if not isinstance(pt_model, dict) else pt_model
    converted_weights = []
    for layer_name, layer_variables in pt_layers.items():
        converted_variables = pt_convert_layer_weights(layer_variables, layer_name) if 'embedding' not in layer_name else layer_variables
        converted_weights += converted_variables
        
        logger.log(logging.INFO if verbose else logging.DEBUG, "Layer : {} \t {} \t {}".format(
            layer_name, 
            [tuple(v.shape) for v in layer_variables],
            [tuple(v.shape) for v in converted_variables],
        ))
    return converted_weights

def pt_convert_model_weights(pt_model, tf_model, verbose = False):
    converted_weights = get_pt_variables(pt_model, verbose = verbose)
    
    partial_transfer_learning(tf_model, converted_weights, verbose = verbose)
    logger.info("Weights converted successfully !")
    
    
""" Tensorflow to Pytorch converter """

def get_tf_layers(tf_model, ** kwargs):
    return get_layers_mapping(tf_model, sep = '/', ** kwargs)

def tf_convert_layer_weights(layer_weights, name = None):
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
        
        logger.log(logging.INFO if verbose else logging.DEBUG, "Layer : {} \t {} \t {}".format(
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
    logger.info("Weights converted successfully !")

""" Partial transfer learning """

def name_based_partial_transfer_learning(target_model,
                                         pretrained_model,
                                         partial_transfer      = True,
                                         partial_initializer   = 'zeros',
                                         tqdm   = lambda x: x,
                                         verbose    = False,
                                         ** kwargs
                                        ):
    """
        Make transfer learning on model with either : 
            - different number of layers (and same shapes for some layers)
            - different shapes (and same number of layers)
            
        Arguments : 
            - target_model  : tf.keras.Model instance (model where weights will be transfered to)
            - pretrained_model  : pretrained model to transfer weights from
            - partial_transfer  : whether to perform partial transfer for layers with different shapes
            - partial_initializer   : how to initialize weights when shapes differ
            - kwargs    : forwarded to `find_layers_mapping`
        
        Note : see `help(find_layers_mapping)` for more information about mappings' creation
    """
    from utils.generic_utils import get_enum_item
    
    def partial_weight_transfer(target, pretrained_v):
        if target.shape == pretrained_v.shape:
            return pretrained_v
        elif target.shape == np.squeeze(pretrained_v).shape:
            return np.squeeze(pretrained_v), {'transform' : 'squeeze'}
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
                minval = np.min(pretrained_v), maxval = np.abs(pretrained_v), size = target.shape
            )
        
        max_idx = [min(v.shape[i],pretrained_v.shape[i]) for i in range(v.ndim)]
        if v.ndim == 1:
            v[: max_idx[0]] = pretrained_v[: max_idx[0]]
        elif v.ndim == 2:
            v[: max_idx[0], : max_idx[1]] = pretrained_v[: max_idx[0], : max_idx[1]]
        elif v.ndim == 3:
            v[: max_idx[0], : max_idx[1], : max_idx[2]] = pretrained_v[
                : max_idx[0], : max_idx[1], : max_idx[2]
            ]
        elif v.ndim == 4:
            v[: max_idx[0], : max_idx[1], : max_idx[2], : max_idx[3]] = pretrained_v[
                : max_idx[0], : max_idx[1], : max_idx[2], : max_idx[3]
            ]
        elif v.ndim == 5:
            v[: max_idx[0], : max_idx[1], : max_idx[2], : max_idx[3], : max_idx[4]] = pretrained_v[
                : max_idx[0], : max_idx[1], : max_idx[2], : max_idx[3], : max_idx[4]
            ]
        else:
            raise ValueError("Unhandled variable dimension : {}".format(target.shape))
        
        return v, {'transform' : 'partial'}
    
    
    partial_initializer = get_enum_item(partial_initializer, PartialInitializer)
    
    mapping, pretrained_layers = find_layers_mapping(
        target_model, pretrained_model, partial = True, tqdm = tqdm, ** kwargs
    )
    layer_var_idx = {k : 0 for k in pretrained_layers.keys()}
    
    target_variables = target_model.variables
    all_var_names    = [v.name for v in target_variables]
    
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
    
    mapping_infos   = {}
    
    new_weights = []
    for i, v in enumerate(target_variables):
        var_layer   = _get_layer_name(v.name, all_var_names)

        mapping_infos[v.name] = {'layer' : var_layer, 'shape' : tuple(v.shape)}

        map_layer   = mapping.get(var_layer, [])
        if len(map_layer) == 0:
            logger.info('Variable {} from layer {} does not have any mapping : re-using its current weights'.format(v.name, var_layer))
            new_weights.append(v.numpy())
            continue
        
        map_layer   = map_layer[0]
        
        map_weight  = pretrained_layers[map_layer][layer_var_idx[map_layer]]
        layer_var_idx[map_layer] += 1
        
        mapping_infos[v.name].update({
            'Map layer' : map_layer, 'Map shape' : tuple(map_weight.shape)
        })
        
        new_weight  = partial_weight_transfer(v, map_weight)
        if isinstance(new_weight, tuple):
            new_weight, info = new_weight
            mapping_infos[v.name].update(info)
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
