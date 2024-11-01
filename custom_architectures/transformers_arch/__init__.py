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

import os
import glob
import keras
import inspect

from utils import load_json, dump_json, partial, import_objects

_transformers = sorted(import_objects(
    __package__.replace('.', os.path.sep), classes = keras.Model, allow_functions = False
).items(), key = lambda p: len(p[0]), reverse = True)

_hf_path    = os.path.expanduser('~/.cache/huggingface/hub')

_huggingface_mapping    = {
    'BertForMaskedLM'   : 'BertMLM',
}

_huggingface_config_mapping = {
    'embedding_dim' : ('hidden_size', 'd_model'),
    'num_layers'    : ('num_hidden_layers', 'layers'),
    'epsilon'       : ('layer_norm_eps', ),
    'drop_rate'     : ('hidden_dropout_prob', ),
    
    'mha_num_heads' : ('num_attention_heads', 'attention_heads'),
    'mha_num_kv_heads'  : ('num_key_value_heads', ),
    'mha_drop_rate' : ('hidden_dropout_prob', ),
    
    'ffn_dim'       : ('intermediate_size', 'ffn_dim'),
    'ffn_activation'    : ('hidden_act', 'activation_function'),
    
    'sos_token'     : ('bos_token_id', ),
    'eos_token'     : ('eos_token_id', ),
    'pad_token'     : ('pad_token_id', ),

    'max_token_types'   : ('type_vocab_size', ),
    'max_input_length'  : ('max_position_embeddings', ),
    
    'transform_activation'  : ('hidden_act', )
}

def download_hf_model(model_name, reload = False, ** kwargs):
    path    = os.environ.get('HF_HUB_CACHE', _hf_path)
    path    = os.path.join(path, 'models--{}'.format(model_name.replace('/', '--')), 'snapshots')
    if not os.path.exists(path) or not glob.glob(path + '/**/config.json') or reload:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            model_name, allow_patterns = ('*.json', '*.bin', '*.pt'), ** kwargs
        )
    return glob.glob(path + '/*')[0]

def load_hf_checkpoint(model_name, map_location = 'cpu', ** kwargs):
    import torch
    
    kwargs['map_location'] = map_location
    kwargs = {k : v for k, v in kwargs.items() if k in inspect.signature(torch.load).parameters}
    
    path = download_hf_model(model_name)
    
    weights = torch.load(glob.glob(os.path.join(path, '*model.bin'))[0], ** kwargs)
    
    for f in glob.glob(os.path.join(path, '*.pt')):
        basename    = os.path.basename(f)[:-3]
        additional_weights  = torch.load(f, ** kwargs)
        weights.update({
            '{}.{}'.format(basename, k) : v for k, v in additional_weights.items()
        })
    return weights

def load_hf_config(model_name):
    path = download_hf_model(model_name)
    
    return load_json(os.path.join(path, 'config.json'))

def get_hf_class(model_name):
    arch = load_hf_config(model_name)['architectures'][0]
    return _huggingface_mapping.get(arch, arch)

def convert_hf_config(config, hparams, prefix = None):
    converted_config = {}
    for k, v in config.items():
        if prefix and k.startswith(prefix): k = k.replace(prefix + '_', '')

        if k in hparams:
            converted_config[k] = v
            continue

        for normalized, candidates in _huggingface_config_mapping.items():
            if k in candidates:
                converted_config[normalized] = v
                break

    return hparams(** converted_config)

def get_hf_equivalent_class(model_name = None, model_class = None, ** _):
    assert model_name or model_class
    if model_class is None: model_class = get_hf_class(model_name)
    model_class = model_class.lower()
    
    for name, obj in _transformers:
        if name.lower() in model_class:
            return obj
    raise ValueError('No matching class found for {}\n  Candidates : {}'.format(
        model_class, _transformers
    ))

def get_pretrained_transformer(pretrained, ** kwargs):
    return get_hf_equivalent_class(pretrained, ** kwargs).from_pretrained(pretrained, ** kwargs)
