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
import json
import keras
import inspect
import importlib

from .hparams import HParams
from .current_blocks import _keras_layers, set_cudnn_lstm
from .simple_models import classifier, perceptron, simple_cnn

for module in os.listdir(__package__.replace('.', os.path.sep)):
    if module.startswith(('.', '_')) or '_old' in module: continue
    elif '_arch' not in module and module != 'transformers': continue
    module = importlib.import_module(__package__ + '.' + module.replace('.py', ''))
    
    globals().update({
        k : v for k, v in vars(module).items()
        if (k[0].isupper()) and (
            inspect.isfunction(v) or (isinstance(v, type) and issubclass(v, (keras.Model, keras.layers.Layer)))
        )
    })

_architectures = {
    k : v for k, v in globals().items()
    if (k[0].isupper()) and (
        inspect.isfunction(v) or (isinstance(v, type) and issubclass(v, keras.Model))
    )
}
_architectures.update({
    'classifier' : classifier, 'perceptron' : perceptron, 'simple_cnn' : simple_cnn
})

_custom_layers = {
    k : v for k, v in globals().items()
    if (isinstance(v, type) and issubclass(v, keras.layers.Layer))
}

_architectures_lower = {k.lower() : v for k, v in _architectures.items()}

def get_architecture(architecture, * args, ** kwargs):
    return _architectures_lower[architecture.lower()](* args, ** kwargs)

def get_custom_objects():
    return {
        'Sequential'    : keras.Sequential,
        'Functional'    : keras.Model,
        ** _architectures,
        ** _custom_layers,
        ** _keras_layers
    }

def deserialize_keras2_model(config, safe_mode = True, replace_lambda_by_l2 = True):
    _objects = get_custom_objects()
    if config['class_name'] == 'Sequential':
        def _update_keras2_config(config):
            if isinstance(config, list):
                return [_update_keras2_config(it) for it in config]
            elif not isinstance(config, dict): return config

            if 'class_name' in config and 'config' in config:
                if config['class_name'] == 'Lambda':
                    if replace_lambda_by_l2:
                        return {
                            'class_name'    : 'CustomActivation',
                            'config'    : {'activation' : 'l2_norm'}
                        }
                    elif not safe_mode:
                        keras.config.enable_unsafe_deserialization()
                    

                config = config.copy()
                config['config'] = config['config'].copy()
                if 'batch_input_shape' in config['config']:
                    config['config']['batch_shape'] = config['config'].pop('batch_input_shape')

                if config['class_name'] in _objects:
                    params = inspect.signature(_objects[config['class_name']]).parameters
                    if 'args' not in params:
                        config['config'] = {
                            k : v for k, v in config['config'].items() if k in params
                        }

                for k, v in config['config'].items():
                    if 'initializer' in k and isinstance(v, dict) and 'class_name' in v:
                        try:
                            keras.initializers.get(v)
                        except:
                            config['config'][k] = {'class_name' : v['class_name'].lower(), 'config' : v['config']}
                config['config'] = _update_keras2_config(config['config'])
                return config
            else:
                return {k : _update_keras2_config(v) for k, v in config.items()}

        config['config'] = _update_keras2_config(config['config'])

    json_config = json.dumps(config)
    with keras.utils.CustomObjectScope(_objects):
        return keras.models.model_from_json(json_config)

