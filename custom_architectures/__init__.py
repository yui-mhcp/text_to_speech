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
import json
import keras
import inspect

from utils import import_objects, get_object, print_objects, partial, dispatch_wrapper
from .simple_models import classifier, perceptron, simple_cnn

_architectures = {
    'perceptron'    : perceptron,
    'simple_cnn'    : simple_cnn,
    ** import_objects(
        __package__.replace('.', os.path.sep),
        filters = lambda name, val: name[0].isupper() and 'current_blocks' not in val.__module__,
        classes = keras.Model
    ),
    ** {
        k : partial(classifier, v)
        for k, v in import_objects(keras.applications, types = type).items()
    }
}
globals().update(_architectures)

@dispatch_wrapper(_architectures, 'architecture')
def get_architecture(architecture, * args, ** kwargs):
    return get_object(
        _architectures, architecture, * args, print_name = 'architecture', ** kwargs
    )

def get_custom_objects():
    import custom_layers
    
    return {
        'Sequential'    : keras.Sequential,
        'Functional'    : keras.Model,
        ** _architectures,
        ** import_objects(custom_layers, classes = keras.layers.Layer),
        ** import_objects(keras.layers, classes = keras.layers.Layer)
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

def print_architectures():
    print_objects(_architectures, 'model architectures')

