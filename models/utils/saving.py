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

from utils import load_json
from custom_train_objects.history import History

_pretrained_models_dir = 'pretrained_models'

def get_saving_dir():
    global _pretrained_models_dir
    return _pretrained_models_dir

def get_pretrained_weights_dir():
    return os.path.join(get_saving_dir(), 'pretrained_weights')

def get_model_dir(name, * args):
    return os.path.join(get_saving_dir(), name, * args)

def is_model_name(name):
    """ Check if the model `name` has a directory with `config.json` file """
    return os.path.exists(get_model_dir(name, 'config.json'))

def get_model_infos(name):
    if name is None: return {}
    if not isinstance(name, str):
        return {
            'class_name' : name.__class__.__name__,
            'config'     : name.get_config()
        }
    return load_json(get_model_dir(name, 'config.json'), default = {})

def get_model_class(name):
    """ Return the (str) class of model named `name` """
    return get_model_infos(name).get('class_name', None)

def get_model_config(name):
    return get_model_infos(name).get('config', {})

def get_model_history(name):
    """ Return the `History` class for model `name` """
    return History.load(get_model_dir(name, 'saving', 'history.json'))

def remove_training_checkpoint(name):
    """ Remove checkpoints in `{model}/training-logs/checkpoints/*` """
    training_ckpt_dir = get_model_dir(name, 'training-logs', 'checkpoints')
    for file in os.listdir(training_ckpt_dir):
        os.remove(os.path.join(training_ckpt_dir, file))
