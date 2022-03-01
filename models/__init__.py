
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

import os
import glob

from models.model_utils import _pretrained_models_folder, infer_model_class
from utils.generic_utils import get_object, print_objects

def __load():
    for module_name in glob.glob('models/*'):
        if not os.path.isdir(module_name): continue
        module_name = module_name.replace(os.path.sep, '.')

        module = __import__(
            module_name, fromlist = ['_models']
        )
        if hasattr(module, '_models'):
            _models.update(module._models)

def get_model(model_name, *args, **kwargs):
    return get_object(
        _models, model_name, * args, print_name = 'models', err = True, ** kwargs
    )

def print_models():
    print_objects(_models, 'models')

def get_pretrained(model_name):
    model_class = infer_model_class(model_name, _models)
    if model_class is None:
        print_pretrained()
        raise ValueError("Model {} does not exist or its configuration file is corrupted !".format(model_name))
    
    return model_class(nom = model_name)

def print_pretrained():
    _str_classes = {k : k for k in _models.keys()}
    _groups = {}
    
    for f in os.listdir(_pretrained_models_folder):
        class_name = infer_model_class(f, _str_classes)
        if class_name: _groups.setdefault(class_name, []).append(f)
    
    for class_name, models in _groups.items():
        print("Models for class {} :".format(class_name))
        for m in models: print("- {}".format(m))
        print()

def update_models():
    names = [
        f for f in os.listdir(_pretrained_models_folder)
        if os.path.exists(os.path.join(_pretrained_models_folder, f, 'config.json'))
    ]
    
    for name in names:
        print("Update model '{}'".format(name))
        model = get_pretrained(name)
        model.save(save_ckpt = False)
        print(model)
        del model
    
_models = {}

__load()