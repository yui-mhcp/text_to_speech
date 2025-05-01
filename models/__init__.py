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
import sys
import logging
import importlib

from .utils import *
from .interfaces.base_model import BaseModel

logger = logging.getLogger(__name__)

def get_pretrained(name, ** kwargs):
    if isinstance(name, dict): name, kwargs = name.pop('name'), {** kwargs, ** name}
    
    model_class = get_model_class(name)
    if model_class is None:
        print_pretrained()
        raise ValueError("Model {} does not exist or has an invalid `config.json` !".format(name))
    elif model_class not in globals():
        _import_model_classes(model_class)
        if model_class not in globals():
            raise ValueError('The model class {} is not supported anymore'.format(model_class))
    
    return globals()[model_class](name = name, ** kwargs)

def print_pretrained():
    _groups = {}
    
    for f in os.listdir(get_saving_dir()):
        class_name = get_model_class(f)
        if class_name: _groups.setdefault(class_name, []).append(f)
    
    msg = ''
    for class_name, models in _groups.items():
        msg += 'Models for class {} :\n- {}\n'.format(class_name, '\n- '.join(models))
    
    logger.info(msg)

def _import_classes(module, _globals):
    _globals.update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, BaseModel)
    })

def _import_model_classes(target_class):
    _globals = globals()
    for name, module in sys.modules.items():
        if name.startswith('models.') and name.count('.') == 1:
            _import_classes(module, _globals)
    
    if target_class and target_class in _globals: return
    
    for module in os.listdir(__package__.replace('.', os.path.sep)):
        if module.startswith(('.', '_')) or '_old' in module: continue
        module = importlib.import_module(__package__ + '.' + module.replace('.py', ''))

        _import_classes(module, _globals)
