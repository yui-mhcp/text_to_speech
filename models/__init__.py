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
import logging

from .utils import *
from .interfaces.base_model import BaseModel

from utils import import_objects

logger = logging.getLogger(__name__)

_models = import_objects(
    __package__.replace('.', os.path.sep),
    filters = lambda name, val: 'interfaces' not in val.__module__,
    classes = BaseModel,
    allow_functions = False
)
globals().update(_models)


def get_pretrained(name, ** kwargs):
    model_class = infer_model_class(name, _models)
    if model_class is None:
        print_pretrained()
        raise ValueError("Model {} does not exist or has an invalid `config.json` !".format(
            name
        ))
    
    return model_class(name = name, ** kwargs)

def print_pretrained():
    _str_classes = {k : k for k in _models.keys()}
    _groups = {}
    
    for f in os.listdir(get_saving_dir()):
        class_name = infer_model_class(f, _str_classes)
        if class_name: _groups.setdefault(class_name, []).append(f)
    
    msg = ''
    for class_name, models in _groups.items():
        msg += 'Models for class {} :\n- {}\n'.format(
            class_name, '\n- '.join([m for m in models])
        )
    logger.info(msg)
    return _groups

def update_models():
    names = [
        f for f in os.listdir(_pretrained_models_folder)
        if os.path.exists(os.path.join(_pretrained_models_folder, f, 'config.json'))
    ]
    
    for name in names:
        print("Update model '{}'".format(name))
        model = get_pretrained(name)
        model.save()
        print(model)
        del model
    