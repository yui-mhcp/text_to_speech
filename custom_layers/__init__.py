# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
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
import importlib

from hparams import HParams
from custom_layers.custom_activations import *

def __load():
    for module in glob.glob(os.path.join(* __package__.split('.'), '*.py')):
        if module.endswith(('__init__.py', '_old.py')): continue
        module = importlib.import_module(module.replace(os.path.sep, '.')[:-3])
        
        globals().update({
            k : v for k, v in vars(module).items()
            if not k.startswith('_') and isinstance(v, (type, HParams))
        })

__load()
