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
import enum
import keras
import importlib

from ..hparams import HParams
from .custom_activations import get_activation
from .custom_rnn_dropout_cell import CustomRNNDropoutCell

for module in os.listdir(__package__.replace('.', os.path.sep)):
    if module.startswith(('.', '_')) or '_old' in module: continue
    module = importlib.import_module(__package__ + '.' + module.replace('.py', ''))
    
    globals().update({
        k : v for k, v in vars(module).items()
        if (not k.startswith('_')) and (
            (isinstance(v, type) and issubclass(v, (keras.layers.Layer, enum.Enum)))
            or isinstance(v, HParams)
        )
    })
