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
import enum
import keras

from utils import HParams, import_objects
from .custom_activations import get_activation
from .custom_rnn_dropout_cell import CustomRNNDropoutCell

globals().update(import_objects(
    __package__.replace('.', os.path.sep),
    classes = (keras.layers.Layer, enum.Enum),
    types   = (type, HParams)
))
