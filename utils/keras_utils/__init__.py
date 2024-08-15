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

import keras
import logging

from keras import tree

from . import ops
from .compile import *
from .gpu_utils import *

logger = logging.getLogger(__name__)

def show_version():
    logger.info('Keras version : {} - backend ({}) : {}'.format(
        keras.__version__, ops.get_backend(), ops.get_backend_module().__version__
    ))