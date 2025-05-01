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

from functools import cache

from .builder import Ops
from . import core, image, linalg, math, nn, numpy, random

from .core import _creation_functions, _aliases as core_aliases
from .numpy import _aliases as numpy_aliases

from .core import *
from .image import *
from .linalg import *
from .math import *
from .nn import *
from .numpy import *
from .random import *
from .execution_contexts import *

@cache
def __getattr__(name):
    if name in _aliases: name = _aliases[name]
    if name not in globals(): return Ops(name, disable_np = name in _creation_functions)
    return globals()[name]

_aliases    = {
    ** core_aliases,
    ** numpy_aliases
}