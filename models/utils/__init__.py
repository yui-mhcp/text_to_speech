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

from . import describe, prediction, saving

from .describe import *
from .prediction import *
from .saving import *
from .trt_llm_runner import TRTLLMRunner

def _get_tracked_type(value, types):
    if isinstance(value, (list, tuple)) and len(value) > 0: value = value[0]
    for t in types:
        if isinstance(value, t): return t
    return None