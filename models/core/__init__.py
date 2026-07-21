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

from .base_model import BaseModel   # eager (pulls the always-on core mixins)
from . import mixins as _mixins

def __getattr__(name):
    """
        Re-exports the mixins lazily so leaf models can `from ..core import TextModelMixin` while the
        underlying modality package is still only imported on first access (see `mixins.__getattr__`).
    """
    if name in _mixins.__all__:
        return getattr(_mixins, name)
    raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))

__all__ = ['BaseModel', * _mixins.__all__]
