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

from utils.generic_utils import import_submodules
from .vector_index import VectorIndex

_indexes = {}
for module in import_submodules(__package__):
    _indexes.update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, VectorIndex)
    })
globals().update(_indexes)

_indexes = {k.lower() : v for k, v in _indexes.items()}
_indexes.update({k[:-5] : v for k, v in _indexes.items()})

def init_index(_index = 'numpy', /, filename = None, ** kwargs):
    if isinstance(_index, VectorIndex):
        return _index
    elif isinstance(_index, dict):
        # a `dict` carries configuration : an optional `index` (backend name / class,
        # defaulting to `numpy`), an optional `filename`, and extra `kwargs`. The
        # caller's `dict` is left untouched (we work on a copy).
        config = dict(_index)
        if 'filename' in config: filename = config.pop('filename')
        _index = config.pop('index', 'numpy')
        kwargs.update(config)
    
    if filename and 'embeddings' in kwargs:
        raise ValueError('You should specify either `filename` either `embeddings` but not both')
    
    if isinstance(_index, str):
        _index = _index.lower()
        if _index not in _indexes:
            raise ValueError('The vectors index `{}` does not exist !\n  Accepted : {}'.format(
                _index, tuple(_indexes.keys())
            ))
        _index = _indexes[_index]
    
    assert issubclass(_index, VectorIndex), 'Invalid vector index : {}'.format(_index)
    
    if filename:
        return _index.load(filename, ** kwargs)
    else:
        return _index(** kwargs)