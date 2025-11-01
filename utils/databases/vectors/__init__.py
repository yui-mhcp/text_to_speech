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
import importlib

from .vector_index import VectorIndex

_indexes = {}

for module in os.listdir(__package__.replace('.', os.path.sep)):
    if module.startswith(('.', '_')) or '_old' in module: continue
    module = importlib.import_module(__package__ + '.' + module.replace('.py', ''))
    
    _indexes.update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, VectorIndex)
    })

globals().update(_indexes)

_indexes = {k.lower() : v for k, v in _indexes.items()}

def init_index(_index = None, /, path = None, ** kwargs):
    assert _index is not None or path
    
    if isinstance(_index, VectorIndex):
        return _index
    
    if path and os.path.exists(path + '-config.json'):
        config = VectorIndex.load_config(path)
        cls = config.pop('class_name', 'numpy')
        if not _index: _index = cls
        kwargs.update(config)
    
    kwargs['path'] = path
    if isinstance(_index, str):
        _index = _index.lower()
        if not _index.endswith('index'):
            _index = _index + 'index'
        
        if _index not in _indexes:
            raise ValueError('The vectors index class {} does not exist !\n  Accepted : {}'.format(
                _index, tuple(_indexes.keys())
            ))
        _index = _indexes[_index]
    
    assert issubclass(_index, VectorIndex), 'Invalid database : {}'.format(_index)
    
    return _index(** kwargs)
