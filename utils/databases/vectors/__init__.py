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

def init_index(_index = None, /, path = None, ** kwargs):
    assert _index is not None or path
    
    if isinstance(_index, VectorIndex):
        return _index
    
    if not isinstance(_index, dict) and path and os.path.exists(path + '-config.json'):
        _index = VectorIndex.load_config(path)
    
    if isinstance(_index, dict):
        assert 'class_name' in _index, 'Invalid index (missing `class_name`) : {}'.format(_index)
        
        if path and 'vectors' not in _index: _index['vectors'] = path
        
        cls = _index.pop('class_name')
        _index, kwargs = cls, {** kwargs, ** _index}
    
    if isinstance(_index, str):
        if _index not in _indexes:
            raise ValueError('The database class {} does not exist !\n  Accepted : {}'.format(
                _index, tuple(_indexes.keys())
            ))
        _index = _indexes[_index]
    
    assert issubclass(_index, VectorIndex), 'Invalid database : {}'.format(_index)
    
    return _index(** kwargs)
