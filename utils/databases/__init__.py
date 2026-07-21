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
from .database import Database

_databases = {}

for module in import_submodules(__package__):
    _databases.update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, Database)
    })

globals().update(_databases)

def init_database(_database = None, /, path = None, ** kwargs):
    assert _database is not None or path
    
    if isinstance(_database, Database):
        return _database
    
    if isinstance(_database, dict):
        assert 'class_name' in _database, 'Invalid database (missing `class_name`) : {}'.format(_database)
        
        cls = _database.pop('class_name')
        _database, path, kwargs = cls, _database.pop('path', path), {** kwargs, ** _database}
    
    if isinstance(_database, str):
        if _database not in _databases:
            raise ValueError('The database class {} does not exist !\n  Accepted : {}'.format(
                _database, tuple(_databases.keys())
            ))
        _database = _databases[_database]
    
    assert issubclass(_database, Database), 'Invalid database : {}'.format(_database)
    # `path is None` creates an in-memory database (not saved, not cached)
    assert path is None or isinstance(path, str), '`path` should be a string, got {}'.format(path)
    
    return _database(path, ** kwargs)