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

from functools import wraps

from .database import Database

def _redirect(name):
    @wraps(getattr(Database, name))
    def inner(self, * args, ** kwargs):
        return getattr(self._db, name)(* args, ** kwargs)
    return inner

class DatabaseWrapper(Database):
    def __init__(self, path, primary_key, *, database, ** kwargs):
        super().__init__(path, primary_key)
        
        if not isinstance(database, Database):
            from . import init_database
            database = init_database(
                database, path = path, primary_key = primary_key, _nested = True, ** kwargs
            )
        
        self._db    = database
    
    def __len__(self):  return len(self._db)
    def __contains__(self, key):    return key in self._db
    
    def get(self, key):     return self._db.get(key)
    def insert(self, data): return self._db.insert(data)
    def update(self, data): return self._db.update(data)
    def pop(self, key):     return self._db.pop(key)

    def get_column(self, column):   return self._db.get_column(column)

    def save_data(self):    return self._db.save_data()
    
    multi_get     = _redirect('multi_get')
    multi_insert  = _redirect('multi_insert')
    multi_update  = _redirect('multi_update')
    multi_pop     = _redirect('multi_pop')

    insert_or_update    = _redirect('insert_or_update')
    
    close   = _redirect('close')
    
    def get_config(self):
        return {
            ** super().get_config(),
            'database'  : self._db.get_config()
        }
    