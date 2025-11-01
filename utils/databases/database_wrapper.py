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

class DatabaseWrapper(Database):
    def __init__(self, path, primary_key, *, database, ** kwargs):
        super().__init__(path, primary_key)
        
        if not isinstance(database, Database):
            from . import init_database
            database = init_database(
                database, path = path, primary_key = primary_key, _nested = True, ** kwargs
            )
        
        self._db    = database
    
    def __len__(self):
        """ Return the number of entries in the database """
        return len(self._db)
    
    def __contains__(self, key):
        """ Return whether the entry is in the database or not """
        return key in self._db
    
    def get(self, key, ** kwargs):
        """ Return the information stored for the given entry """
        return self._db.get(key, ** kwargs)

    def insert(self, data, ** kwargs):
        """
            Add a new entry to the database
            Raise a `ValueError` if `data` is already in the database
        """
        return self._db.insert(data, ** kwargs)

    def update(self, data, ** kwargs):
        """
            Update an entry from the database
            Raise a `KeyError` if the data is not in the database
        """
        return self._db.update(data, ** kwargs)
    
    def pop(self, key):
        """
            Remove an entry from the database and return its value
            Raise a `KeyError` if the entry is not in the database
        """
        return self._db.pop(key)

    def get_column(self, column):
        """ Return the values stored in `column` for each data in the database """
        return self._db.get_column(column)
    
    def save_data(self, ** kwargs):
        """ Save the database to `self.path` """
        return self._db.save_data(** kwargs)
    
    def __enter__(self):
        self._db.__enter__()
        return self
    
    def __setitem__(self, key, value):
        """
            Add a new entry (`data`) with the given `value`, or update its current value
            
        Example usage :
        ````python
        db = JSONFile('test.json', 'filename')
        
        data = {'filename' : 'test1.jpg', 'label' : 'cat'}
        
        db['test1.jpg'] = data  # equivalent to `db[data] = data`, insert the data
        db['test1.jpg', 'label'] = 'dog'    # update the entry
        ```
        
        **IMPORTANT** if `data` is already in the database, it will update its value (`self.update`), and will not trigger the `insert` method
        """
        self._db[key] = value
    
    def __getitem__(self, key):
        return self._db[key]
        
    def __delitem__(self, key):
        """ Remove an entry from the database """
        del self._db[key]
    
    def insert_or_update(self, data, ** kwargs):
        return self._db.insert_or_update(data, ** kwargs)
    
    def multi_get(self, iterable, /, ** kwargs):
        return self._db.multi_get(iterable, ** kwargs)

    def multi_insert(self, iterable, /, ** kwargs):
        return self._db.multi_insert(iterable, ** kwargs)
    
    def multi_update(self, iterable, /, ** kwargs):
        return self._db.multi_update(iterable, ** kwargs)
    
    def multi_pop(self, iterable, /, ** kwargs):
        return self._db.multi_pop(iterable, ** kwargs)
    
    def close(self):
        return self._db.close()
    
    def get_config(self):
        return {
            ** super().get_config(),
            'database'  : self._db.get_config()
        }
    