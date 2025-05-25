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
import uuid

from .database import Database
from ..generic_utils import to_json
from ..file_utils import dump_json, load_json

class JSONDatabase(Database):
    """ Stores all the data in a single `.json` file. The `primary_key` has to be a string. """
    def __init__(self, path, primary_key):
        super().__init__(path, primary_key)
        
        self._is_single_key = isinstance(self.primary_key, str)
        
        data  = load_json(path, default = {})
        if data and ('data' not in data or 'entries' not in data):
            data = {'data' : data, 'entries' : {}}
        
        self._data  = data.pop('data', {})
        self._id_to_entry   = data.pop('entries', {})
        
        self._entry_to_id = {
            v if isinstance(v, str) else tuple(v) : k for k, v in self._id_to_entry.items()
        }
        self._updated = False
    
    def _create_id(self):
        _id = str(uuid.uuid4())
        while _id in self._id_to_entry:
            _id = str(uuid.uuid4())
        return _id

    def __len__(self):
        """ Return the number of data in the database """
        return len(self._data)
    
    def __contains__(self, key):
        """ Return whether the entry is in the database or not """
        entry   = self._get_entry(key)
        data_id = self._entry_to_id.get(entry, entry)
        return data_id in self._data
    
    def get(self, key):
        """ Return the information stored for the given entry """
        entry   = self._get_entry(key)
        data_id = self._entry_to_id.get(entry, entry)
        return self._add_entry_to_value(entry, self._data[data_id])

    def insert(self, data):
        """
            Add a new entry to the database
            Raise a `ValueError` if `data` is already in the database
        """
        key, value = self._assert_not_contains(data)
        
        value = to_json(value)
        if self._is_single_key:
            self._data[key] = value
        else:
            data_id = self._create_id()
            
            self._data[data_id] = value
            self._id_to_entry[data_id]  = list(key)
            self._entry_to_id[key]      = data_id
        
        self._updated = True

    def update(self, data):
        """
            Update an entry from the database
            Raise a `KeyError` if the data is not in the database
        """
        entry, value = self._assert_contains(data)
        
        data_id = self._entry_to_id.get(entry, entry)
        
        value       = to_json(value)
        original    = self._data[data_id]
        # the second check is only used to avoid unnecessary re-saving
        # therefore, if `_updated` is already True, it is not relevant anymore
        if self._updated or any(k not in original or v != original[k] for k, v in value.items()):
            self._data[data_id].update(value)
            self._updated = True

    def pop(self, key):
        """
            Remove an entry from the database and return its value
            Raise a `KeyError` if the entry is not in the database
        """
        entry = self._get_entry(key)
        
        if self._is_single_key:
            item = self._data.pop(entry)
        else:
            data_id = self._entry_to_id.get(entry, entry)
            
            item = self._data.pop(data_id)
            self._id_to_entry.pop(data_id)
            self._entry_to_id.pop(entry)
        
        self._updated = True

        return self._add_entry_to_value(entry, item)

    def get_column(self, column):
        """ Return the values stored in `column` for each data in the database """
        if self._is_single_key and column == self.primary_key:
            return list(self._data.keys())
        elif not self._is_single_key and column in self.primary_key:
            idx = list(self.primary_key).index(column)
            return [entry[idx] for entry in self._id_to_entry.values()]
        else:
            return [value.get(column, None) for value in self._data.values()]
    
    def save_data(self, ** kwargs):
        """ Save the database to `self.path` """
        if self._updated:
            dump_json(self.path, {'data' : self._data, 'entries' : self._id_to_entry}, safe = True)
            
            self._updated = False
