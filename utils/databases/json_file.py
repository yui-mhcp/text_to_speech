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

from .database import Database
from ..file_utils import dump_json, load_json

class JSONFile(Database):
    """ Stores all the data in a single `.json` file. The `primary_key` has to be a string. """
    def __init__(self, path, primary_key):
        assert isinstance(primary_key, str), 'JSONFile only supports single primary key. Use `JSONDir` instead'
        super().__init__(path, primary_key)
        
        self._data  = load_json(path, default = {})
    
    def __len__(self):
        """ Return the number of data in the database """
        return len(self._data)
    
    def __contains__(self, key):
        """ Return whether the entry is in the database or not """
        return self._get_entry(key) in self._data
    
    def get(self, key):
        """ Return the information stored for the given entry """
        entry = self._get_entry(key)
        return self._add_entry_to_value(entry, self._data[entry])

    def insert(self, data):
        """
            Add a new entry to the database
            Raise a `ValueError` if `data` is already in the database
        """
        key, value = self._assert_not_contains(data)
        self._data[key] = value

    def update(self, data):
        """
            Update an entry from the database
            Raise a `KeyError` if the data is not in the database
        """
        key, value = self._assert_contains(data)
        self._data[key].update(value)

    def pop(self, key):
        """
            Remove an entry from the database and return its value
            Raise a `KeyError` if the entry is not in the database
        """
        entry = self._get_entry(key)
        return self._add_entry_to_value(entry, self._data.pop(entry))

    def insert_or_update(self, data):
        key, value = self._prepare_data(data)
        self._data.setdefault(key, {}).update(value)

    def get_column(self, column):
        """ Return the values stored in `column` for each data in the database """
        if column == self.primary_key:
            return list(self._data.keys())
        else:
            return [value.get(column, None) for value in self._data.values()]

    def save_data(self, ** kwargs):
        """ Save the database to `self.path` """
        dump_json(self.path, self._data, ** kwargs)
    
