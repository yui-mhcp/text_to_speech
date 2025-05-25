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
from ..file_utils import dump_json, load_json

class JSONDir(Database):
    """
        Stores all data in a directory of `.json` files
        Each entry has its own file identified by a unique ID
        The mapping `entry -> ID` is stored in a special file `{path}/map.json`
    """
    def __init__(self, path, primary_key, group_key = None):
        super().__init__(path, primary_key)
        
        self._id_to_entry = load_json(self.map_file, default = {})
        self._entry_to_id = {
            v if isinstance(v, str) else tuple(v) : k for k, v in self._id_to_entry.items()
        }
        
        self._cache = {}
        self._updated   = set()
    
    @property
    def map_file(self):
        return self._get_data_file('map')
    
    def _get_data_file(self, data_id):
        assert '..' not in data_id and '/' not in data_id

        return os.path.join(self.path, '{}.json'.format(data_id))
    
    def _create_id(self):
        _id = str(uuid.uuid4())
        while _id in self._id_to_entry:
            _id = str(uuid.uuid4())
        return _id
    
    def _load(self, data_id):
        if data_id not in self._cache:
            self._cache[data_id] = load_json(self._get_data_file(data_id), default = {})
        
        return self._cache[data_id]
    
    def __len__(self):
        """ Return the number of data in the database """
        return len(self._id_to_entry)
    
    def __contains__(self, key):
        """ Return whether the entry is in the database or not """
        return self._get_entry(key) in self._entry_to_id
    
    def get(self, key):
        """ Return the information stored for the given entry """
        entry   = self._get_entry(key)
        data_id = self._entry_to_id[entry]
        return self._add_entry_to_value(entry, self._load(data_id))

    def insert(self, data):
        """
            Add a new entry to the database
            Raise a `ValueError` if `data` is already in the database
        """
        entry, value = self._assert_not_contains(data)

        data_id = self._create_id()
        
        self._entry_to_id[entry]    = data_id
        self._id_to_entry[data_id]  = entry
        self._cache[data_id] = value
        
        self._updated.add(data_id)

    def update(self, data):
        """
            Update an entry from the database
            Raise a `KeyError` if the data is not in the database
        """
        key, value = self._assert_contains(data)
        data_id = self._entry_to_id[key]
        
        self._load(data_id)
        self._cache[data_id].update(value)
        
        self._updated.add(data_id)

    def pop(self, key):
        """
            Remove an entry from the database and return its value
            Raise a `KeyError` if the entry is not in the database
        """
        entry = self._get_entry(key)
        
        data_id = self._entry_to_id[entry]
        self._load(data_id)
        
        data_file = self._get_data_file(data_id)
        if os.path.exists(data_file): os.remove(data_file)
        
        self._id_to_entry.pop(data_id)
        self._entry_to_id.pop(entry)
        if data_id in self._updated: self._updated.remove(data_id)
        return self._add_entry_to_value(entry, self._cache.pop(data_id))

    def get_column(self, column):
        """ Return the values stored in `column` for each data in the database """
        if isinstance(self.primary_key, str) and column == self.primary_key:
            return list(self._data.keys())
        elif not isinstance(self.primary_key, str) and column in self.primary_key:
            idx = list(self.primary_key).index(column)
            return [entry[idx] for entry in self._id_to_entry.values()]
        else:
            return [self._load(data_id).get(column, None) for data_id in self._id_to_entry.keys()]

    def save_data(self, ** kwargs):
        """ Save the database to the given path """
        os.makedirs(self.path, exist_ok = True)

        dump_json(self.map_file, self._id_to_entry, ** kwargs)
        for data_id in self._updated:
            filename = self._get_data_file(data_id)
            if self._cache[data_id]:
                dump_json(filename, self._cache[data_id])
            elif os.path.exists(filename):
                os.remove(filename)
        
        self._updated = set()
    