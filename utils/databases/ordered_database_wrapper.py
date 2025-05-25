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

import numpy as np

from .database_wrapper import DatabaseWrapper

class OrderedDatabaseWrapper(DatabaseWrapper):
    def __init__(self, path, primary_key, *, database, entries = None, ** kwargs):
        super().__init__(path, primary_key, database = database)
        
        self._idx_to_entry = entries if entries else []
        if not isinstance(self.primary_key, str):
            self._idx_to_entry = [tuple(entry) for entry in self._idx_to_entry]
        self._entry_to_idx = {entry : i for i, entry in enumerate(self._idx_to_entry)}

    def __len__(self):
        """ Return the number of data in the database """
        return len(self._idx_to_entry)
    
    def __iter__(self):
        for key in self._idx_to_entry:
            yield self[key]
        
    def __contains__(self, key):
        """ Return whether the entry is in the database or not """
        return self._get_entry(key) in self._entry_to_idx
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            index = list(range(len(self)))[index]
        elif isinstance(index, np.ndarray):
            index = index.tolist()
        
        if isinstance(index, int):
            index = self._idx_to_entry[index]
        elif isinstance(index, list) and len(index) and isinstance(index[0], int):
            index = [self._idx_to_entry[idx] for idx in index]
        
        return super().__getitem__(index)
    
    def index(self, key):
        return self._entry_to_idx[self._get_entry(key)]
    
    def insert(self, data):
        """ Add a new entry to the database """
        super().insert(data)
        
        entry = self._get_entry(data)
        self._idx_to_entry.append(entry)
        self._entry_to_idx[entry] = len(self._entry_to_idx)
        
    def pop(self, key):
        """ Remove and return the given entry from the database """
        entry   = self._get_entry(key)
        item    = super().pop(entry)
        
        idx = self._entry_to_idx[entry]
        self._idx_to_entry.pop(idx)
        self._entry_to_idx = {entry : i for i, entry in enumerate(self._idx_to_entry)}
        
        return item
    
    def insert_or_update(self, data):
        try:
            self.insert(data)
        except ValueError:
            self.update(data)

    def multi_insert(self, iterable, /):
        super().multi_insert(iterable)
        
        entries = [self._get_entry(data) for data in iterable]
        self._idx_to_entry.extend(entries)
        self._entry_to_idx.update({
            entry : len(self._entry_to_idx) + i for i, entry in enumerate(entries)
        })

    def multi_pop(self, iterable, /):
        items   = super().multi_pop(iterable)
        
        entries = [self._get_entry(data) for data in iterable]
        indexes = [self._entry_to_idx[entry] for idx in entries]
        
        for idx in sorted(indexes, reverse = True):
            self._idx_to_entry.pop(idx)
        self._entry_to_idx = {entry : i for i, entry in enumerate(self._idx_to_entry)}
        
        return items

    def get_config(self):
        return {
            ** super().get_config(),
            'entries'   : self._idx_to_entry
        }
    