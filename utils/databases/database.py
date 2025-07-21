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
import logging

from abc import ABCMeta, abstractmethod

from ..file_utils import dump_json, load_json

logger = logging.getLogger(__name__)

class DatabaseLoader(ABCMeta):
    _instances = {}
    
    def __call__(cls, path, * args, reload = False, _nested = False, ** kwargs):
        if _nested:
            return super().__call__(path, * args, ** kwargs)
        
        if reload or path not in cls._instances:
            config = Database.load_config(path)
            if config:
                cls_name = config.pop('class_name')
                if cls_name != cls.__name__:
                    raise ValueError("Database stored at {} does not match the class !\n  Expected : {}\n  Got : {}".format(path, cls_name, cls.__name__))

                config.pop('path')
                
                args = ()
                kwargs.update(config)
            
            try:
                cls._instances[path] = super().__call__(path, * args, ** kwargs)
            except Exception as e:
                logger.critical('An error occured while loading database {} : {}'.format(path, e))
                raise e

        return cls._instances[path]

class Database(metaclass = DatabaseLoader):
    """
        This class is an abstraction for database system. All sub-classes have to define the different methods allowing to add, remove, save, load and explore the database.
        All methods (e.g., `__contains__`, `__getitem__`, ...) accept a `data` argument that can either be the primary key entry (i.e., the data identifier), either a `dict` containing the primary key(s).
        
        Example usage :
        ````python
        db = JSONFile('test.json', 'filename')
        
        data = {'filename' : 'test1.jpg', 'label' : 'cat'}
        
        db['test1.jpg'] = data
        # db[data] = data # equivalent to the previous line
        print('test1.jpg' in db)    # True
        print(data in db)           # True
        print(db['test1.jpg'] == data)    # True

        print('test2.jpg' in db)    # False
        ```
    """
    def __init__(self, path, primary_key):
        self.path   = path
        self.primary_key    = primary_key
        
        self._is_single_key = isinstance(primary_key, str)
    
    def _get_entry(self, data):
        """ Return the `data` primary key(s) used to identify the data in the database """
        if self._is_single_key:
            if isinstance(data, dict):
                return str(data[self.primary_key])
            elif isinstance(data, str):
                return data
            elif isinstance(data, int):
                return str(data)
            else:
                raise ValueError('Data of type {} is unsupported for key {}'.format(
                    type(data), self.primary_key
                ))
        elif isinstance(data, dict):
            return tuple(str(data[k]) for k in self.primary_key)
        elif isinstance(data, tuple):
            if len(data) == len(self.primary_key):
                return tuple(str(d) for d in data)
            else:
                raise ValueError('Expected {} values for entry but {} are given'.format(
                    len(self.primary_key), len(data)
                ))
        else:
            raise ValueError('Data of type {} is unsupported : {}'.format(type(data), data))
    
    def _prepare_data(self, data):
        """ Return a tuple `(entry, data_wo_primary_keys)` """
        data  = data.copy()
        if self._is_single_key:
            entry = str(data.pop(self.primary_key))
        else:
            entry = tuple(str(data.pop(k)) for k in self.primary_key)
        
        return entry, data

    def _assert_contains(self, data):
        """ Check if `data in self`, otherwise raise an `KeyError` """
        key, value = self._prepare_data(data)
        if key not in self:
            raise KeyError('The entry `{}` is not in the database'.format(key))
        else:
            return key, value

    def _assert_not_contains(self, data):
        """ Check if `data not in self`, otherwise raise an `ValueError` """
        key, value = self._prepare_data(data)
        if key in self:
            raise ValueError('The entry `{}` is already in the database'.format(key))
        else:
            return key, value

    def _add_entry_to_value(self, entry, value):
        value = value.copy()
        if self._is_single_key:
            value[self.primary_key] = entry
        else:
            value.update({k : v for k, v in zip(self.primary_key, entry)})
        return value

    
    @property
    def is_single_key(self):
        return self._is_single_key

    @property
    def config_file(self):
        if os.path.isdir(self.path):
            return os.path.join(self.path, 'config.json')
        else:
            return os.path.splitext(self.path)[0] + '-config.json'

    @abstractmethod
    def __len__(self):
        """ Return the number of entries in the database """
    
    @abstractmethod
    def __contains__(self, key):
        """ Return whether the entry is in the database or not """
    
    @abstractmethod
    def get(self, key):
        """ Return the information stored for the given entry """

    @abstractmethod
    def insert(self, data):
        """
            Add a new entry to the database
            Raise a `ValueError` if `data` is already in the database
        """

    @abstractmethod
    def update(self, data):
        """
            Update an entry from the database
            Raise a `KeyError` if the data is not in the database
        """
    
    @abstractmethod
    def pop(self, key):
        """
            Remove an entry from the database and return its value
            Raise a `KeyError` if the entry is not in the database
        """

    @abstractmethod
    def get_column(self, column):
        """ Return the values stored in `column` for each data in the database """

    @abstractmethod
    def save_data(self, ** kwargs):
        """ Save the database to `self.path` """

    def __enter__(self):
        return self
    
    def __exit__(self, * args):
        self.close()
    
    def __repr__(self):
        return '<{} path={} key={} length={}>'.format(
            self.__class__.__name__, self.path, self.primary_key, len(self)
        )
    
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
        if (
            (isinstance(key, tuple) and len(key) == 2)
            and isinstance(key[1], str)
            and (isinstance(self.primary_key, str) or isinstance(data[0], (tuple, dict)))):
            key, column = key
            value = {column : value}
        
        entry = self._get_entry(key)
        if self._is_single_key:
            if self.primary_key not in value:
                value = value.copy()
                value[self.primary_key] = entry
        elif any(k not in value for k in self.primary_key):
            value = value.copy()
            value.update({k : e for k, e in zip(self.primary_key, entry)})
        
        self.insert_or_update(value)
    
    def __getitem__(self, key):
        if isinstance(key, list):
            return self.multi_get(key)
        else:
            return self.get(key)
        
    def __delitem__(self, key):
        """ Remove an entry from the database """
        if isinstance(key, list): self.multi_pop(key)
        else: self.pop(key)
    
    def insert_or_update(self, data):
        try:
            self.insert(data)
        except ValueError:
            self.update(data)
    
    def multi_get(self, iterable, /):
        return [self.get(data) for data in iterable]

    def multi_insert(self, iterable, /):
        for data in iterable: self.insert(data)
    
    def multi_update(self, iterable, /):
        for data in iterable: self.update(data)
    
    def multi_pop(self, iterable, /):
        return [self.pop(data) for data in iterable]
    

    def extend(self, iterable, /, ** kwargs):
        return self.multi_insert(iterable, ** kwargs)
    
    def close(self):
        pass

    def get_config(self):
        return {
            'class_name'    : self.__class__.__name__,
            'path'          : self.path,
            'primary_key'   : self.primary_key
        }
    
    def save_config(self):
        dump_json(self.config_file, self.get_config(), indent = 4)
    
    def save(self, ** kwargs):
        self.save_data(** kwargs)
        self.save_config(** kwargs)
    
    @staticmethod
    def load_config(path):
        """
            Load the config from a database path :
                - `{path}-config.json` if `path` is a file
                - `{path}/config.json` if `path` is a directory
        """
        if os.path.isdir(path):
            config_file = os.path.join(path, 'config.json')
        else:
            config_file = os.path.splitext(path)[0] + '-config.json'
        
        return load_json(config_file, default = {})
