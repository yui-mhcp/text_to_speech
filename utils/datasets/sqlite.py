# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd

class SQLiteDataset:
    def __init__(self, path, default_table = 'documents'):
        self.__path = path
        self.__default_table = default_table
        
        self.__connection = None
        self.__cursor = None
        self.__length = None
    
    @property
    def path(self):
        return self.__path
    
    @property
    def connection(self):
        if self.__connection is None:
            import sqlite3
            self.__connection = sqlite3.connect(self.path)
        return self.__connection
    
    @property
    def cursor(self):
        if self.__cursor is None:
            import sqlite3
            self.__cursor = self.connection.cursor()
            self.__cursor.row_factory = sqlite3.Row
        return self.__cursor
    
    @property
    def length(self):
        if self.__length is None:
            self.__length = self.select('COUNT(id)').fetchone()[0]
        return self.__length
    
    @property
    def columns(self):
        row = self.select('*').fetchone()
        return row.keys()
        
    @property
    def ids(self):
        return self.select('id')
    
    @property
    def dataframe(self):
        return pd.DataFrame([
            {k : row[k] for k in row.keys()} for row in self.select('*')
        ])
    
    def __iter__(self):
        return self.select('*')
    
    def __str__(self):
        return "SQLite database for path {}".format(self.path)
    
    def __len__(self):
        return self.length
    
    def __contains__(self, item):
        return self[item] != []
    
    def __getitem__(self, item):
        return self.get(item)
    
    def get(self, id = None, table = None, ** kwargs):
        if table is None: table = self.__default_table
        if id is not None: kwargs['id'] = id
        keys, values = [], []
        for k, v in kwargs.items():
            keys.append('{} = ?'.format(k))
            values.append(v)

        cond = ' WHERE {}'.format(' AND '.join(keys)) if len(keys) > 0 else ''

        rows = self.select('*', table = table, cond = cond, values = values).fetchall()
        return rows[0] if len(rows) == 1 else rows
    
    def select(self, column, table = None, cond = '', values = {}):
        if table is None: table = self.__default_table
        return self.cursor.execute('SELECT {} FROM {}{}'.format(column, table, cond), values)
    
def preprocess_sqlite_database(directory, filename, as_dataframe = True):
    db = SQLiteDataset(os.path.join(directory, filename))
    return db.dataframe if as_dataframe else db
