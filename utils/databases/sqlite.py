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
import json
import sqlite3

from loggers import Timer, timer
from .database import Database
from ..generic_utils import to_json

class SQLiteDatabase(Database):
    def __init__(self, path, primary_key):
        super().__init__(path, primary_key)
        
        # Ensure directory exists for the database
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok = True)
        
        # Initialize the database
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        
        self._create_table()
    
    def _create_table(self):
        """Create the SQLite table with appropriate primary keys"""
        cursor = self.conn.cursor()
        
        # Generate column definitions
        primary_keys = [self.primary_key] if isinstance(self.primary_key, str) else self.primary_key
        
        # Create a default schema with primary key columns
        columns = [
            '"{}" TEXT'.format(k) for k in primary_keys
        ] + ['"_data" TEXT']
        
        # Create primary key constraint
        pk_constraint = "PRIMARY KEY ({})".format(
            ', '.join('"{}"'.format(key) for key in primary_keys)
        )
        
        # Create the table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS data (
            {},
            {}
        )
        """.format(', '.join(columns), pk_constraint)
        cursor.execute(create_table_sql)
        
        # Create indexes for better performance
        for key in primary_keys:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_{} ON data ("{}")'.format(key, key))
    
    def _build_where_clause(self, entry):
        if isinstance(self.primary_key, str):
            return '"{}" = ?'.format(self.primary_key), (entry, )
        else:
            return ' AND '.join('"{}" = ?'.format(k) for k in self.primary_key), entry

    def __del__(self):
        """ Ensure connection is closed when object is deleted """
        self.close()
    
    def __len__(self):
        """ Return the number of entries in the database """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM data")
        return cursor.fetchone()[0]

    def __contains__(self, key):
        """ Return whether the entry is in the database or not """
        entry  = self._get_entry(key)
        cursor = self.conn.cursor()
        
        clause, args = self._build_where_clause(entry)
        
        cursor.execute('SELECT 1 FROM data WHERE {} LIMIT 1'.format(clause), args)
        return cursor.fetchone() is not None

    def get(self, key):
        """ Return the information stored for the given entry """
        entry  = self._get_entry(key)
        cursor = self.conn.cursor()
        
        clause, args = self._build_where_clause(entry)
        
        cursor.execute('SELECT * FROM data WHERE {}'.format(clause), args)
        
        row = cursor.fetchone()
        if row is None:
            raise KeyError("Entry {} not found in database".format(entry))
        
        row_dict = dict(row)
        row_dict.update(json.loads(row_dict.pop('_data')))
        
        return row_dict

    def insert(self, data):
        """
            Add a new entry to the database
            Raise a `ValueError` if `data` is already in the database
        """
        entry, value = self._prepare_data(data)
        
        cursor = self.conn.cursor()
        
        primary_keys = self.primary_key
        if isinstance(primary_keys, str):
            primary_keys = [primary_keys]
            entry = (entry, )

        # Prepare SQL
        placeholders = ', '.join(['?'] * (len(primary_keys) + 1))
        columns = ', '.join(['"{}"'.format(k) for k in primary_keys] + ['_data'])
            
        # Execute insert
        cursor.execute(
            "INSERT OR IGNORE INTO data ({}) VALUES ({})".format(columns, placeholders),
            entry + (json.dumps(to_json(value)), )
        )
        if cursor.rowcount == 0:
            raise ValueError('The entry {} is already in the database'.format(entry))
        

    def update(self, data):
        """
            Update an entry from the database
            Raise a `KeyError` if the data is not in the database
        """
        key, value = self._prepare_data(data)
        
        updated_data = self[key]
        updated_data.update(value)
        
        cursor = self.conn.cursor()
        
        clause, args = self._build_where_clause(key)

        cursor.execute(
            "UPDATE data SET _data = ? WHERE {}".format(clause),
            (json.dumps(updated_data), ) + args
        )
    
    def pop(self, key):
        """
            Remove an entry from the database and return its value
            Raise a `KeyError` if the entry is not in the database
        """
        data = self[key]  # This will raise KeyError if not found
        
        clause, args = self._build_where_clause(self._get_entry(key))

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM data WHERE {}".format(clause), args)
        return data
    
    def get_column(self, column):
        cursor = self.conn.cursor()
        cursor.execute('SELECT {} FROM data'.format(column))
        return [dict(row)[column] for row in cursor.fetchall()]
    
    def __multi_get(self, iterable, /):
        cursor = self.conn.cursor()
        results = {}

        if isinstance(self.primary_key, list) or isinstance(self.primary_key, tuple):
            # Clés primaires multiples
            # Construire une requête IN avec tuples
            # Note: SQLite ne supporte pas directement IN avec tuples, donc on utilise OR avec AND

            # Construire la clause WHERE dynamiquement
            where_clauses = []
            params = []

            for entry in entries:
                if not isinstance(entry, tuple) or len(entry) != len(self.primary_key):
                    raise ValueError(f"Invalid key format: {entry}. Expected a tuple with {len(self.primary_key)} elements.")

                conditions = []
                for i, pk in enumerate(self.primary_key):
                    conditions.append(f'"{pk}" = ?')
                    params.append(entry[i])

                where_clauses.append(f"({' AND '.join(conditions)})")

            # Construire et exécuter la requête SQL
            query = f"SELECT * FROM data WHERE {' OR '.join(where_clauses)}"
            cursor.execute(query, params)
        else:
            # Clé primaire simple
            # On peut utiliser l'opérateur IN standard
            placeholders = ','.join(['?'] * len(entries))
            query = f'SELECT * FROM data WHERE "{self.primary_key}" IN ({placeholders})'
            cursor.execute(query, entries)

        # Traiter les résultats
        for row in cursor.fetchall():
            row_dict = dict(row)

            # Extraire la clé primaire
            if isinstance(self.primary_key, list) or isinstance(self.primary_key, tuple):
                key = tuple(row_dict[pk] for pk in self.primary_key)
            else:
                key = row_dict[self.primary_key]

            # Extraire et parser les données JSON
            data_json = json.loads(row_dict.pop('_data', '{}'))

            # Fusionner les colonnes de clés primaires avec les données JSON
            results[key] = {**row_dict, **data_json}

        return results
    
    @timer
    def multi_insert(self, iterable, /):
        try:
            with Timer('preparation'):
                _keys = [self.primary_key] if isinstance(self.primary_key, str) else self.primary_key

                columns      = ['"{}"'.format(k) for k in _keys] + ['"_data"']
                placeholders = ', '.join(['?'] * len(columns))

                params = []
                insert_sql = "INSERT INTO data ({}) VALUES ({})".format(
                    ', '.join(columns), placeholders
                )

                for data in iterable:
                    entry, value = self._prepare_data(data)
                    if not isinstance(entry, tuple): entry = (entry, )

                    params.append(entry + (json.dumps(to_json(value)), ))

            with Timer('transaction'):
                cursor = self.conn.cursor()
                self.conn.execute("BEGIN TRANSACTION")
                cursor.executemany(insert_sql, params)
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def save_data(self, ** kwargs):
        """ Save the database """
        self.conn.commit()
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
