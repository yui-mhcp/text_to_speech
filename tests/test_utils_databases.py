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
import shutil
import unittest

from functools import partial
from absl.testing import parameterized

from . import CustomTestCase, temp_dir
from utils.databases import *

class TestDatabase(CustomTestCase, parameterized.TestCase):
    path    = None
    db_class    = None
    
    def clear(self):
        pass
    
    def testUp(self):
        self.clear()
    
    def testDown(self):
        self.clear()

    def create_db(self, key = 'id', ** kwargs):
        if self.db_class is None:
            self.skipTest('This is an abstract test class. Set the `db_class` variable')
        
        self.clear()
        try:
            return self.db_class(self.path, key, reload = True, ** kwargs)
        except AssertionError as e:
            if isinstance(key, tuple):
                self.skipTest('The `multi_key` test is not supported for this database')
            else:
                raise e
    
    @parameterized.parameters(
        {'key' : 'id'}, {'key' : 'file'}, {'key' : ('filename', 'text')}
    )
    def test_initial_state(self, key):
        """ Test the initial state of a newly created database. """
        with self.create_db(key) as db:
            self.assertEqual(len(db), 0)
            self.assertEqual(db.primary_key, key)
            self.assertEqual(db.path, self.path)
    
    def test_insert(self):
        """ Test adding entries to the database. """
        with self.create_db() as db:
            data = {'id' : 'item1', 'name' : 'First Item', 'value' : 100}

            db.insert(data)

            # Verify entry was added
            self.assertEqual(len(db), 1)
            self.assertTrue('item1' in db)
            self.assertTrue(data in db)

            # Add another entry directly with object
            data2 = {'id': 'item2', 'name': 'Second Item', 'value': 200}
            db[data2['id']] = data2

            # Verify second entry was added
            self.assertEqual(len(db), 2)
            self.assertTrue('item2' in db)
            self.assertTrue('item1' in db)
    
    def test_multi_key_insert(self):
        """ Test adding entries to the database. """
        with self.create_db(('id', 'name')) as db:
            data = {'id' : 'item1', 'name' : 'First Item', 'value' : 100}

            db.insert(data)

            # Verify entry was added
            self.assertEqual(len(db), 1)
            self.assertTrue(data in db)

            # Add another entry directly with object
            data2 = {'id': 'item1', 'name': 'Second Item', 'value': 200}
            db[data2] = data2

            # Verify second entry was added
            self.assertEqual(len(db), 2)
            self.assertTrue((data2['id'], data2['name']) in db)
            self.assertTrue(data in db)

    def test_multi_insert(self):
        """ Test adding multiple entries at once to the database. """
        with self.create_db() as db:
            data = {'id' : 'item1', 'name' : 'First Item', 'value' : 100}
            data2 = {'id': 'item2', 'name': 'Second Item', 'value': 200}

            db.extend([data, data2])
            self.assertEqual(len(db), 2)
            self.assertTrue('item2' in db)
            self.assertTrue('item1' in db)

            db.extend([
                {'id': 'item3', 'name': 'Third Item', 'value': 300},
                {'id': 'item4', 'name': 'Fourth Item', 'value': 400},
                {'id': 'item5', 'name': 'Fifth Item', 'value': 500},
            ])
            self.assertEqual(len(db), 5)
            self.assertTrue('item1' in db)
            self.assertTrue('item2' in db)
            self.assertTrue('item3' in db)
            self.assertTrue('item4' in db)
            self.assertTrue('item5' in db)
    
    def test_multi_key_multi_insert(self):
        """ Test adding multiple entries at once to the database. """
        with self.create_db(('id', 'name')) as db:
            data = {'id' : 'item1', 'name' : 'First Item', 'value' : 100}
            data2 = {'id': 'item2', 'name': 'Second Item', 'value': 200}

            db.extend([data, data2])
            self.assertEqual(len(db), 2)
            self.assertTrue(data in db)
            self.assertTrue(data2 in db)

            db.extend([
                {'id': 'item', 'name': 'Third Item', 'value': 300},
                {'id': 'item', 'name': 'Fourth Item', 'value': 400},
                {'id': 'item', 'name': 'Fifth Item', 'value': 500},
            ])
            self.assertEqual(len(db), 5)

    def test_get(self):
        """ Test retrieving entries from the database. """
        with self.create_db() as db:
            data = {'id': 'item1', 'name': 'Test Item', 'value': 42}
            db[data['id']] = data

            self.assertEqual(db['item1'], data)
            self.assertEqual(db[data], data)
    
    def test_multi_key_get(self):
        with self.create_db(('id', 'name')) as db:
            data1 = {'id': 'item', 'name': 'Test Item 1', 'value': 42}
            data2 = {'id': 'item', 'name': 'Test Item 2', 'value': 42}
            db.extend([data1, data2])

            self.assertEqual(db[data1], data1)
            self.assertEqual(db[{'id' : data2['id'], 'name' : data2['name']}], data2)
            self.assertEqual(db[(data1['id'], data1['name'])], data1)
            
            self.assertEqual(db[[data2, data1]], [data2, data1], 'The `multi_get` test failed')
    
    def test_update(self):
        """ Test updating existing entries. """
        with self.create_db() as db:
            # Add initial data
            data = {'id': 'item1', 'name': 'Original Name', 'value': 100}
            db[data['id']] = data

            # Update the entry
            updated_data = {'id': 'item1', 'name': 'Updated Name', 'value': 200}
            db.update(updated_data)

            # Verify update
            self.assertEqual(len(db), 1)  # Length should remain the same
            self.assertEqual(db['item1'], updated_data)
            self.assertNotEqual(db['item1'], data)
    
    def test_multi_key_update(self):
        with self.create_db(('id', 'name')) as db:
            data1 = {'id': 'item', 'name': 'Test Item 1', 'value': 42}
            data2 = {'id': 'item', 'name': 'Test Item 2', 'value': 42}
            db.extend([data1, data2])

            updated_data = {** data1, 'value' : 10}
            db.update(updated_data)

            # Verify update
            self.assertEqual(len(db), 2)  # Length should remain the same
            self.assertEqual(db[data1], updated_data)
            self.assertEqual(db[data2], data2)
            self.assertNotEqual(db[data1], data1)

    def test_delete(self):
        """ Test removing entries from the database. """
        with self.create_db() as db:
            # Add test data
            data1 = {'id': 'item1', 'name': 'First Item', 'value': 100}
            data2 = {'id': 'item2', 'name': 'Second Item', 'value': 200}

            db[data1['id']] = data1
            db[data2['id']] = data2
            self.assertEqual(len(db), 2)

            # Delete using __delitem__
            del db['item1']
            self.assertEqual(len(db), 1)
            self.assertFalse('item1' in db)
            self.assertTrue('item2' in db)

            # Delete using pop
            popped = db.pop(data2)
            self.assertEqual(popped, data2)
            self.assertEqual(len(db), 0)
            self.assertFalse('item2' in db)
    
    def test_multi_key_delete(self):
        """ Test removing entries from the database. """
        with self.create_db(('id', 'name')) as db:
            # Add test data
            data1 = {'id': 'item', 'name': 'First Item', 'value': 100}
            data2 = {'id': 'item', 'name': 'Second Item', 'value': 200}
            data3 = {'id': 'item', 'name': 'Third Item', 'value': 200}
            
            db.extend([data1, data2, data3])

            del db[(data2['id'], data2['name'])]
            self.assertEqual(len(db), 2)
            self.assertTrue(data1 in db)
            self.assertFalse(data2 in db)
            self.assertTrue(data3 in db)

            # Delete using pop
            popped = db.multi_pop([data1, (data3['id'], data3['name'])])
            self.assertEqual(popped, [data1, data3])
            self.assertEqual(len(db), 0)

    def test_save_load(self):
        """ Test saving and loading the database. """
        try:
            with self.create_db() as db:
                # Add some data
                data1 = {'id': 'item1', 'name': 'First Item', 'value': 100}
                data2 = {'id': 'item2', 'name': 'Second Item', 'value': 200}

                db[data1['id']] = data1
                db[data2['id']] = data2

                # Save database
                db.save()
            
            self.assertTrue(os.path.exists(self.path), 'No file has been created')

            # Load database in a new instance
            with db.__class__(self.path, reload = True) as db2:
                self.assertEqual(len(db2), 2)
                self.assertTrue('item1' in db2)
                self.assertTrue('item2' in db2)
                self.assertEqual(db2['item1'], data1)
                self.assertEqual(db2['item2'], data2)
        finally:
            self.clear()
        
    def test_error_handling(self):
        """Test error handling for non-existent entries."""
        with self.create_db() as db:
            # Try to access a non-existent entry
            with self.assertRaises(KeyError):
                _ = db['non_existent']

            # Try to pop a non-existent entry
            with self.assertRaises(KeyError):
                db.pop('non_existent')

            with self.assertRaises(KeyError):
                del db['non_existent']

            with self.assertRaises(KeyError):
                db.update({'id' : 'non_existent', 'value' : 2})

            data = {'id' : 1, 'value' : 1}
            db[data] = data

            self.assertTrue(1 in db)
            # the data is already present and cannot be re-inserted
            with self.assertRaises(ValueError):
                db.insert(data)
        
    @parameterized.parameters(
        {'id': 'test1', 'data': 123},
        {'id': 'test2', 'data': "string value"},
        {'id': 'test3', 'data': [1, 2, 3]},
        {'id': 'test4', 'data': {'nested': 'dict'}}
    )
    def test_various_data_types(self, id, data):
        """Test handling different data types."""
        with self.create_db() as db:

            entry = {'id': id, 'data': data}
            db[entry] = entry
            self.assertEqual(db[id]['data'], data)
    
    def test_order(self):
        with self.create_db() as db:
            if not isinstance(db, OrderedDatabaseWrapper):
                self.skipTest('This test is only relevant for `OrderedDatabase`')

            data = [{'id' : i, 'data' : i} for i in range(5)]
            db.extend(data)

            for i in reversed(range(5)):
                self.assertEqual(i, db.index(i))
        

class TestJSONFileDatabase(TestDatabase):
    db_class    = JSONFile
    path        = os.path.join(temp_dir, 'test_db.json')
    
    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            os.remove(self.path.replace('.json', '-config.json'))

class TestJSONDatabase(TestDatabase):
    db_class    = JSONDatabase
    path        = os.path.join(temp_dir, 'test_db.json')
    
    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            os.remove(self.path.replace('.json', '-config.json'))

class TestJSONDirDatabase(TestDatabase):
    db_class    = JSONDir
    path        = os.path.join(temp_dir, 'test_db_json_dir')
    
    def clear(self):
        if os.path.exists(self.path): shutil.rmtree(self.path)

class TestSQLiteDatabase(TestDatabase):
    db_class    = SQLiteDatabase
    path        = os.path.join(temp_dir, 'test_sqlite.db')
    
    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        if os.path.exists(self.path.replace('.db', '-config.json')):
            os.remove(self.path.replace('.db', '-config.json'))

class TestOrderedDatabase(TestJSONFileDatabase):
    db_class    = partial(OrderedDatabaseWrapper, database = JSONFile)
