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

""" Generic contract tests shared by every ``Database`` implementation.

Each test takes the ``make_db`` factory (parametrized over ``backend x key`` in
``conftest.py``), so a single test body runs against every backend and both primary-key
arities. ``VectorDatabase`` keeps its own suite (``test_vector_database.py``) because its
signatures differ (it requires embeddings).

Backends and key-builders live in ``_base.py``. The suite is keras-free (unmarked).
"""

import os

import pytest

from ._base import entry_id, make_entry, make_entries, primary_key_for


# --- insertion / lookup ------------------------------------------------------------

def test_insert_and_contains(make_db, key, asserts):
    db = make_db()
    entries = make_entries(key, 3)

    out = [db.insert(e) for e in entries]

    asserts.assert_equal(3, len(db))
    asserts.assert_equal([entry_id(key, i) for i in range(3)], out)

    # membership by identifier and by full `dict`
    assert entry_id(key, 1) in db
    assert entries[1] in db
    assert entry_id(key, 99) not in db

def test_get_returns_value_with_primary_key(make_db, key, asserts):
    db = make_db()
    db.insert(make_entry(key, 0))

    value = db[entry_id(key, 0)]
    asserts.assert_equal('text 0', value['text'])
    # `get` re-injects the primary key(s) in the returned value
    assert value == make_entry(key, 0)

def test_get_by_dict(make_db, key, asserts):
    db = make_db()
    entry = make_entry(key, 0)
    db.insert(entry)
    asserts.assert_equal('text 0', db[entry]['text'])

def test_insert_duplicate_raises(make_db, key):
    db = make_db()
    db.insert(make_entry(key, 0))
    with pytest.raises(ValueError):
        db.insert(make_entry(key, 0))

def test_get_missing_raises(make_db, key):
    db = make_db()
    with pytest.raises(KeyError):
        db.get(entry_id(key, 0))


# --- update ------------------------------------------------------------------------

def test_update(make_db, key, asserts):
    db = make_db()
    db.insert(make_entry(key, 0))

    db.update(make_entry(key, 0, text = 'updated', extra = 'new'))

    value = db[entry_id(key, 0)]
    asserts.assert_equal('updated', value['text'])
    asserts.assert_equal('new', value['extra'])

def test_update_missing_raises(make_db, key):
    db = make_db()
    with pytest.raises(KeyError):
        db.update(make_entry(key, 0))

def test_insert_or_update(make_db, key, asserts):
    db = make_db()

    db.insert_or_update(make_entry(key, 0))            # inserts
    asserts.assert_equal(1, len(db))
    db.insert_or_update(make_entry(key, 0, text = 'v2'))   # updates in place
    asserts.assert_equal(1, len(db))
    asserts.assert_equal('v2', db[entry_id(key, 0)]['text'])


# --- __setitem__ / __delitem__ -----------------------------------------------------

def test_setitem_insert(make_db, key, asserts):
    db = make_db()
    entry = make_entry(key, 0)

    db[entry_id(key, 0)] = entry

    assert entry_id(key, 0) in db
    asserts.assert_equal(1, len(db))
    asserts.assert_equal('text 0', db[entry_id(key, 0)]['text'])

def test_setitem_column_update(make_db, key, asserts):
    db = make_db()
    db.insert(make_entry(key, 0))

    db[entry_id(key, 0), 'text'] = 'patched'

    asserts.assert_equal(1, len(db))
    asserts.assert_equal('patched', db[entry_id(key, 0)]['text'])

def test_delitem(make_db, key, asserts):
    db = make_db()
    db.insert(make_entry(key, 0))
    db.insert(make_entry(key, 1))

    del db[entry_id(key, 0)]

    asserts.assert_equal(1, len(db))
    assert entry_id(key, 0) not in db
    assert entry_id(key, 1) in db


# --- removal -----------------------------------------------------------------------

def test_pop(make_db, key, asserts):
    db = make_db()
    db.insert(make_entry(key, 0))
    db.insert(make_entry(key, 1))

    item = db.pop(entry_id(key, 0))
    asserts.assert_equal('text 0', item['text'])
    assert item == make_entry(key, 0)   # popped value carries the primary key(s)

    asserts.assert_equal(1, len(db))
    assert entry_id(key, 0) not in db

def test_pop_missing_raises(make_db, key):
    db = make_db()
    with pytest.raises(KeyError):
        db.pop(entry_id(key, 0))


# --- batch operations --------------------------------------------------------------

def test_multi_insert_get_pop(make_db, key, asserts):
    db = make_db()
    entries = make_entries(key, 3)

    out = db.multi_insert(entries)
    asserts.assert_equal([entry_id(key, i) for i in range(3)], out)

    got = db.multi_get([entry_id(key, i) for i in range(3)])
    asserts.assert_equal(['text 0', 'text 1', 'text 2'], [v['text'] for v in got])

    popped = db.multi_pop([entry_id(key, 0), entry_id(key, 2)])
    asserts.assert_equal(['text 0', 'text 2'], [v['text'] for v in popped])
    asserts.assert_equal(1, len(db))
    assert entry_id(key, 1) in db

def test_extend(make_db, key, asserts):
    db = make_db()
    db.extend(make_entries(key, 2))
    asserts.assert_equal(2, len(db))


# --- introspection : keys / values / items / columns -------------------------------

def test_keys(make_db, key, asserts):
    db = make_db()
    db.multi_insert(make_entries(key, 3))

    expected = [entry_id(key, i) for i in range(3)]
    asserts.assert_equal(sorted(map(str, expected)), sorted(map(str, db.keys())))

def test_items_and_values_carry_primary_key(make_db, key, asserts):
    db = make_db()
    db.multi_insert(make_entries(key, 3))

    items = dict((str(k), v) for k, v in db.items())
    asserts.assert_equal(3, len(items))
    for i in range(3):
        value = items[str(entry_id(key, i))]
        # unified contract : the value returned by `items` embeds the primary key(s)
        assert value == make_entry(key, i)

    texts = sorted(v['text'] for v in db.values())
    asserts.assert_equal(['text 0', 'text 1', 'text 2'], texts)

def test_get_column(make_db, key, asserts):
    db = make_db()
    db.multi_insert(make_entries(key, 3))

    asserts.assert_equal(['text 0', 'text 1', 'text 2'], db.get_column('text'))


# --- filter ------------------------------------------------------------------------

def test_filter_equality(make_db, key, asserts):
    db = make_db()
    db.insert(make_entry(key, 0, group = 'a'))
    db.insert(make_entry(key, 1, group = 'b'))
    db.insert(make_entry(key, 2, group = 'a'))

    removed = db.filter(group = 'a')

    asserts.assert_equal({'text 0', 'text 2'}, {v['text'] for v in removed})
    asserts.assert_equal(1, len(db))
    assert entry_id(key, 1) in db

def test_filter_callable(make_db, key, asserts):
    db = make_db()
    db.multi_insert([make_entry(key, i, group = g) for i, g in enumerate('abab')])

    db.filter(group = lambda g: g == 'b')

    asserts.assert_equal(2, len(db))
    assert entry_id(key, 0) in db and entry_id(key, 2) in db


# --- positional indexing (ordered backends only) -----------------------------------

def test_positional_indexing(make_db, backend, key, asserts):
    if not backend.ordered:
        pytest.skip('backend `{}` does not support positional indexing'.format(backend.id))

    db = make_db()
    db.multi_insert(make_entries(key, 4))

    asserts.assert_equal('text 0', db[0]['text'])
    asserts.assert_equal('text 3', db[-1]['text'])
    asserts.assert_equal(['text 1', 'text 2'], [v['text'] for v in db[1:3]])

    # positional order stays consistent after a removal
    db.pop(entry_id(key, 1))
    asserts.assert_equal(['text 0', 'text 2', 'text 3'], [v['text'] for v in db[:]])


# --- persistence : save / reload ---------------------------------------------------

def test_save_and_reload(make_db, backend, key, db_path, asserts):
    db = make_db(path = db_path)
    db.multi_insert(make_entries(key, 3))
    db.save()

    assert os.path.exists(os.path.join(db_path, 'config.json'))

    # the class, primary key and content are restored purely from the saved config/data
    reloaded = backend.cls(db_path, reload = True)

    asserts.assert_equal(3, len(reloaded))
    # a composite key round-trips through JSON as a list ; `assert_equal` compares
    # `tuple` / `list` element-wise so this holds for both arities
    asserts.assert_equal(primary_key_for(key), reloaded.primary_key)
    asserts.assert_equal('text 2', reloaded[entry_id(key, 2)]['text'])
    assert entry_id(key, 0) in reloaded


# --- in-memory ---------------------------------------------------------------------

def test_in_memory_cannot_save(make_db, key):
    db = make_db()   # path is None
    db.insert(make_entry(key, 0))
    with pytest.raises(ValueError):
        db.save()

def test_in_memory_not_cached(make_db, key, asserts):
    db = make_db()
    db.multi_insert(make_entries(key, 2))
    # in-memory databases are never cached : a fresh instance starts empty
    asserts.assert_equal(0, len(make_db()))
