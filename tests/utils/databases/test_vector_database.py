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

""" Tests for ``utils.databases.vector_database.VectorDatabase``.

The database couples an ordered key-value store (``JSONDatabase``) with a ``VectorIndex``
(1 vector per entry, in insertion order). Tests run on the ``NumpyIndex`` backend, which is
keras-free -> unmarked.
"""

import os
import shutil
import numpy as np
import pytest

from utils.databases import VectorDatabase


@pytest.fixture
def db_path(temp_dir):
    path = os.path.join(temp_dir, 'test_vector_db')
    if os.path.exists(path): shutil.rmtree(path)
    yield path
    if os.path.exists(path): shutil.rmtree(path)

def _make(path = None, ** kwargs):
    kwargs.setdefault('index', 'numpy')
    kwargs.setdefault('metric', 'euclidian')
    return VectorDatabase(path, 'hash', ** kwargs)

def _entries(n, source = 'file 1'):
    return [
        {'hash' : 'chunk-{}'.format(i), 'text' : 'text {}'.format(i), 'source' : source}
        for i in range(n)
    ]

def _linear_vectors(values, dim = 4):
    """ Row ``i`` equals ``values[i] * ones(dim)`` -> nearest-neighbor order is unambiguous """
    return np.array(values, dtype = 'float32')[:, None] * np.ones((dim, ), dtype = 'float32')


# --- insertion ---------------------------------------------------------------------

def test_insert(asserts):
    db = _make()
    entries = _entries(3)

    out = db.multi_insert(entries, vectors = _linear_vectors([0., 1., 2.]))

    asserts.assert_equal(3, len(db))
    asserts.assert_equal(3, len(db.vectors))
    asserts.assert_equal(['chunk-0', 'chunk-1', 'chunk-2'], out)

    assert 'chunk-1' in db
    assert entries[1] in db
    assert 'unknown' not in db

    asserts.assert_equal('text 1', db['chunk-1']['text'])

def test_insert_with_vector_key(asserts):
    db = _make(vector_key = 'embedding')

    db.insert({'hash' : 'chunk-0', 'text' : 'a', 'embedding' : np.ones((4, ), 'float32')})

    asserts.assert_equal(1, len(db))
    asserts.assert_equal(1, len(db.vectors))
    # the vector is not stored in the key-value entry
    assert 'embedding' not in db['chunk-0']

def test_insert_duplicate_raises(asserts):
    db = _make()
    db.multi_insert(_entries(2), vectors = _linear_vectors([0., 1.]))

    with pytest.raises(ValueError):
        db.insert({'hash' : 'chunk-1', 'text' : 'dup', 'embedding' : np.ones((4, ), 'float32')})


# --- search ------------------------------------------------------------------------

def test_search(asserts):
    db = _make()
    sources = ['a', 'b', 'a', 'b', 'a', 'b']
    entries = [{'hash' : 'c-{}'.format(i), 'source' : src} for i, src in enumerate(sources)]
    db.multi_insert(entries, vectors = _linear_vectors([0., 1., 2., 3., 4., 5.]))

    query = _linear_vectors([4.9])

    results = db.search(query, k = 2)
    asserts.assert_equal(1, len(results))
    asserts.assert_equal(['c-5', 'c-4'], [res['hash'] for res in results[0]])
    assert all('score' in res for res in results[0])

    # `k` larger than the database is clamped
    results = db.search(query, k = 100)
    asserts.assert_equal(6, len(results[0]))

    # `reverse` returns the results in ascending relevance order
    results = db.search(query, k = 2, reverse = True)
    asserts.assert_equal(['c-4', 'c-5'], [res['hash'] for res in results[0]])

def test_search_multi_queries(asserts):
    db = _make()
    db.multi_insert(_entries(4), vectors = _linear_vectors([0., 1., 2., 3.]))

    results = db.search(_linear_vectors([0.1, 2.9]), k = 1)
    asserts.assert_equal(2, len(results))
    asserts.assert_equal('chunk-0', results[0][0]['hash'])
    asserts.assert_equal('chunk-3', results[1][0]['hash'])

def test_search_filtered(asserts):
    db = _make()
    sources = ['a', 'b', 'a', 'b', 'a', 'b']
    entries = [{'hash' : 'c-{}'.format(i), 'source' : src} for i, src in enumerate(sources)]
    db.multi_insert(entries, vectors = _linear_vectors([0., 1., 2., 3., 4., 5.]))

    query = _linear_vectors([4.9])

    # the filtered search must return **global** entries (not subset-local ones)
    results = db.search(query, k = 2, filters = {'source' : 'a'})
    asserts.assert_equal(['c-4', 'c-2'], [res['hash'] for res in results[0]])

    # callable filter
    results = db.search(query, k = 2, filters = {'source' : lambda s: s == 'b'})
    asserts.assert_equal(['c-5', 'c-3'], [res['hash'] for res in results[0]])

    # filter matching nothing -> 1 empty list per query
    results = db.search(query, k = 2, filters = {'source' : 'unknown'})
    asserts.assert_equal([[]], results)


# --- removal -----------------------------------------------------------------------

def test_pop(asserts):
    db = _make()
    db.multi_insert(_entries(3), vectors = _linear_vectors([0., 1., 2.]))

    item = db.pop('chunk-1')
    asserts.assert_equal('text 1', item['text'])
    asserts.assert_equal(2, len(db))
    asserts.assert_equal(2, len(db.vectors))
    assert 'chunk-1' not in db

    # remaining vectors stay aligned with the remaining entries
    results = db.search(_linear_vectors([2.1]), k = 1)
    asserts.assert_equal('chunk-2', results[0][0]['hash'])


# --- save / reload -----------------------------------------------------------------

def test_save_and_reload(db_path, asserts):
    db = _make(db_path)
    db.multi_insert(_entries(3), vectors = _linear_vectors([0., 1., 2.]))
    db.save()

    assert os.path.exists(os.path.join(db_path, 'config.json'))

    # the primary key, index backend and metric are restored from the saved config
    reloaded = VectorDatabase(db_path, reload = True)

    asserts.assert_equal(3, len(reloaded))
    asserts.assert_equal('hash', reloaded.primary_key)
    asserts.assert_equal('euclidian', reloaded.vectors.metric)
    asserts.assert_equal('NumpyIndex', type(reloaded.vectors).__name__)
    asserts.assert_equal(3, len(reloaded.vectors))
    asserts.assert_equal('text 2', reloaded['chunk-2']['text'])

    # 1.9 is nearest to the vector 2.0 -> chunk-2 (confirms vectors stay aligned on reload)
    results = reloaded.search(_linear_vectors([1.9]), k = 1)
    asserts.assert_equal('chunk-2', results[0][0]['hash'])


# --- in-memory ---------------------------------------------------------------------

def test_in_memory(asserts):
    db = _make()
    db.multi_insert(_entries(2), vectors = _linear_vectors([0., 1.]))

    with pytest.raises(ValueError):
        db.save()

    # in-memory databases are not cached : a new instance is empty
    asserts.assert_equal(0, len(_make()))
