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

""" Fixtures for the generic ``utils.databases`` test-suite.

The ``backend`` and ``key`` fixtures are parametrized, so any test taking ``make_db``
is automatically expanded over every ``backend x primary-key`` combination. The
``_isolate_db_cache`` autouse fixture neutralizes the ``DatabaseLoader`` per-path
singleton cache, which otherwise leaks instances between tests sharing a path.

Kept dependency-light (only ``numpy`` transitively) : no keras / tensorflow, so the
whole suite stays unmarked.
"""

import os
import shutil

import pytest

from utils.databases import Database
from ._base import BACKENDS, KEYS, primary_key_for


@pytest.fixture(params = BACKENDS, ids = [b.id for b in BACKENDS])
def backend(request):
    """ The database implementation under test (see ``_base.BACKENDS``). """
    return request.param

@pytest.fixture(params = KEYS)
def key(request):
    """ Primary-key arity : ``'single'`` (``str``) or ``'multi'`` (``tuple``). """
    return request.param


@pytest.fixture(autouse = True)
def _isolate_db_cache():
    """ Clear (and restore) the ``DatabaseLoader`` per-path cache around each test. """
    saved = dict(Database._instances)
    Database._instances.clear()
    yield
    Database._instances.clear()
    Database._instances.update(saved)


@pytest.fixture
def make_db(backend, key):
    """ Factory building an (empty) database of the current ``backend`` / ``key``.

        ``make_db()``            -> in-memory database (``path is None``)
        ``make_db(path = ...)``  -> on-disk database at ``path``

        ``primary_key`` defaults to the current ``key`` arity but can be overridden.
    """
    created = []

    def _make(path = None, primary_key = None, ** kwargs):
        if primary_key is None:
            primary_key = primary_key_for(key)
        db = backend.cls(path, primary_key, ** {** backend.kwargs, ** kwargs})
        created.append(db)
        return db

    yield _make

    for db in created:
        try: db.close()
        except Exception: pass


@pytest.fixture
def db_path(backend, key, temp_dir):
    """ A unique, freshly-cleaned directory path for persistence tests. """
    path = os.path.join(temp_dir, 'db_{}_{}'.format(backend.id, key))
    if os.path.exists(path): shutil.rmtree(path)
    yield path
    if os.path.exists(path): shutil.rmtree(path)
