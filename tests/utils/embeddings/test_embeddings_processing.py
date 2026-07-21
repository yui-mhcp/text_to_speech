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

""" Pure-``pytest`` tests for :mod:`utils.embeddings.embeddings_processing`.

Plain module-level ``test_*`` functions + ``@pytest.mark.parametrize`` (no
``unittest.TestCase`` / ``absl.parameterized``). Tolerant comparisons go through
the ``asserts`` fixture (:mod:`tests.asserts`).

Backend handling mirrors :mod:`tests.utils.test_distances` :
  - ``numpy``  : pure ``np.ndarray`` inputs. ``utils.embeddings`` never imports
    ``keras`` for these (``utils.keras.ops`` dispatches ``np.ndarray`` to numpy),
    so the case is left unmarked.
  - ``tensor`` : both inputs converted to ``keras`` tensors -> marked ``keras``.

Graph / XLA compatibility is checked through ``asserts.assert_graph_compatible``
and gated behind the ``tensorflow`` marker (auto-skipped when TF is missing).
"""

import random
import numpy as np
import pytest

from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.datasets import make_blobs

from utils.embeddings import compute_centroids, get_embeddings_with_ids, select_embedding

MAX_ERR = 1e-5
N_IDS   = 7
TARGET_IDS = [0, 2]


# --- backend helpers ---------------------------------------------------------------

def _to_tensor(x, dtype):
    """ Lazily import keras so the ``numpy`` cases stay keras-free. """
    import keras.ops as K
    return K.convert_to_tensor(x, dtype = dtype)

_BACKENDS = [
    pytest.param('numpy',  id = 'numpy'),
    pytest.param('tensor', id = 'tensor', marks = pytest.mark.keras),
]

@pytest.fixture(params = _BACKENDS)
def backend(request):
    return request.param

def _prepare(backend, embeddings, ids):
    """ Convert the (numpy) reference arrays to the inputs for a given backend. """
    if backend == 'numpy':  return embeddings, ids
    return _to_tensor(embeddings, 'float32'), _to_tensor(ids, 'int32')

def _prepare_ids(backend, ids_list):
    arr = np.array(ids_list, dtype = 'int32')
    return arr if backend == 'numpy' else _to_tensor(arr, 'int32')

def _assert_backend_type(backend, x, asserts):
    """ ``utils.keras.ops`` central property : np in -> np out, tensor in -> tensor out. """
    if backend == 'numpy':  asserts.assert_array(x)
    else:                   asserts.assert_tensor(x)


# --- fixtures ----------------------------------------------------------------------

@pytest.fixture(scope = 'module')
def blobs():
    """ Deterministic (seeded) embeddings / ids drawn from ``make_blobs`` (N_IDS clusters). """
    embeddings, ids = sklearn_shuffle(* make_blobs(
        n_samples = 250, n_features = 64, centers = N_IDS, cluster_std = 1., random_state = 10
    ), random_state = 10)
    return embeddings.astype(np.float32), ids.astype(np.int32)

@pytest.fixture(scope = 'module')
def matrix():
    """ Deterministic 2-D embeddings matrix for the ``select_embedding`` tests. """
    return np.random.default_rng(0).normal(size = (10, 16)).astype(np.float32)


# --- compute_centroids -------------------------------------------------------------

@pytest.mark.parametrize(
    'num_ids', [N_IDS, None], ids = ['explicit_num_ids', 'inferred_num_ids']
)
def test_compute_centroids(blobs, backend, asserts, num_ids):
    embeddings, ids = blobs
    true_centroids  = np.stack([
        embeddings[ids == i].mean(axis = 0) for i in range(N_IDS)
    ])

    emb, idx = _prepare(backend, embeddings, ids)
    centroid_ids, centroids = compute_centroids(emb, idx, num_ids, run_eagerly = True)

    asserts.assert_equal(np.arange(N_IDS), centroid_ids)
    asserts.assert_equal(true_centroids, centroids, max_err = MAX_ERR)
    _assert_backend_type(backend, centroids, asserts)

@pytest.mark.tensorflow
def test_compute_centroids_graph_compatible(blobs, asserts):
    embeddings, ids = blobs
    true_centroids  = np.stack([
        embeddings[ids == i].mean(axis = 0) for i in range(N_IDS)
    ])
    asserts.assert_graph_compatible(
        compute_centroids, embeddings, ids, N_IDS,
        target = (np.arange(N_IDS, dtype = 'int32'), true_centroids)
    )


# --- get_embeddings_with_ids -------------------------------------------------------

def test_get_embeddings_with_ids(blobs, backend, asserts):
    embeddings, ids = blobs
    mask = np.isin(ids, TARGET_IDS)
    true_emb, true_ids = embeddings[mask], ids[mask]

    emb, idx = _prepare(backend, embeddings, ids)
    selected, selected_ids = get_embeddings_with_ids(
        emb, idx, _prepare_ids(backend, TARGET_IDS), run_eagerly = True
    )

    asserts.assert_equal(true_emb, selected, max_err = MAX_ERR)
    asserts.assert_equal(true_ids, selected_ids)
    _assert_backend_type(backend, selected, asserts)
    _assert_backend_type(backend, selected_ids, asserts)

@pytest.mark.tensorflow
def test_get_embeddings_with_ids_graph_compatible(blobs, asserts):
    embeddings, ids = blobs
    mask = np.isin(ids, TARGET_IDS)
    asserts.assert_graph_compatible(
        get_embeddings_with_ids, embeddings, ids, np.array(TARGET_IDS, dtype = 'int32'),
        target = (embeddings[mask], ids[mask])
    )


# --- select_embedding --------------------------------------------------------------

def test_select_embedding_int(matrix, asserts):
    asserts.assert_equal(matrix[3], select_embedding(matrix, mode = 3))

@pytest.mark.parametrize('mode', ['mean', 'avg', 'average'])
def test_select_embedding_mean(matrix, asserts, mode):
    asserts.assert_equal(matrix.mean(axis = 0), select_embedding(matrix, mode = mode), max_err = MAX_ERR)

def test_select_embedding_callable(matrix, asserts):
    asserts.assert_equal(
        matrix.sum(axis = 0),
        select_embedding(matrix, mode = lambda e: e.sum(axis = 0)),
        max_err = MAX_ERR
    )

def test_select_embedding_random_returns_a_row(matrix):
    random.seed(0)
    out = select_embedding(matrix, mode = 'random')
    assert any(np.allclose(out, row) for row in matrix), \
        '`random` mode must return one of the input rows'

def test_select_embedding_unknown_mode_raises(matrix):
    with pytest.raises(ValueError):
        select_embedding(matrix, mode = 'not_a_mode')


# --- select_embedding : `pd.DataFrame` filtering -----------------------------------

@pytest.fixture
def labelled_df():
    pd = pytest.importorskip('pandas')
    emb = np.random.default_rng(1).normal(size = (6, 8)).astype(np.float32)
    ids = [0, 0, 0, 1, 1, 1]
    return pd.DataFrame({'id' : ids, 'embedding' : list(emb)}), emb

def test_select_embedding_dataframe_filter(labelled_df, asserts):
    df, emb = labelled_df
    # filter `id == 0` (first 3 rows) then average
    asserts.assert_equal(
        emb[:3].mean(axis = 0), select_embedding(df, mode = 'mean', id = 0), max_err = MAX_ERR
    )

def test_select_embedding_dataframe_empty_filter_falls_back(labelled_df, asserts):
    df, emb = labelled_df
    # no row matches -> the function falls back to the whole collection
    asserts.assert_equal(
        emb.mean(axis = 0), select_embedding(df, mode = 'mean', id = 999), max_err = MAX_ERR
    )
