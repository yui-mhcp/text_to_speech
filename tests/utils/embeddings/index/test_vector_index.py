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

""" Pure-``pytest`` tests for the :class:`VectorIndex` implementations.

The original ``test_index.py`` used an abstract ``TestVectorIndex`` base shared by
three ``unittest`` subclasses (numpy / keras / torch). That hierarchy is now a
single parametrized ``index_cls`` fixture :
  - ``NumpyIndex``  : keras-free -> unmarked.
  - ``KerasIndex``  : builds keras tensors -> marked ``keras``.
  - ``TorchIndex``  : builds torch tensors -> marked ``torch``.

All raw-value comparisons use ``metric = 'euclidian'`` so the stored embeddings
are kept as-is (``cosine`` would L2-normalize them on ``add``).
"""

import os
import pytest
import numpy as np

from utils.keras import ops

# --- backend (index class) fixture -------------------------------------------------

_INDEXES = [
    pytest.param('NumpyIndex', id = 'numpy'),
    pytest.param('KerasIndex', id = 'keras', marks = pytest.mark.keras),
    pytest.param('TorchIndex', id = 'torch', marks = pytest.mark.torch),
]

@pytest.fixture(params = _INDEXES)
def index_cls(request):
    from utils.embeddings import index as index_pkg
    return getattr(index_pkg, request.param)

def _make(index_cls, metric = 'euclidian', ** kwargs):
    return index_cls(metric = metric, ** kwargs)

def _random_vectors(n_vectors = 100, embedding_dim = 128, seed = 0):
    return np.random.default_rng(seed).random((n_vectors, embedding_dim)).astype(np.float32)


# --- add / shape -------------------------------------------------------------------

def test_add(index_cls, asserts):
    dim   = 64
    index = _make(index_cls)

    emb_1 = _random_vectors(50, dim, seed = 1)
    emb_2 = _random_vectors(25, dim, seed = 2)

    index.add(emb_1)
    asserts.assert_equal(50, len(index))
    asserts.assert_equal((50, dim), index.shape)

    index.add(emb_2)
    asserts.assert_equal(75, len(index))
    asserts.assert_equal((75, dim), index.shape)

    asserts.assert_equal(np.concatenate([emb_1, emb_2], axis = 0), index.embeddings)


# --- __getitem__ -------------------------------------------------------------------

def test_get(index_cls, asserts):
    dim   = 64
    index = _make(index_cls)
    emb_1 = _random_vectors(50, dim, seed = 1)

    index.add(emb_1)

    asserts.assert_equal(emb_1,      index.embeddings)
    asserts.assert_equal(emb_1,      index[:])
    asserts.assert_equal(emb_1[0],   index[0])
    asserts.assert_equal(emb_1[:5],  index[np.arange(5)])


# --- remove ------------------------------------------------------------------------

def test_remove(index_cls, asserts):
    dim   = 64
    index = _make(index_cls)
    emb_1 = _random_vectors(50, dim, seed = 1)

    index.add(emb_1)
    asserts.assert_equal(emb_1, index.embeddings)

    index.remove(0)
    asserts.assert_equal(49, len(index))
    asserts.assert_equal(emb_1[1:], index.embeddings)

    index.remove(np.arange(45, 49))
    asserts.assert_equal(45, len(index))
    asserts.assert_equal(emb_1[1:46], index.embeddings)


# --- top_k -------------------------------------------------------------------------

def test_top_k(index_cls, asserts):
    dim   = 8
    index = _make(index_cls)

    emb = np.arange(dim * 64).reshape((64, dim)).astype('float32')
    emb_1, emb_2 = emb[:48], emb[48:]

    index.add(emb_1)
    asserts.assert_equal(48, len(index))

    indices, scores = index.top_k(emb_2[0], k = 5)
    asserts.assert_equal((1, 5), tuple(scores.shape))
    asserts.assert_equal((1, 5), tuple(indices.shape))
    asserts.assert_equal([47, 46, 45, 44, 43], indices[0])

    indices, scores = index.top_k(emb_2[:2], k = 3)
    asserts.assert_equal((2, 3), tuple(scores.shape))
    asserts.assert_equal((2, 3), tuple(indices.shape))
    asserts.assert_equal([[47, 46, 45], [47, 46, 45]], indices)

    index.add(emb_2)
    indices, scores = index.top_k(emb_2[0], k = 5)
    asserts.assert_equal(48, indices[0, 0])
    asserts.assert_equal(0,  scores[0, 0])

def test_top_k_clamps_k(index_cls, asserts):
    dim   = 8
    index = _make(index_cls)
    index.add(np.arange(dim * 4).reshape((4, dim)).astype('float32'))

    indices, scores = index.top_k(_random_vectors(1, dim, seed = 1), k = 10)
    asserts.assert_equal((1, 4), tuple(ops.convert_to_numpy(indices).shape))
    asserts.assert_equal((1, 4), tuple(ops.convert_to_numpy(scores).shape))

def test_top_k_masked(index_cls, asserts):
    dim   = 8
    index = _make(index_cls)
    index.add(np.arange(dim * 10).reshape((10, dim)).astype('float32'))

    query = np.full((dim, ), dim * 10., dtype = 'float32')

    # indices must be **global** (relative to the whole index, not to the masked subset)
    indices, scores = index.top_k(query, k = 3, mask = [0, 2, 4, 6])
    asserts.assert_equal((1, 3), tuple(ops.convert_to_numpy(indices).shape))
    asserts.assert_equal([6, 4, 2], ops.convert_to_numpy(indices)[0])

    # boolean masks are supported as well
    bool_mask = np.arange(10) % 2 == 0
    indices, scores = index.top_k(query, k = 3, mask = bool_mask)
    asserts.assert_equal([8, 6, 4], ops.convert_to_numpy(indices)[0])

    # `k` larger than the masked subset is clamped
    indices, scores = index.top_k(query, k = 10, mask = [1, 3])
    asserts.assert_equal([3, 1], ops.convert_to_numpy(indices)[0])


# --- error handling ----------------------------------------------------------------

def test_error_handling(index_cls):
    index = _make(index_cls)
    index.add(_random_vectors(64, 128, seed = 1))

    with pytest.raises(Exception):
        _ = index[64]

    with pytest.raises(IndexError):
        index.remove(64)

    with pytest.raises(ValueError):
        index.add(_random_vectors(64, 64, seed = 2))


# --- save / load -------------------------------------------------------------------

@pytest.fixture
def index_path(temp_dir):
    path = os.path.join(temp_dir, 'test_vector_index.index')
    yield path
    for p in (path, path + '.npy'):
        if os.path.exists(p): os.remove(p)

def test_io(index_cls, index_path, asserts):
    dim   = 64
    index = _make(index_cls)
    emb_1 = _random_vectors(50, dim, seed = 1)

    index.add(emb_1)
    asserts.assert_equal(50, len(index))

    index.save(index_path)
    assert os.path.exists(index_path) or os.path.exists(index_path + '.npy')

    loaded = index_cls.load(index_path)
    asserts.assert_equal(50, len(loaded))
    asserts.assert_equal((50, dim), loaded.shape)
    asserts.assert_equal(emb_1, loaded.embeddings)
