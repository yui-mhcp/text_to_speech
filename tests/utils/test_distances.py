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

""" Pure-``pytest`` tests for :mod:`utils.distances`.

Written as plain module-level functions + ``@pytest.mark.parametrize`` (no
``unittest.TestCase`` / ``absl.parameterized``). Tolerant numpy comparisons go
through the ``asserts`` fixture (``tests.asserts``).

Backend handling (mirrors the old ``TestNumpyDistance`` / ``TestTensorDistance``
/ ``TestTensorAndArrayDistance`` class hierarchy, now a parametrized fixture) :
  - ``numpy``  : pure ``np.ndarray`` inputs. ``utils.distances`` never imports
    ``keras`` (``utils.keras.ops`` dispatches ``np.ndarray`` to numpy), so these
    cases run **without keras** and are therefore left unmarked.
  - ``tensor`` : both inputs converted to ``keras`` tensors -> marked ``keras``.
  - ``mixed``  : numpy query + tensor points -> marked ``keras``. The output type
    is backend-dependent here, so only the *values* are checked (as the old
    ``TestTensorAndArrayDistance`` did).

Graph / XLA compatibility is checked through ``asserts.assert_graph_compatible``
and gated behind the ``tensorflow`` marker (auto-skipped when TF is missing).
"""

import numpy as np
import pytest

from contextlib import nullcontext

from sklearn.utils import shuffle as sklearn_shuffle

from utils.distances import distance, knn, euclidian_distance

MAX_ERR = 1e-5


# --- backend helpers ---------------------------------------------------------------

def _to_tensor(x, dtype = 'float32'):
    """ Lazily import keras so the ``numpy`` cases stay keras-free. """
    import keras.ops as K
    return K.convert_to_tensor(x, dtype = dtype)

_BACKENDS = [
    pytest.param('numpy',  id = 'numpy'),
    pytest.param('tensor', id = 'tensor', marks = pytest.mark.keras),
    pytest.param('mixed',  id = 'mixed',  marks = pytest.mark.keras),
]

@pytest.fixture(params = _BACKENDS)
def backend(request):
    return request.param

def _prepare(backend, q, p, dtype = 'float32'):
    """ Convert the (numpy) reference arrays to the inputs for a given backend. """
    q, p = q.astype(dtype), p.astype(dtype)
    if backend == 'numpy':  return q, p
    if backend == 'mixed':  return q, _to_tensor(p, dtype)
    return _to_tensor(q, dtype), _to_tensor(p, dtype)

def _device_cpu(backend):
    """ Force CPU for keras backends : `matmul` / `einsum` (matrix mode) accumulate
        differently on CPU vs GPU, which can break the tolerant comparisons against
        the numpy reference. ``numpy`` is already CPU-only -> no-op.
    """
    if backend == 'numpy': return nullcontext()
    import keras
    return keras.device('cpu')

def _assert_backend_type(backend, result, asserts):
    """ ``utils.keras.ops`` central property : np in -> np out, tensor in -> tensor out.

        For ``mixed`` inputs the output type is backend-dependent, so it is not checked.
    """
    if backend == 'numpy':      asserts.assert_array(result)
    elif backend == 'tensor':   asserts.assert_tensor(result)


# --- reference (numpy) implementations ---------------------------------------------

def _l2_normalize(x):
    return x / np.linalg.norm(x, axis = -1, keepdims = True)

# aligned (point-wise / row-wise) references : `q` is (d,) or (N, d), `p` is (N, d)
_ALIGNED = {
    'manhattan' : lambda q, p: np.sum(np.abs(q - p), axis = -1),
    'euclidian' : lambda q, p: np.linalg.norm(q - p, axis = -1),
    'dp'        : lambda q, p: np.sum(q * p, axis = -1),
    'cosine'    : lambda q, p: np.sum(_l2_normalize(q) * _l2_normalize(p), axis = -1),
}

# matrix references : `q` is (Nq, d), `p` is (Np, d) -> (Nq, Np)
_MATRIX = {
    'manhattan' : lambda q, p: np.stack([np.sum(np.abs(qi - p), axis = -1) for qi in q]),
    'euclidian' : lambda q, p: np.stack([np.linalg.norm(qi - p, axis = -1) for qi in q]),
    'dp'        : lambda q, p: q @ p.T,
    'cosine'    : lambda q, p: _l2_normalize(q) @ _l2_normalize(p).T,
}

_VECTOR_METHODS = list(_ALIGNED)


# --- fixtures ----------------------------------------------------------------------

@pytest.fixture(scope = 'module')
def vectors():
    """ Deterministic (seeded) query / point matrices, ``float32``. """
    rng = np.random.default_rng(0)
    n, d = 64, 16
    queries = rng.normal(size = (n, d)).astype(np.float32)
    points  = rng.normal(size = (n, d)).astype(np.float32)
    return queries, points


# --- `distance` : aligned (point-wise & multi) -------------------------------------

@pytest.mark.parametrize('shape', ['point', 'multi'])
@pytest.mark.parametrize('method', _VECTOR_METHODS)
def test_distance_aligned(vectors, backend, asserts, method, shape):
    q_np, p_np = vectors
    qc, pc = _prepare(backend, q_np, p_np)

    if shape == 'point':    q_in, q_ref = qc[0], q_np[0]
    else:                   q_in, q_ref = qc,    q_np

    result = distance(q_in, pc, method = method)

    asserts.assert_equal(_ALIGNED[method](q_ref, p_np), result, max_err = MAX_ERR)
    asserts.assert_equal((len(p_np), ), tuple(result.shape))
    _assert_backend_type(backend, result, asserts)


# --- `distance` : matrix -----------------------------------------------------------

@pytest.mark.parametrize('method', _VECTOR_METHODS)
def test_distance_matrix(vectors, backend, asserts, method):
    q_np, p_np = vectors
    qc, pc = _prepare(backend, q_np, p_np)

    with _device_cpu(backend):
        result = distance(qc, pc, method = method, as_matrix = True)

    asserts.assert_equal((len(q_np), len(p_np)), tuple(result.shape))
    asserts.assert_equal(_MATRIX[method](q_np, p_np), result, max_err = MAX_ERR)
    _assert_backend_type(backend, result, asserts)

def test_distance_matrix_is_rectangular(vectors, asserts):
    """ Matrix mode must accept different numbers of queries and points. """
    q_np, p_np = vectors
    half    = len(p_np) // 2
    result  = distance(q_np, p_np[:half], method = 'manhattan', as_matrix = True)
    asserts.assert_equal((len(q_np), half), tuple(result.shape))


# --- `distance` : element-wise (l1 / l2) -------------------------------------------

@pytest.mark.parametrize('method,ref', [('l1', np.abs), ('l2', np.square)])
def test_distance_elementwise(vectors, backend, asserts, method, ref):
    q_np, p_np = vectors
    qc, pc = _prepare(backend, q_np, p_np)

    result = distance(qc, pc, method = method)

    asserts.assert_equal(ref(q_np - p_np), result, max_err = MAX_ERR)
    asserts.assert_equal(q_np.shape, tuple(result.shape))
    _assert_backend_type(backend, result, asserts)


# --- `distance` : dice_coeff -------------------------------------------------------

def test_dice_coeff(vectors, backend, asserts):
    q_np, p_np = vectors
    qc, pc = _prepare(backend, q_np, p_np)

    result = distance(qc, pc, method = 'dice_coeff')

    inter   = np.sum(q_np * p_np)
    union   = np.sum(q_np) + np.sum(p_np)
    asserts.assert_equal(2. * inter / union, result, max_err = MAX_ERR)

def test_dice_coeff_matrix_not_implemented(vectors):
    q_np, p_np = vectors
    with pytest.raises(NotImplementedError):
        distance(q_np, p_np, method = 'dice_coeff', as_matrix = True)


# --- `distance` : dtype coherence (numpy, keras-free) ------------------------------

@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('method', _VECTOR_METHODS)
def test_distance_preserves_dtype(vectors, method, dtype):
    q_np, p_np = vectors
    q, p = q_np.astype(dtype), p_np.astype(dtype)

    result = distance(q, p, method = method)
    assert result.dtype == np.dtype(dtype), \
        'Expected {} output, got {}'.format(np.dtype(dtype), result.dtype)


# --- `distance` : mode forcing & errors (numpy, keras-free) ------------------------

def test_mode_forces_distance_from_similarity(vectors, asserts):
    """ A similarity (`dp`) forced to `distance` is negated. """
    q, p = vectors
    asserts.assert_equal(
        - distance(q, p, method = 'dp'),
        distance(q, p, method = 'dp', mode = 'distance'),
        max_err = MAX_ERR
    )

def test_mode_forces_similarity_from_distance(vectors, asserts):
    """ A distance (`euclidian`) forced to `similarity` is negated. """
    q, p = vectors
    asserts.assert_equal(
        - distance(q, p, method = 'euclidian'),
        distance(q, p, method = 'euclidian', mode = 'similarity'),
        max_err = MAX_ERR
    )

def test_mode_noop_when_already_matching(vectors, asserts):
    """ Forcing the mode a method already has must leave the result unchanged. """
    q, p = vectors
    asserts.assert_equal(
        distance(q, p, method = 'dp'),
        distance(q, p, method = 'dp', mode = 'similarity'),
        max_err = MAX_ERR
    )
    asserts.assert_equal(
        distance(q, p, method = 'euclidian'),
        distance(q, p, method = 'euclidian', mode = 'distance'),
        max_err = MAX_ERR
    )

def test_unknown_method_raises(vectors):
    q, p = vectors
    with pytest.raises(ValueError):
        distance(q, p, method = 'does_not_exist')


# --- `euclidian` : fast path equals the naive one ----------------------------------

def test_euclidian_fast_equals_slow(vectors, asserts):
    q, p = vectors
    asserts.assert_equal(
        euclidian_distance(q, p, fast = False),
        euclidian_distance(q, p, fast = True),
        max_err = MAX_ERR
    )
    asserts.assert_equal(
        euclidian_distance(q, p, fast = False, as_matrix = True),
        euclidian_distance(q, p, fast = True,  as_matrix = True),
        max_err = MAX_ERR
    )


# --- `knn` (faithful conversion of the original test) ------------------------------

@pytest.mark.keras
def test_knn(asserts):
    points_x = np.array([
        [1., 1.], [2., 2.], [2., 1.], [1., 2.],
        [-1., -1.], [-2., -2.], [-2., -1.], [-1., -2.],
        [-1., 1.], [-2., 2.], [-2., 1.], [-1., 2.],
        [1., -1.], [2., -2.], [2., -1.], [1., -2.]
    ], dtype = np.float32)
    points_y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype = np.int32)

    rng = np.random.default_rng(10)
    points_x, points_y = sklearn_shuffle(points_x, points_y, random_state = 10)
    for idx, (x, y) in enumerate(zip(points_x, points_y)):
        indices = [i for i in range(len(points_x)) if i != idx]
        sub_x, sub_y = points_x[indices], points_y[indices]

        asserts.assert_equal(
            np.array([y]), knn(x, sub_x, distance_metric = 'euclidian', ids = sub_y)
        )
        asserts.assert_equal(
            np.array([y]),
            knn(x, sub_x, distance_metric = 'euclidian', ids = sub_y, weighted = True)
        )

        many_x = x + rng.uniform(-0.1, 0.1, size = (16, len(x)))
        asserts.assert_equal(
            [y] * len(many_x), knn(many_x, sub_x, distance_metric = 'euclidian', ids = sub_y)
        )


# --- graph / XLA compatibility (requires tensorflow) -------------------------------

@pytest.mark.tensorflow
@pytest.mark.parametrize('as_matrix', [False, True])
@pytest.mark.parametrize('method', _VECTOR_METHODS)
def test_distance_graph_compatible(vectors, asserts, method, as_matrix):
    # force CPU : `assert_graph_compatible` compares eager vs `tf.function` outputs,
    # and `matmul` / `einsum` accumulate differently on CPU vs GPU (see `_device_cpu`).
    q, p = vectors
    with _device_cpu('tensor'):
        asserts.assert_graph_compatible(distance, q, p, method, as_matrix = as_matrix)

@pytest.mark.tensorflow
def test_knn_graph_compatible(vectors, asserts):
    q, p = vectors
    ids = np.arange(len(p)).astype(np.int32)
    with _device_cpu('tensor'):
        asserts.assert_graph_compatible(
            knn, q, p, distance_metric = 'euclidian', ids = ids
        )
