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

""" Pure-``pytest`` tests for :class:`utils.threading.InflightBatcher`.

The batcher groups concurrent single-sample calls into one padded batch, runs ``fn``
once, then scatters the per-sample outputs back to each caller. The input contract is
the usual *in-flight batching* one : **each call carries a leading batch-dim of 1**
(shape ``(1, ...)``) ; the batcher concatenates them into a single ``(N, ...)`` batch
and splits the result back into ``(1, ...)`` slices.

All inputs here are ``np.ndarray`` : ``_build_batch`` keeps the batch as numpy (no
backend conversion), so these tests stay **keras-free** — the assertions explicitly
expect ``np.ndarray`` (via the ``asserts`` fixture / ``tests.asserts``).

Determinism : grouping is time-based (the worker waits up to ``wait_time`` for stragglers
to arrive). A comfortable ``wait_time`` is used so the whole burst is reliably batched
together — short enough to keep the suite fast, long enough to beat the thread-pool
scheduling jitter.

Assertions inside the worker ``fn`` are surfaced to the caller : on error the batcher
forwards the exception to every pending callback and ``__call__`` re-raises it — so a
failing ``assert`` in ``fn`` fails the test.
"""

import numpy as np
import pytest

from multiprocessing.pool import ThreadPool

from utils import InflightBatcher
from utils.threading.inflight_batcher import _build_batch, _unbatch, _get_slice

# comfortable grouping window (see module docstring) — well above scheduling jitter
WAIT_TIME = 0.05

# the batcher must never hang the run (e.g. a dropped callback)
pytestmark = pytest.mark.timeout(15)


# --- end-to-end batching -----------------------------------------------------------

def test_simple(asserts):
    """ A burst of 4 single-sample calls is grouped, run once, then scattered back. """
    def foo(batch):
        asserts.assert_array(batch)
        assert len(batch) == 4
        return batch ** 2

    batcher = InflightBatcher(foo, wait_time = WAIT_TIME)
    try:
        with ThreadPool(4) as pool:
            outputs = [pool.apply_async(batcher, ([i], )) for i in range(4)]
            outputs = [out.get() for out in outputs]
        asserts.assert_equal([[i ** 2] for i in range(4)], outputs)

        images = [np.random.uniform(size = (1, 32, 32, 3)) for _ in range(4)]
        with ThreadPool(4) as pool:
            outputs = [pool.apply_async(batcher, (img, )) for img in images]
            outputs = [out.get() for out in outputs]
        asserts.assert_equal([img ** 2 for img in images], outputs)
    finally:
        batcher.stop()


def test_max_batch_size(asserts):
    """ With ``max_batch_size = 4`` a flood of 16 calls is split into batches of 4. """
    def foo(batch):
        asserts.assert_array(batch)
        assert len(batch) == 4
        return batch ** 2

    batcher = InflightBatcher(foo, 4, wait_time = WAIT_TIME)
    try:
        with ThreadPool(16) as pool:
            outputs = [pool.apply_async(batcher, ([i], )) for i in range(16)]
            outputs = [out.get() for out in outputs]
        asserts.assert_equal([[i ** 2] for i in range(16)], outputs)
    finally:
        batcher.stop()


def test_padding_with_kwargs_and_multiple_outputs(asserts):
    """ Variable-length inputs are padded to the longest, per-call kwargs are gathered
        into a list, and a tuple output is correctly un-batched. """
    _lengths = [16, 32, 64, 128]
    tokens   = [np.arange(length)[None] for length in _lengths]

    def foo(batch, length):
        asserts.assert_array(batch)
        asserts.assert_equal((4, max(_lengths)), batch.shape)
        assert isinstance(length, list)
        assert set(_lengths) == set(length)
        return batch, np.array(length)

    batcher = InflightBatcher(foo, wait_time = WAIT_TIME, pad_value = -1)
    try:
        with ThreadPool(4) as pool:
            outputs = [
                pool.apply_async(batcher, (tok, ), {'length' : l})
                for tok, l in zip(tokens, _lengths)
            ]
            outputs = [out.get() for out in outputs]
    finally:
        batcher.stop()

    for out, t, l in zip(outputs, tokens, _lengths):
        assert isinstance(out, tuple), str(out)
        out, length = out
        asserts.assert_equal(t, out[:, :l])
        asserts.assert_equal([l], length)


def test_single_call_is_not_batched(asserts):
    """ A lone call (no concurrent traffic) is forwarded as-is, keeping its ``(1, ...)``
        shape — the single-sample fast-path must not add/drop a batch dimension. """
    def foo(batch):
        asserts.assert_array(batch)
        asserts.assert_equal((1, 3), batch.shape)
        return batch ** 2

    batcher = InflightBatcher(foo, wait_time = WAIT_TIME)
    try:
        out = batcher(np.array([[1., 2., 3.]]))
    finally:
        batcher.stop()
    asserts.assert_equal([[1., 4., 9.]], out)


# --- error handling / lifecycle ----------------------------------------------------

def test_exception_is_raised_to_caller():
    """ An exception raised by ``fn`` is forwarded to the caller, not swallowed. """
    def foo(batch):
        raise ValueError('boom')

    batcher = InflightBatcher(foo)
    try:
        with pytest.raises(ValueError, match = 'boom'):
            batcher([1])
    finally:
        batcher.stop()


def test_exception_is_broadcast_to_every_caller():
    """ A failing batch forwards the exception to *all* pending callbacks. """
    def foo(batch):
        raise ValueError('boom')

    batcher = InflightBatcher(foo, wait_time = WAIT_TIME)
    try:
        with ThreadPool(4) as pool:
            results = [pool.apply_async(batcher, ([i], )) for i in range(4)]
            for res in results:
                with pytest.raises(ValueError, match = 'boom'):
                    res.get()
    finally:
        batcher.stop()


def test_join_stops_thread():
    """ ``join`` stops the worker loop and the thread actually terminates. """
    batcher = InflightBatcher(lambda batch: batch)
    assert batcher.is_alive()
    batcher.join(timeout = 5)
    assert not batcher.is_alive()


def test_stop_is_idempotent():
    """ Calling ``stop`` twice must not raise. """
    batcher = InflightBatcher(lambda batch: batch)
    batcher.stop()
    batcher.stop()


# --- pure helpers (deterministic, thread-free) -------------------------------------

class TestBuildBatch:
    """ ``_build_batch`` concatenates per-request ``(1, ...)`` inputs into ``(N, ...)``. """

    def test_concatenates_same_shape(self, asserts):
        batch = _build_batch([np.zeros((1, 3)), np.ones((1, 3))])
        asserts.assert_array(batch)
        asserts.assert_equal((2, 3), batch.shape)
        asserts.assert_equal([[0, 0, 0], [1, 1, 1]], batch)

    def test_pads_variable_length(self, asserts):
        batch = _build_batch([np.arange(2)[None], np.arange(4)[None]], pad_value = -1)
        asserts.assert_array(batch)
        asserts.assert_equal((2, 4), batch.shape)
        asserts.assert_equal([[0, 1, -1, -1], [0, 1, 2, 3]], batch)

    def test_tuple_inputs_build_tuple_of_batches(self, asserts):
        data  = [(np.zeros((1, 2)), np.arange(3)[None]),
                 (np.ones((1, 2)),  np.arange(3)[None])]
        first, second = _build_batch(data)
        asserts.assert_equal((2, 2), first.shape)
        asserts.assert_equal((2, 3), second.shape)

    def test_output_is_numpy_not_backend_tensor(self, asserts):
        # regression : `_build_batch` must not pull a backend tensor (keras-free contract)
        asserts.assert_array(_build_batch([np.zeros((1, 4)), np.zeros((1, 4))]))


class TestUnbatch:
    """ ``_unbatch`` splits a ``(N, ...)`` batch into a list of ``(1, ...)`` slices. """

    def test_array_split_keeps_unit_batch_dim(self, asserts):
        out = _unbatch(np.arange(6).reshape(3, 2))
        assert isinstance(out, list) and len(out) == 3
        for i, sl in enumerate(out):
            asserts.assert_equal((1, 2), sl.shape)
            asserts.assert_equal([[2 * i, 2 * i + 1]], sl)

    def test_scalar_batch_is_returned_as_is(self, asserts):
        scalar = np.array(5)
        assert _unbatch(scalar) is scalar

    def test_tuple_output_is_split_componentwise(self, asserts):
        # a 2-output batch (each member shape (2,)) -> a 2-tuple of 2-slice lists
        a, b = _unbatch((np.array([0, 1]), np.array([10, 11])))
        asserts.assert_equal([[0], [1]], a)        # one (1,) slice per sample ...
        asserts.assert_equal([[10], [11]], b)      # ... mirrored on each tuple member


class TestGetSlice:
    """ ``_get_slice`` picks the ``i``-th element of an already un-batched structure. """

    def test_list(self, asserts):
        assert _get_slice(['a', 'b', 'c'], 1) == 'b'

    def test_tuple_recurses_into_members(self, asserts):
        assert _get_slice((['a', 'b'], ['x', 'y']), 1) == ('b', 'y')

    def test_dict_recurses_into_values(self, asserts):
        assert _get_slice({'k' : ['a', 'b']}, 0) == {'k' : 'a'}

    def test_scalar_is_returned_as_is(self, asserts):
        assert _get_slice(42, 3) == 42
