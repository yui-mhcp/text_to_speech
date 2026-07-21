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

""" Pure-``pytest`` tests for :class:`utils.threading.Stream`.

Rewritten from the former ``unittest`` / ``absl.parameterized`` suite : plain
module-level ``test_*`` functions + ``@pytest.mark.parametrize``. The per-``TestCase``
counter attributes are replaced by the thread-safe :class:`Recorder` spy (``recorder``
fixture). Comparisons are plain Python (everything here is ints / sets / sentinels),
so the tolerant ``asserts`` fixture is not needed.

Every test is bounded by ``@pytest.mark.timeout`` (module-level) so a regression that
deadlocks a worker thread fails loudly instead of hanging the run.
"""

import time
import queue
import multiprocessing

from functools import partial
from threading import Thread

import pytest

from utils import STOP, KEEP_ALIVE, IS_RUNNING, CONTROL, Stream

from tests.utils.threading._helpers import drain_queue, run_with_timeout

# safety-net : no Stream test below should ever need more than a few seconds
pytestmark = pytest.mark.timeout(15)


def _generator():
    for i in range(1, 5):
        yield i


def _results(q):
    """ Drain a callback ``Queue`` of ``DataWithResult`` into the list of results. """
    return [item.result for item in drain_queue(q)]


# --- iterator mode -----------------------------------------------------------------

@pytest.mark.parametrize('stream', [
    pytest.param([1, 2, 3, 4],          id = 'list'),
    pytest.param((1, 2, 3, 4),          id = 'tuple'),
    pytest.param({1, 2, 3, 4},          id = 'set'),
    pytest.param(range(1, 5),           id = 'range'),
    pytest.param('array',               id = 'array'),       # built lazily (keras-free numpy)
    pytest.param(_generator,            id = 'function'),
    pytest.param(_generator(),          id = 'generator'),
])
def test_iterator_simple(recorder, stream):
    if stream == 'array':
        import numpy as np
        stream = np.arange(1, 5)

    res = list(Stream(recorder.counter_check, stream))

    recorder.check_counters(start = 0, stop = 0, callback = 0, counter = 4)
    assert [1, 2, 3, 4] == res


def test_iterator_with_callbacks(recorder):
    res = list(Stream(
        recorder.counter_check,
        range(1, 5),
        callback        = recorder.callback_check,
        start_callback  = recorder.start_check,
        stop_callback   = recorder.stop_check,
    ))

    recorder.check_counters(start = 1, stop = 1, callback = 4, counter = 4)
    assert [1, 2, 3, 4] == res


@pytest.mark.parametrize('make_queue', [
    pytest.param(queue.Queue,           id = 'simple'),
    pytest.param(multiprocessing.Queue, id = 'multiprocessing'),
])
def test_iterator_with_callback_queue(recorder, make_queue):
    q = make_queue()
    res = list(Stream(
        recorder.counter_check,
        range(1, 5),
        callback        = [q, recorder.callback_check],
        start_callback  = recorder.start_check,
        stop_callback   = recorder.stop_check,
    ))

    recorder.check_counters(start = 1, stop = 1, callback = 4, counter = 4)
    assert [1, 2, 3, 4] == res
    assert q.qsize() == 4
    assert [1, 2, 3, 4] == _results(q)


@pytest.mark.parametrize('max_workers', [0, 1, 2])
def test_iterator_with_keep_alive(recorder, max_workers):
    q = queue.Queue()
    res = list(Stream(
        partial(recorder.counter_check, max_workers = max_workers),
        [1, 2, KEEP_ALIVE, 3, 4],
        callback        = [recorder.callback_check, q],
        start_callback  = recorder.start_check,
        stop_callback   = recorder.stop_check,
        max_workers     = max_workers,
    ))

    recorder.check_counters(start = 1, stop = 1, callback = 4, counter = 4)
    if max_workers > 1: res = sorted(res)
    assert [1, 2, 3, 4] == res

    assert q.qsize() == 4
    if max_workers <= 1:
        assert [1, 2, 3, 4] == _results(q)
    else:
        assert {1, 2, 3, 4} == set(_results(q))


@pytest.mark.parametrize('max_workers', [0, 1, 2])
def test_iterator_with_is_running(recorder, max_workers):
    q = queue.Queue()
    res = list(Stream(
        partial(recorder.counter_check, max_workers = max_workers),
        [1, 2, IS_RUNNING, 3, 4],
        callback        = [recorder.callback_check, q],
        start_callback  = recorder.start_check,
        stop_callback   = recorder.stop_check,
        max_workers     = max_workers,
    ))

    recorder.check_counters(start = 1, stop = 1, callback = 4, counter = 4)
    if max_workers > 1: res = sorted(res)
    assert [1, 2, 3, 4] == res

    # IS_RUNNING forwards a CONTROL marker to the callback queue (but not to `fn`)
    assert q.qsize() == 5
    if max_workers <= 1:
        assert [1, 2, CONTROL, 3, 4] == _results(q)
    else:
        assert {1, 2, CONTROL, 3, 4} == set(_results(q))


@pytest.mark.parametrize('max_workers', [0, 1, 2])
def test_iterator_with_stop(recorder, max_workers):
    q = queue.Queue()
    res = list(Stream(
        partial(recorder.counter_check, max_workers = max_workers),
        [1, 2, STOP, 3, 4],
        callback        = [recorder.callback_check, q],
        start_callback  = recorder.start_check,
        stop_callback   = recorder.stop_check,
        max_workers     = max_workers,
    ))

    # STOP halts the stream : items after it are never processed
    recorder.check_counters(start = 1, stop = 1, callback = 2, counter = 2)
    if max_workers > 1: res = sorted(res)
    assert [1, 2] == res

    if max_workers <= 1:
        assert q.qsize() == 3
        assert [1, 2, CONTROL] == _results(q)
    else:
        assert {1, 2, CONTROL} == set(_results(q))


@pytest.mark.parametrize('max_workers', [0, 1, 2])
def test_iterable_with_errors(recorder, max_workers):
    def foo(i):
        # `0` is a deliberate "bad" input : raise an explicit, readable error
        # (caught by the Stream via `stop_on_error`) rather than a real
        # ZeroDivisionError. It must surface asap so the stream stops before
        # starting the 4th item.
        if i == 0:
            raise ValueError('intentional test error')
        res = 10 // i
        time.sleep(0.01)
        return res

    q = queue.Queue()
    res = list(Stream(
        foo,
        [1, 2, 0, 3, 4, 0],
        callback        = [recorder.callback_check, q],
        stop_on_error   = True,
        start_callback  = recorder.start_check,
        stop_callback   = recorder.stop_check,
        max_workers     = max_workers,
    ))

    recorder.check_counters(start = 1, stop = 1, callback = 3, counter = 0)

    assert q.qsize() == 3
    results = _results(q)
    # the error (`foo(0)`) raises instantly while the successes sleep, so with several
    # workers it can be enqueued *between* them — only the multiset of results is stable,
    # not the error's position. Split successes / errors instead of indexing.
    errors    = [r for r in results if isinstance(r, Exception)]
    successes = [r for r in results if not isinstance(r, Exception)]
    assert len(errors) == 1 and isinstance(errors[0], ValueError), str(results)
    if max_workers <= 1:
        assert [10, 5] == successes      # sequential mode keeps input order
    else:
        assert {10, 5} == set(successes)


@pytest.mark.parametrize('max_workers', [0, 1, 2])
def test_iterable_with_priority(recorder, max_workers):
    def foo(i):
        time.sleep(0.05)
        return i

    s = queue.PriorityQueue()
    q = queue.Queue()
    for i in [4, 3, 2, 1]:
        s.put(i)

    res = list(Stream(
        foo, s,
        callback        = [recorder.callback_check, q],
        start_callback  = recorder.start_check,
        stop_callback   = recorder.stop_check,
        max_workers     = max_workers,
        timeout         = 1e-3,
    ))

    recorder.check_counters(start = 1, stop = 1, callback = 4, counter = 0)
    if max_workers <= 1:
        assert [1, 2, 3, 4] == res
        assert q.qsize() == 4
        assert [1, 2, 3, 4] == _results(q)
    else:
        assert {1, 2} == set(res[:2])
        assert q.qsize() == 4
        assert {1, 2, 3, 4} == set(_results(q))

    # second round : items are pushed *after* the stream starts (delayed producer)
    def delayed_fill_buffer():
        time.sleep(0.01)
        for i in [4, 3, 2, 1]:
            s.put(i)

    s.put(6)
    s.put(5)
    Thread(target = delayed_fill_buffer).start()

    recorder.reset()
    res = list(Stream(
        foo, s,
        callback        = [recorder.callback_check, q],
        start_callback  = recorder.start_check,
        stop_callback   = recorder.stop_check,
        max_workers     = max_workers,
        timeout         = 0.025,
    ))

    recorder.check_counters(start = 1, stop = 1, callback = 6, counter = 0)
    if max_workers <= 1:
        assert [5, 1, 2, 3, 4, 6] == res
        assert q.qsize() == 6
        assert [5, 1, 2, 3, 4, 6] == _results(q)
    else:
        assert {5, 6, 1, 2} == set(res[:4])
        assert q.qsize() == 6
        assert {1, 2, 3, 4, 5, 6} == set(_results(q))


def test_dict_as_kwargs():
    def foo(x, y = -1):
        assert y == 2
        return x ** y

    res = list(Stream(
        foo, [{'x' : i, 'y' : 2} for i in range(1, 5)], dict_as_kwargs = True
    ))
    assert [1, 4, 9, 16] == res

    def foo2(x, y = -1):
        assert y == -1
        return x['x'] ** x['y']

    res = list(Stream(
        foo2, [{'x' : i, 'y' : 2} for i in range(1, 5)], dict_as_kwargs = False
    ))
    assert [1, 4, 9, 16] == res


@pytest.mark.parametrize('max_workers', [1, 2, 3, 4])
def test_iterator_multi_threaded(recorder, max_workers):
    """ Items are processed concurrently : the wall-clock time must be close to
        ``ceil(n / max_workers)`` slots rather than ``n`` sequential slots. """
    timings = {}

    def foo(x):
        time.sleep(0.1)
        return x

    def _set_start_time():
        assert 'stop' not in timings
        timings['start'] = time.time()

    n     = max(2, 2 * max_workers)
    max_t = 0.1 * n / max_workers + 0.09

    stream = Stream(
        foo,
        range(n),
        start_callback  = [recorder.start_check, _set_start_time],
        stop_callback   = recorder.stop_check,
        max_workers     = max_workers,
    )
    assert not stream.is_alive(), 'The stream should not start before the iteration start'

    res = []
    for it in stream:
        timings['stop'] = time.time()
        res.append(it)

    recorder.check_counters(start = 1, stop = 1, callback = 0)

    total_t = timings['stop'] - timings['start']
    assert total_t < max_t, \
        'The function took {:.3f}s, expected ~{:.3f}s'.format(total_t, max_t)

    if max_workers > 1: res = sorted(res)
    assert list(range(2 * max_workers)) == res


# --- callable (function-like) mode -------------------------------------------------

def test_callable_simple(recorder):
    stream = Stream(recorder.counter_check)

    res = [stream(it) for it in range(1, 5)]
    stream.join()

    recorder.check_counters(start = 0, stop = 0, callback = 0, counter = 4)
    assert [1, 2, 3, 4] == res


def test_callable_with_callbacks(recorder):
    stream = Stream(
        recorder.counter_check,
        callback        = recorder.callback_check,
        start_callback  = recorder.start_check,
        stop_callback   = recorder.stop_check,
    )

    res = [stream(it) for it in range(1, 5)]
    # in sequential mode `on_start` fires on the first call, `on_stop` only on join
    recorder.check_counters(start = 1, stop = 0, callback = 4, counter = 4)

    stream.join()

    recorder.check_counters(start = 1, stop = 1, callback = 4, counter = 4)
    assert [1, 2, 3, 4] == res


@pytest.mark.parametrize('max_workers', [1, 2])
def test_callable_threaded_resolves_async_results(recorder, max_workers):
    """ `stream(x)` returns an `AsyncResult` that must eventually resolve (regression :
        the promise used to be silently overwritten and `get` hanged forever) """
    stream = Stream(
        partial(recorder.counter_check, max_workers = max_workers),
        max_workers = max_workers
    )

    res = [stream(it) for it in range(1, 5)]
    out = [run_with_timeout(r.get, timeout = 5) for r in res]
    stream.join()

    assert [1, 2, 3, 4] == out
    recorder.check_counters(start = 0, stop = 0, callback = 0, counter = 4)


@pytest.mark.parametrize('max_workers', [1, 2])
def test_callable_threaded_reraises_errors(max_workers):
    def foo(i):
        if i == 0:
            raise ValueError('intentional test error')
        return i

    stream = Stream(foo, max_workers = max_workers, stop_on_error = False)

    ok, ko = stream(1), stream(0)
    assert run_with_timeout(ok.get, timeout = 5) == 1
    with pytest.raises(ValueError, match = 'intentional test error'):
        run_with_timeout(ko.get, timeout = 5)

    stream.join()


def test_callable_threaded_rapid_calls_start_once():
    """ Two immediate calls must not both `Thread.start()` the stream (regression :
        the started flag was only set asynchronously, in the `Stream` thread) """
    stream = Stream(lambda x: x, max_workers = 1)

    res = [stream(it) for it in range(1, 5)]    # no `RuntimeError : threads can only be started once`
    assert [1, 2, 3, 4] == [run_with_timeout(r.get, timeout = 5) for r in res]
    stream.join()


def test_join_with_pending_items_terminates(recorder):
    """ `join` must terminate even when items are still queued when it is called
        (regression : the `STOP` token was only put when the queue was empty) """
    s = queue.Queue()
    for i in range(1, 5):
        s.put(i)

    stream = Stream(recorder.counter_check, s, max_workers = 1).start()
    run_with_timeout(stream.join, timeout = 5)

    recorder.check_counters(start = 0, stop = 0, callback = 0, counter = 4)


def test_early_break_with_bounded_prefetch_does_not_deadlock():
    """ Breaking out of `items()` with a bounded `prefetch_size` must not deadlock the
        workers blocked on the full results buffer """
    def consume():
        stream = Stream(lambda x: x, range(50), max_workers = 2, prefetch_size = 1)
        for i, (inp, out) in enumerate(stream.items()):
            if i >= 2: break
        return True

    assert run_with_timeout(consume, timeout = 10)


# --- lifecycle / control surface (new coverage) ------------------------------------

def test_stop_blocks_further_calls():
    """ Once stopped, a started stream refuses new items. """
    stream = Stream(lambda x: x)     # sequential (max_workers = 0)
    assert stream(1) == 1            # first call starts the stream (`on_start`)
    stream.stop()
    with pytest.raises(RuntimeError):
        stream(2)


def test_clear_empties_pending_stream():
    """ ``clear()`` drains the input queue without processing it. """
    s = queue.Queue()
    for i in range(5):
        s.put(i)

    stream = Stream(lambda x: x, s, max_workers = 1)
    stream.clear()
    assert s.qsize() == 0
