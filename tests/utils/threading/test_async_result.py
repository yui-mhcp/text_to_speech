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

""" Pure-``pytest`` tests for :class:`utils.threading.AsyncResult`.

``AsyncResult`` is the ``multiprocessing.Pool``-style future used across the
threading utils : a worker fills it via ``__call__(result)`` and consumers read it
with ``get`` / ``wait`` (blocking) or ``aget`` (asyncio). New coverage — the class
had none before.
"""

import time
import asyncio

from threading import Thread

import pytest

from utils import AsyncResult

from tests.utils.threading._helpers import run_with_timeout

pytestmark = pytest.mark.timeout(10)


def test_not_ready_before_set():
    res = AsyncResult()
    assert not res.ready
    # `get` with a timeout raises instead of silently returning the unset default
    with pytest.raises(TimeoutError):
        res.get(timeout = 0.01)
    assert not res.ready


def test_get_returns_result_after_set():
    res = AsyncResult()
    res(42)
    assert res.ready
    assert res.get() == 42
    assert res.get(timeout = 0) == 42      # idempotent, never blocks once ready


def test_callback_is_called_with_result():
    captured = []
    res = AsyncResult(callback = captured.append)
    res(7)
    assert res.ready
    assert captured == [7]


def test_get_blocks_until_set_from_another_thread():
    res = AsyncResult()

    def producer():
        time.sleep(0.05)
        res('done')

    Thread(target = producer, daemon = True).start()
    # `get` must block until the producer sets the result (bounded against deadlock)
    assert run_with_timeout(res.get, timeout = 2) == 'done'


def test_wait_timeout_returns_when_unset():
    res = AsyncResult()
    started = time.time()
    res.wait(0.02)                          # returns even though never set
    assert (time.time() - started) < 1
    assert not res.ready


def test_set_exception_reraises_in_get():
    res = AsyncResult()
    res.set_exception(ValueError('boom'))
    assert res.ready
    with pytest.raises(ValueError, match = 'boom'):
        res.get()


def test_set_exception_reraises_in_aget():
    async def main():
        loop = asyncio.get_running_loop()
        res  = AsyncResult(loop = loop)

        def producer():
            time.sleep(0.05)
            res.set_exception(ValueError('boom'))

        Thread(target = producer, daemon = True).start()
        return await res.aget()

    with pytest.raises(ValueError, match = 'boom'):
        asyncio.run(main())


def test_exception_result_value_is_not_raised():
    """ An exception passed as a *regular* result (`res(exc)`) stays a value :
        only `set_exception` switches `get` to re-raise mode (cf `InflightBatcher`) """
    res = AsyncResult()
    res(ValueError('value, not error'))
    assert isinstance(res.get(), ValueError)


def test_aget_without_loop_raises():
    res = AsyncResult()

    async def main():
        return await res.aget()

    with pytest.raises(RuntimeError):
        asyncio.run(main())


def test_aget_resolves_via_event_loop():
    async def main():
        loop = asyncio.get_running_loop()
        res  = AsyncResult(loop = loop)

        def producer():
            time.sleep(0.05)
            res(123)                        # cross-thread set -> schedules on the loop

        Thread(target = producer, daemon = True).start()
        return await res.aget()

    assert asyncio.run(main()) == 123
