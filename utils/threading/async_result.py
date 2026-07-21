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

import asyncio
import threading

class AsyncResult:
    """
        This class is inspired from the `AsyncResult` used in `multiprocessing.Pool`s
        Additionally, it supports `await` and `aget` for async-compatible functions

        A producer either fills it with `result(value)` (regular result), or with
        `result.set_exception(exc)` : in the latter case, `get / aget` re-raise `exc`
        instead of returning it
    """
    def __init__(self, callback = None, *, loop = None):
        self.loop   = loop
        self._callback  = callback

        self._event     = threading.Event()
        self._abuffer   = asyncio.Queue() if loop is not None else None
        self._result    = None
        self._exception = None

    @property
    def ready(self):
        return self._event.is_set()

    def __call__(self, result):
        self._result = result
        self._notify()

        if self._callback is not None:
            self._callback(result)

    def set_exception(self, exception):
        """ Marks the result as failed : `get / aget` will re-raise `exception` """
        self._exception = exception
        self._notify()

    def _notify(self):
        self._event.set()

        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._abuffer.put(True), self.loop)

    def wait(self, timeout = None):
        return self._event.wait(timeout)

    def get(self, timeout = None):
        if not self.wait(timeout):
            raise TimeoutError('The result is not available after {}s'.format(timeout))
        elif self._exception is not None:
            raise self._exception
        return self._result

    async def aget(self, timeout = None):
        if self._abuffer is None:
            raise RuntimeError(
                '`aget` requires the `loop` argument at construction time (e.g., `AsyncResult(loop = asyncio.get_running_loop())`)'
            )

        if timeout is None:
            await self._abuffer.get()
        else:
            await asyncio.wait_for(self._abuffer.get(), timeout)

        if self._exception is not None:
            raise self._exception
        return self._result
