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

import queue

from threading import Thread

from loggers import Timer, timer
from .async_result import AsyncResult
from ..sequence_utils import pad_batch

class InflightBatcher(Thread):
    def __init__(self, fn, max_batch_size = None, wait_time = 1e-4, ** kwargs):
        Thread.__init__(self, daemon = True)
        
        self.fn = fn
        self.kwargs = kwargs
        self.wait_time  = wait_time
        self.max_batch_size = max_batch_size or float('inf')
        
        self._buffer    = queue.Queue()
        self._stopped   = False
        
        self.start()
    
    def __call__(self, inputs, ** kwargs):
        res = AsyncResult()
        self._buffer.put((inputs, kwargs, res))
        res = res.get()
        if isinstance(res, Exception): raise res
        return res
    
    @timer
    def _fill_batch(self, data):
        batch, callbacks, kwargs = [data[0]], [data[2]], data[1]
        try:
            while len(batch) < self.max_batch_size:
                with Timer('wait batch'):
                    # block up to `wait_time` so concurrent in-flight requests have a
                    # chance to arrive and be grouped (raises `queue.Empty` on timeout)
                    data = self._buffer.get(timeout = self.wait_time)

                    if data is None:
                        self._stopped = True
                        break

                for k, v in data[1].items():
                    if k not in kwargs:
                        kwargs[k] = [None] * len(batch) + [v]
                    elif isinstance(kwargs[k], list):
                        kwargs[k].append(v)
                    elif kwargs[k] != v:
                        kwargs[k] = [kwargs[k]] * len(batch) + [v]
                batch.append(data[0])
                callbacks.append(data[2])
        except queue.Empty:
            pass
        
        if len(batch) == 1:
            return batch[0], callbacks, kwargs
        else:
            return _build_batch(batch, ** self.kwargs), callbacks, kwargs
    
    @timer
    def run(self):
        while not self._stopped:
            data = self._buffer.get()
            if data is None:
                self._stopped = True
                continue
            
            batch, callbacks, kwargs = self._fill_batch(data)
            if batch is None: continue
            
            try:
                with Timer('call'):
                    outputs = self.fn(batch, ** kwargs)
                
                with Timer('set results'):
                    if len(callbacks) == 1:
                        callbacks[0](outputs)
                    else:
                        outputs = _unbatch(outputs)
                        for i, callback in enumerate(callbacks):
                            callback(_get_slice(outputs, i))

            except Exception as e:
                for callback in callbacks: callback(e)
    
    def join(self, * args, ** kwargs):
        self.stop()
        return super().join(* args, ** kwargs)
    
    def stop(self):
        self._stopped = True
        self._buffer.put(None)

@timer
def _build_batch(data, ** kwargs):
    if isinstance(data[0], tuple):
        return tuple(_build_batch(b, ** kwargs) for b in zip(* data))
    else:
        # each request carries a leading batch-dim of 1 (`d[0]` drops it), so `pad_batch`
        # concatenates them into a single `(N, ...)` batch. No backend conversion : the
        # batcher stays runtime-agnostic and preserves the input type (numpy stays numpy).
        return pad_batch([d[0] for d in data], ** kwargs)

def _rebuild_tuple(data, items):
    # `namedtuple` takes positional fields (`cls(a, b)`) while a plain `tuple` takes a
    # single iterable (`tuple([a, b])`) — `cls(*items)` would raise on the latter.
    return data.__class__(* items) if hasattr(data, '_fields') else data.__class__(items)

@timer
def _unbatch(data):
    if hasattr(data, 'shape'):
        if len(data.shape):
            return [data[i, None] for i in range(len(data))]
        else:
            return data
    elif isinstance(data, list):
        return tuple(_unbatch(d) for d in data)
    elif isinstance(data, tuple):
        return _rebuild_tuple(data, [_unbatch(d) for d in data])
    elif isinstance(data, dict):
        return {k : _unbatch(v) for k, v in data.items()}
    else:
        return data

@timer
def _get_slice(data, i):
    match data:
        case list():
            return data[i]
        case tuple():
            return _rebuild_tuple(data, [_get_slice(d, i) for d in data])
        case dict():
            return {k : _get_slice(v, i) for k, v in data.items()}
        case _:
            return data