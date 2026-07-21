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

""" Pure-``pytest`` tests for :class:`utils.threading.PriorityQueue` and
:class:`utils.threading.PriorityItem`.

``PriorityQueue`` wraps :class:`queue.PriorityQueue` to (a) infer an item's priority
from several shapes (explicit arg, ``(priority, data)`` tuple, ``{'priority': ...}``
dict, or a ``.priority`` attribute) and (b) keep insertion order stable for equal
priorities via a monotonic index. New coverage — only the multi-process variant is
left out (deferred, multiprocessing-heavy).
"""

import queue

from types import SimpleNamespace

import pytest

from utils import STOP, KEEP_ALIVE, IS_RUNNING, PriorityQueue, PriorityItem

pytestmark = pytest.mark.timeout(10)


# --- PriorityItem (the comparable wrapper) -----------------------------------------

def test_priority_item_orders_by_priority_then_index():
    assert PriorityItem(1, 0, 'a') < PriorityItem(2, 0, 'b')
    # equal priority -> the lower index wins (stable FIFO)
    assert PriorityItem(1, 0, 'a') < PriorityItem(1, 1, 'b')


def test_priority_item_ignores_data_in_comparison():
    # `data` is `compare = False` : two items with same (priority, index) are equal
    assert PriorityItem(1, 0, 'a') == PriorityItem(1, 0, 'totally-different')


# --- PriorityQueue : ordering ------------------------------------------------------

def test_orders_by_explicit_priority():
    pq = PriorityQueue()
    pq.put('c', priority = 3)
    pq.put('a', priority = 1)
    pq.put('b', priority = 2)
    assert [pq.get(), pq.get(), pq.get()] == ['a', 'b', 'c']


def test_fifo_within_same_priority():
    pq = PriorityQueue()
    for v in ('x', 'y', 'z'):
        pq.put(v, priority = 1)
    assert [pq.get(), pq.get(), pq.get()] == ['x', 'y', 'z']


# --- PriorityQueue : priority inference --------------------------------------------

def test_priority_from_tuple():
    pq = PriorityQueue()
    pq.put((2, 'two'))
    pq.put((1, 'one'))
    # the stored data is the whole tuple ; priority is taken from item[0]
    assert pq.get() == (1, 'one')
    assert pq.get() == (2, 'two')


def test_priority_from_dict():
    pq = PriorityQueue()
    pq.put({'priority' : 5, 'v' : 'low'})
    pq.put({'priority' : 1, 'v' : 'high'})
    assert pq.get()['v'] == 'high'
    assert pq.get()['v'] == 'low'


def test_priority_from_attribute():
    pq = PriorityQueue()
    pq.put(SimpleNamespace(priority = 2, v = 'b'))
    pq.put(SimpleNamespace(priority = 1, v = 'a'))
    assert pq.get().v == 'a'
    assert pq.get().v == 'b'


def test_missing_priority_raises():
    pq = PriorityQueue()
    with pytest.raises(ValueError):
        pq.put('no-priority-here')


# --- PriorityQueue : control tokens -------------------------------------------------

def test_control_tokens_do_not_raise():
    """ Regression : `Process.stop / keep_alive` put raw tokens (no `priority`), which
        used to crash `_build_item` with a `ValueError` on the consumer side """
    pq = PriorityQueue()
    pq.put(STOP)
    pq.put(KEEP_ALIVE)
    pq.put(IS_RUNNING)
    assert pq.qsize() == 3


def test_control_tokens_ordering():
    """ The liveness ping (`IS_RUNNING`) jumps ahead of everything ; `STOP` and
        `KEEP_ALIVE` are drained last (drain-then-stop semantic) """
    pq = PriorityQueue()
    pq.put('normal', priority = 0)
    pq.put(STOP)
    pq.put(IS_RUNNING)

    assert pq.get() == IS_RUNNING
    assert pq.get() == 'normal'
    assert pq.get() is STOP


def test_explicit_priority_overrides_inference():
    pq = PriorityQueue()
    # tuple would infer priority 99, but the explicit arg takes precedence
    pq.put((99, 'a'), priority = 10)
    pq.put((99, 'b'), priority = 1)
    assert pq.get() == (99, 'b')
    assert pq.get() == (99, 'a')


# --- PriorityQueue : accessors -----------------------------------------------------

def test_return_full_item_exposes_wrapper():
    pq = PriorityQueue()
    pq.put('a', priority = 2)
    item = pq.get(return_full_item = True)
    assert isinstance(item, PriorityItem)
    assert item.priority == 2
    assert item.data == 'a'


def test_put_nowait():
    pq = PriorityQueue()
    pq.put_nowait('a', priority = 1)
    assert pq.qsize() == 1
    assert pq.get() == 'a'


def test_get_nowait():
    pq = PriorityQueue()
    pq.put('a', priority = 1)
    assert pq.get_nowait() == 'a'


def test_get_nowait_raises_when_empty():
    pq = PriorityQueue()
    with pytest.raises(queue.Empty):
        pq.get_nowait()
