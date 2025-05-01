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

import os
import time
import queue
import inspect
import logging
import unittest
import multiprocessing
import multiprocessing.queues
import numpy as np
import pandas as pd

from absl.testing import parameterized

from . import CustomTestCase
from utils import STOP, KEEP_ALIVE, IS_RUNNING, CONTROL, DataWithResult, Stream, create_iterable

def _generator():
    for i in range(1, 5):
        yield i

class TestStream(CustomTestCase, parameterized.TestCase):
    def _reset_counters(self):
        self._counter = 0
        self._start_called      = 0
        self._callback_called   = 0
        self._stop_called       = 0
    
    def _check_counters(self, start, stop, counter, callback):
        self.assertEqual(start, self._start_called)
        self.assertEqual(stop, self._stop_called)
        self.assertEqual(callback, self._callback_called)
        if counter: self.assertEqual(counter, self._counter)
        
    
    def _counter_check(self, i):
        self.assertTrue(i, self._counter + 1)
        self._counter += 1
        return i
    
    def _start_check(self):
        self.assertEqual(0, self._start_called)
        self.assertEqual(0, self._counter)
        self._start_called += 1
    
    def _stop_check(self):
        self.assertEqual(1, self._start_called)
        self.assertEqual(0, self._stop_called)
        self._stop_called += 1

        
    def _callback_check(self, i):
        self._callback_called += 1
        if i is not CONTROL:
            self.assertTrue(self._counter, i)

    @parameterized.named_parameters(
        ('list', [1, 2, 3, 4]),
        ('tuple', (1, 2, 3, 4)),
        ('set', {1, 2, 3, 4}),
        ('range', range(1, 5)),
        ('array', np.arange(1, 5)),
        ('queue', queue.Queue()),
        ('multiprocessing.queue', multiprocessing.Queue()),
        ('dataframe', pd.DataFrame([{'x' : 1}, {'x' : 2}, {'x' : 3}])),
        ('function', _generator),
        ('generator', _generator())
    )
    def test_create_iterable(self, data):
        iterable = create_iterable(data)
        self.assertTrue(inspect.isgenerator(iterable), str(iterable))
        
        if not isinstance(data, (pd.DataFrame, queue.Queue, multiprocessing.queues.Queue)):
            for i, item in enumerate(iterable, start = 1):
                self.assertEqual(i, int(item))
    
    @parameterized.named_parameters(
        ('list', [1, 2, 3, 4]),
        ('tuple', (1, 2, 3, 4)),
        ('set', {1, 2, 3, 4}),
        ('range', range(1, 5)),
        ('array', np.arange(1, 5)),
        ('function', _generator),
        ('generator', _generator())
    )
    def test_simple_iterator(self, stream):
        self._reset_counters()
        res = list(Stream(
            self._counter_check, stream
        ))

        self._check_counters(start = 0, stop = 0, callback = 0, counter = 4)
        self.assertEqual([1, 2, 3, 4], res)

    def test_iterator_with_callbacks(self):
        self._reset_counters()
        res = list(Stream(
            self._counter_check,
            range(1, 5),
            callback    = self._callback_check,
            start_callback  = self._start_check,
            stop_callback   = self._stop_check,
        ))

        self._check_counters(start = 1, stop = 1, callback = 4, counter = 4)
        self.assertEqual([1, 2, 3, 4], res)

    @parameterized.named_parameters(
        ('simple', queue.Queue()),
        ('multiprocessing', multiprocessing.Queue())
    )
    def test_iterator_with_callback_queue(self, q):
        self._reset_counters()
        res = list(Stream(
            self._counter_check,
            range(1, 5),
            callback    = [q, self._callback_check],
            start_callback  = self._start_check,
            stop_callback   = self._stop_check
        ))
        self._check_counters(start = 1, stop = 1, callback = 4, counter = 4)
        self.assertEqual([1, 2, 3, 4], res)
        self.assertEqual(4, q.qsize())
        self.assertEqual([1, 2, 3, 4], [q.get().result for _ in range(q.qsize())])

    @parameterized.parameters(0, 1, 4)
    def test_iterator_with_contols(self, max_workers):
        q = queue.Queue()
        
        self._reset_counters()
        res = list(Stream(
            self._counter_check,
            [1, 2, KEEP_ALIVE, 3, 4],
            callback    = [self._callback_check, q],
            start_callback  = self._start_check,
            stop_callback   = self._stop_check,
            max_workers = max_workers
        ))

        self._check_counters(start = 1, stop = 1, callback = 4, counter = 4)
        if max_workers > 1: res = sorted(res)
        self.assertEqual([1, 2, 3, 4], res)
        self.assertEqual(4, q.qsize())
        if max_workers <= 1:
            self.assertEqual([1, 2, 3, 4], [q.get().result for _ in range(q.qsize())])
        else:
            self.assertEqual({1, 2, 3, 4}, set(q.get().result for _ in range(q.qsize())))
        
        self._reset_counters()
        res = list(Stream(
            self._counter_check,
            [1, 2, IS_RUNNING, 3, 4],
            callback    = [self._callback_check, q],
            start_callback  = self._start_check,
            stop_callback   = self._stop_check,
            max_workers = max_workers
        ))

        self._check_counters(start = 1, stop = 1, callback = 4, counter = 4)
        if max_workers > 1: res = sorted(res)
        self.assertEqual([1, 2, 3, 4], res)
        self.assertEqual(5, q.qsize())
        if max_workers <= 1:
            self.assertEqual([1, 2, CONTROL, 3, 4], [q.get().result for _ in range(q.qsize())])
        else:
            self.assertEqual({1, 2, CONTROL, 3, 4}, set(q.get().result for _ in range(q.qsize())))


        self._reset_counters()
        res = list(Stream(
            self._counter_check,
            [1, 2, STOP, 3, 4],
            callback    = [self._callback_check, q],
            start_callback  = self._start_check,
            stop_callback   = self._stop_check,
            max_workers = max_workers
        ))

        self._check_counters(start = 1, stop = 1, callback = 2, counter = 2)
        if max_workers > 1: res = sorted(res)
        self.assertEqual([1, 2], res)
        self.assertEqual(3, q.qsize())
        if max_workers <= 1:
            self.assertEqual([1, 2, CONTROL], [q.get().result for _ in range(q.qsize())])
        else:
            self.assertEqual({1, 2, CONTROL}, set(q.get().result for _ in range(q.qsize())))

    def test_dict_as_kwargs(self):
        def foo(x, y = -1):
            self.assertEqual(2, y)
            return x ** y
        
        res = list(Stream(
            foo, [{'x' : i, 'y' : 2} for i in range(1, 5)], dict_as_kwargs = True
        ))
        self.assertEqual([1, 4, 9, 16], res)

        def foo2(x, y = -1):
            self.assertEqual(-1, y)
            return x['x'] ** x['y']
        
        res = list(Stream(
            foo2, [{'x' : i, 'y' : 2} for i in range(1, 5)], dict_as_kwargs = False
        ))
        self.assertEqual([1, 4, 9, 16], res)

    @parameterized.parameters(1, 2, 3, 4)
    def test_multi_threaded_iterator(self, max_workers):
        def foo(x):
            time.sleep(0.1)
            return x
        
        def _set_start_time():
            self.assertTrue(not hasattr(self, '_stop_time'))
            self._start_time = time.time()
        
        self._reset_counters()

        n = max(2, 2 * max_workers)
        max_t = 0.1 * n / max_workers + 0.09
        
        stream = Stream(
            foo,
            range(n),
            start_callback  = [self._start_check, _set_start_time],
            stop_callback   = self._stop_check,
            max_workers = max_workers
        )
        self.assertFalse(stream.is_alive(), 'The stream should not start before the iteration start')
        
        res = []
        for it in stream:
            self._stop_time = time.time()
            res.append(it)
        
        self._check_counters(start = 1, stop = 1, callback = 0, counter = None)
        
        total_t = self._stop_time - self._start_time
        self.assertTrue(total_t < max_t, 'The function took {:.3f} sec, expected ~{}'.format(
            total_t, max_t
        ))
        del self._stop_time
        self._check_counters(start = 1, stop = 1, callback = 0, counter = None)
        if max_workers > 1: res = sorted(res)
        self.assertEqual(list(range(2 * max_workers)), res)

    def test_simple_callable(self):
        self._reset_counters()
        stream = Stream(self._counter_check)
        
        res = []
        for it in range(1, 5):
            res.append(stream(it))
        stream.join()
        
        self._check_counters(start = 0, stop = 0, callback = 0, counter = 4)
        self.assertEqual([1, 2, 3, 4], res)

    def test_callable_with_callbacks(self):
        self._reset_counters()
        stream = Stream(
            self._counter_check,
            callback    = self._callback_check,
            start_callback  = self._start_check,
            stop_callback   = self._stop_check,
        )

        res = []
        for it in range(1, 5):
            res.append(stream(it))
        self._check_counters(start = 1, stop = 0, callback = 4, counter = 4)
        
        stream.join()

        self._check_counters(start = 1, stop = 1, callback = 4, counter = 4)
        self.assertEqual([1, 2, 3, 4], res)
