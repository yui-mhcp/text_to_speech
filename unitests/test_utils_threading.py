# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
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
import unittest

from threading import Thread
from absl.testing import parameterized

from unitests import CustomTestCase, timeout
from utils.threading import *

class TestConsumer(CustomTestCase):
    def _listener(self, it, wait = 0):
        if it == 0: return
        if wait: time.sleep(wait)
        self.assertEqual(it, self.prev + 1)
        self.prev = it
    
    @timeout(0.5)
    def test_main_thread(self):
        cons = Consumer(lambda x: x + 1, run_main_thread = True).start()
        cons.add_listener(self._listener)
        
        self.prev = 0
        for i in range(10): self.assertEqual(cons(i), i + 1)
        
        self.assertTrue(cons.empty)
        self.assertFalse(cons.stopped)
        self.assertFalse(cons.finished)
        
        cons.stop()
        self.assertEqual(self.prev, 10, 'The item listener is probably not called')
        self.assertTrue(cons.stopped)
        self.assertTrue(cons.finished)
    
    @timeout(0.1)
    def test_callback(self):
        def callback(* args, ** kwargs):
            self.callback_called = True
        
        cons = Consumer(lambda x: x + 1, run_main_thread = True).start()
        
        self.callback_called = False
        cons(0, callback = callback)
        cons.stop()
        
        self.assertTrue(self.callback_called)
    
    @timeout(0.1)
    def test_callback_parallel(self):
        def callback(* args, ** kwargs):
            self.callback_called = True
        
        cons = Consumer(lambda x: x + 1, run_main_thread = False).start()
        
        self.callback_called = False
        cons(0, callback = callback)
        cons.join()
        
        self.assertTrue(self.callback_called)


    @timeout(.5)
    def test_append_and_wait(self):
        cons = Consumer(lambda x: x + 1, daemon = True).start()
        
        for i in range(5):
            self.assertEqual(cons.append_and_wait(i), i + 1)
        
        self.assertTrue(cons.empty)
        cons.join()

    @timeout(.5)
    def test_append_and_wait2(self):
        cons = Consumer(lambda x: x + 1, daemon = True)
        cons.add_listener(self._listener, wait = 1e-3)
        
        self.prev = 0
        for i in range(10): cons(i)
        
        self.assertFalse(cons.empty)
        self.assertFalse(cons.stopped)
        self.assertFalse(cons.finished)
        
        cons.start()
        for _ in range(5):
            self.assertEqual(cons.append_and_wait(-1), 0)
            time.sleep(1e-3)
        
        time.sleep(0.2)
        
        self.assertTrue(cons.empty)
        cons.join()
        self.assertTrue(cons.stopped)
        self.assertTrue(cons.finished)
        self.assertTrue(self.prev, 10)
    
    @timeout(0.5)
    def test_stop_when_empty(self):
        cons = Consumer(lambda x: x + 1, daemon = True)
        cons.add_listener(self._listener)
        
        self.prev = 0
        for i in range(10): cons(i)
        
        cons.stop_when_empty()
        
        self.assertFalse(cons.empty)
        self.assertFalse(cons.stopped)
        self.assertFalse(cons.finished)
        self.assertEqual(cons.in_index, 10)
        
        cons.start()
        time.sleep(0.1)

        self.assertTrue(cons.empty)
        self.assertTrue(cons.stopped)
        self.assertTrue(cons.finished)
        self.assertTrue(self.prev, 10)


class TestProducer(CustomTestCase):
    def counter(self, n):
        for i in range(n):
            time.sleep(0.1)
            yield i
    
    def _listener(self, it):
        self.assertEqual(it, self.count)
        self.count += 1

    def _listener_stop(self, it):
        self.assertEqual(it, self.count)
        self.count += 1
        if it == 2: raise StoppedException()

    def _start_listener(self):
        self.assertEqual(self.count, 0)

    def _stop_listener(self, target = 5):
        self.assertFalse(self.joined)
        self.assertEqual(self.count, target)

    def test_main_thread(self):
        self.count = 0
        self.joined = False
        
        prod = Producer(self.counter(5), run_main_thread = True)
        prod.add_listener(self._listener)
        prod.add_listener(self._start_listener, event = 'start')
        prod.add_listener(self._stop_listener, event = 'stop')
        prod.start()
        
        self.assertFalse(prod.stopped)
        self.assertTrue(prod.finished)
        self.assertEqual(self.count, 5)
        
        prod.join()
        self.joined = True
        
    def test_parallel(self):
        self.count = 0
        self.joined = False
        
        prod = Producer(self.counter(5), run_main_thread = False)
        prod.add_listener(self._listener)
        prod.add_listener(self._start_listener, event = 'start')
        prod.add_listener(self._stop_listener, event = 'stop')
        prod.start()
        
        self.assertEqual(self.count, 0)
        self.assertFalse(prod.finished)
        
        prod.join()
        self.joined = True
        
        self.assertEqual(self.count, 5)
        self.assertTrue(prod.finished)

    def test_parallel2(self):
        self.count = 0
        self.joined = False
        
        prod = Producer(
            self.counter(5),
            start_listener  = self._start_listener,
            item_listener   = self._listener,
            stop_listener   = self._stop_listener,
        ).start()
        
        self.assertEqual(self.count, 0)
        self.assertFalse(prod.finished)
        
        prod.join()
        self.joined = True
        
        self.assertEqual(self.count, 5)
        self.assertTrue(prod.finished)

    def test_stop(self):
        self.count = 0
        self.joined = False
        
        prod = Producer(
            self.counter(5),
            start_listener  = self._start_listener,
            item_listener   = self._listener_stop,
        ).start()
        
        prod.add_listener(self._stop_listener, target = 3, event = 'stop')
        
        self.assertEqual(self.count, 0)
        self.assertFalse(prod.finished)
        
        prod.join()
        self.joined = True
        self.assertTrue(prod.stopped)
        self.assertTrue(prod.finished)
        self.assertEqual(self.count, 3)

        
class TestThreadedDict(CustomTestCase):
    def test_setitem(self):
        d = ThreadedDict()
        for i in range(5):
            self.assertFalse(i in d)
            self.assertEqual(d.get(i, -1), -1)
            
            d[i] = i ** 2
            
            self.assertTrue(i in d)
            self.assertEqual(d[i], i ** 2)
            self.assertEqual(d.get(i, -1), i ** 2)

    def test_setdefault(self):
        d = ThreadedDict()
        for i in range(5):
            self.assertEqual(d.setdefault(i, i ** 2), i ** 2)
            self.assertEqual(d.setdefault(i, -1), i ** 2)

    def test_pop(self):
        d = ThreadedDict()
        for i in range(5):
            self.assertEqual(d.pop(i, -1), -1)
            d[i] = i ** 2
            self.assertEqual(d.pop(i, -1), i ** 2)
            self.assertFalse(i in d)

    def test_wait(self):
        def wait_and_set(k, v, t):
            time.sleep(t)
            d[k] = v
        
        d = ThreadedDict()
        for i in range(5):
            x = i
            t = Thread(target = wait_and_set, args = (x, x ** 2, (x + 1) / 20.))
            t.start()
        
        for i in range(5):
            self.assertFalse(i in d)
            self.assertEqual(d[i], i ** 2)
    
    def test_wait_update(self):
        def wait_and_set(k, v, t):
            time.sleep(t)
            d[k] = v
        
        d = ThreadedDict()
        for i in range(5):
            x = i
            t = Thread(target = wait_and_set, args = (x, x + 1, (x + 1) / 20.))
            t.start()
            d[i] = i
        
        time.sleep(1e-3)
        for i in range(5):
            self.assertEqual(d.get(i, -1), i)
        
        for i in range(5):
            self.assertEqual(d.wait_for_update(i), i + 1)
