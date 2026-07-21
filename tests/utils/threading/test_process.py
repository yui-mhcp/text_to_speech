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

""" Pure-``pytest`` tests for :mod:`utils.threading.process`.

Scope : the :func:`run_in_thread` decorator, plus the **parent-side** logic of the
:class:`Process` wrapper (registry / name resolution, result registration, `stop` /
`clear` / `terminate` semantics). No real sub-process is ever spawned : the tests
either never call `start`, or monkeypatch ``multiprocessing.Process`` with an inert
fake — full end-to-end (spawn) coverage is still **deferred** to ``slow``-marked
integration tests (top-level picklable workers, OS-sensitive Windows ``spawn``).
"""

import time
import threading

import pytest

from utils import run_in_thread
from utils.threading import KEEP_ALIVE, Process
from utils.threading import process as process_module

from tests.utils.threading._helpers import run_with_timeout

pytestmark = pytest.mark.timeout(10)


@pytest.fixture(autouse = True)
def _clean_registry():
    """ `MetaProcess` keeps a per-name singleton registry : isolate each test. """
    yield
    with process_module._global_mutex:
        process_module._processes.clear()


class _FakeMPProcess:
    """ Inert stand-in for ``multiprocessing.Process`` (records the constructor
        arguments, never spawns anything). """
    def __init__(self, target = None, args = (), kwargs = None, name = None):
        self.target = target
        self.args   = args
        self.kwargs = dict(kwargs or {})
        self.name   = name

        self.exitcode   = None
        self._alive     = False
        self._joined    = threading.Event()

    def start(self):
        self._alive = True

    def is_alive(self):
        return self._alive

    def join(self, timeout = None):
        self._joined.wait(timeout)

    def terminate(self):
        self._alive     = False
        self.exitcode   = -15
        self._joined.set()


@pytest.fixture
def fake_mp_process(monkeypatch):
    monkeypatch.setattr(process_module.multiprocessing, 'Process', _FakeMPProcess)
    return _FakeMPProcess


# --- Process : registry / construction ----------------------------------------------

class _CallableWorker:
    """ Callable without `name` / `__name__` : the name comes from the class. """
    def __call__(self, ** kwargs):
        pass


def test_name_resolved_from_instance_class():
    # regression : `fn` used to be *replaced* by the class name string
    worker = _CallableWorker()
    p = Process(worker, add_stream = False)

    assert p.name == '_CallableWorker'
    assert p.fn is worker


def test_same_name_returns_same_process():
    p1 = Process(lambda ** kw: None, name = 'singleton-test', add_stream = False)
    p2 = Process(lambda ** kw: None, name = 'singleton-test', add_stream = False)
    assert p1 is p2


def test_explicit_input_stream_is_not_overwritten():
    # regression : `input_stream = 'priority'` used to be clobbered back to 'queue'
    p = Process(lambda ** kw: None, name = 'priority-input-test', input_stream = 'priority')
    assert 'priority' in p.buffer_type.lower()


def test_start_forwards_args_and_streams(fake_mp_process):
    p = Process(
        lambda ** kw: None, (1, 2), {'a' : 3}, name = 'args-test', input_stream = 'queue'
    ).start()

    assert p.process.args == (1, 2)          # regression : `args` used to be dropped
    assert p.process.kwargs['a'] == 3
    assert p.process.kwargs['stream'] is p.input_stream
    assert p.process.kwargs['callback'] is p.output_stream
    p.terminate()


def test_start_with_skip_outputs_wires_control_callback(fake_mp_process):
    p = Process(
        lambda ** kw: None, name = 'skip-outputs-wiring-test',
        input_stream = 'queue', skip_outputs = True
    ).start()

    assert 'callback' not in p.process.kwargs
    assert p.process.kwargs['control_callback'] is p.output_stream
    p.terminate()


# --- Process : result registration --------------------------------------------------

def test_skip_outputs_does_not_register_results():
    """ With `skip_outputs`, outputs never come back : the `AsyncResult` must resolve
        immediately instead of being registered (regression : memory leak + hang) """
    p = Process(
        lambda ** kw: None, name = 'skip-outputs-test',
        input_stream = 'queue', skip_outputs = True
    )

    res = p('some data')
    assert res.ready
    assert res.get(timeout = 1) is None
    assert p._waiting_results == {}

    # the item is still forwarded to the worker queue (`get` : the feeder is async)
    item = p.input_stream.get(timeout = 2)
    assert item.args == ('some data', )


def test_keep_alive_token_resolves_immediately():
    p = Process(lambda ** kw: None, name = 'keep-alive-test', input_stream = 'queue')

    res = p(KEEP_ALIVE)
    assert res.ready
    assert p._waiting_results == {}


def test_regular_data_is_registered():
    p = Process(lambda ** kw: None, name = 'register-test', input_stream = 'queue')

    res = p('some data')
    assert not res.ready
    assert sum(len(waiting) for waiting in p._waiting_results.values()) == 1


# --- Process : stop / clear / terminate ---------------------------------------------

def test_apply_after_stop_raises():
    p = Process(lambda ** kw: None, name = 'stop-test', input_stream = 'queue')
    p.stop()
    with pytest.raises(RuntimeError):
        p('too late')


def test_is_running_false_when_never_started():
    p = Process(lambda ** kw: None, name = 'is-running-test', input_stream = 'queue')
    assert p.is_running() is False


def test_clear_resolves_pending_results_with_none():
    p = Process(lambda ** kw: None, name = 'clear-test', input_stream = 'queue')

    res = p('obsolete')

    def _clear_until_resolved():
        # the `multiprocessing.Queue` feeder is asynchronous : `clear` may observe an
        # empty queue right after `put`, so retry until the result is resolved
        while not res.ready:
            p.clear()
            time.sleep(0.01)
        return res.get(timeout = 1)

    assert run_with_timeout(_clear_until_resolved, timeout = 5) is None
    assert p._waiting_results == {}


def test_terminate_wakes_up_pending_getters(fake_mp_process):
    p = Process(lambda ** kw: None, name = 'terminate-test', input_stream = 'queue').start()

    res = p('never processed')
    p.terminate()

    with pytest.raises(RuntimeError, match = 'terminated'):
        run_with_timeout(res.get, timeout = 5)


def test_runs_function_in_a_thread():
    box = {}

    @run_in_thread
    def worker(x):
        box['value'] = x * 2

    thread = worker(21)
    assert isinstance(thread, threading.Thread)
    thread.join(2)
    assert not thread.is_alive()
    assert box['value'] == 42


def test_runs_off_the_main_thread():
    box = {}

    @run_in_thread
    def worker():
        box['ident'] = threading.get_ident()

    worker().join(2)
    assert box['ident'] != threading.get_ident()


def test_default_thread_name_is_function_name():
    @run_in_thread
    def my_worker():
        pass

    thread = my_worker()
    assert thread.name == 'my_worker'
    thread.join(2)


def test_explicit_thread_name():
    @run_in_thread(name = 'custom-name')
    def worker():
        pass

    thread = worker()
    assert thread.name == 'custom-name'
    thread.join(2)


def test_forwards_thread_kwargs():
    release = threading.Event()

    @run_in_thread(daemon = True)
    def worker():
        release.wait(2)

    thread = worker()
    try:
        assert thread.daemon is True
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(2)


def test_passes_args_and_kwargs():
    box = {}

    @run_in_thread
    def worker(a, b, c = 0):
        box['sum'] = a + b + c

    worker(1, 2, c = 3).join(2)
    assert box['sum'] == 6
