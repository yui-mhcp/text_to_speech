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

""" Pure-``pytest`` tests for :mod:`loggers.time_logging`.

The module tracks execution times as a per-thread tree (`RootTimer`), exposed through
the `@timer` decorator, the `Timer` context-manager and the `start_timer` / `stop_timer`
/ `log_time` shortcuts (also patched onto `logging` / `logging.Logger`). Tracking is
enabled only when the root logger level allows `TIME_LEVEL` (or `TIME_DEBUG_LEVEL`).

Durations are made deterministic by the `fake_clock` fixture (manual `time.perf_counter`).
"""

import logging
import threading

import pytest

from loggers.time_logging import (
    TIME_LEVEL, TIME_DEBUG_LEVEL, Timer, timer,
    start_timer, stop_timer, is_timer_running,
    time_to_string, _root_timer, _timer_to_str
)

pytestmark = pytest.mark.timeout(10)


def _main_tree():
    """ Returns the timer tree of the current (main) thread """
    return _root_timer._timers[threading.get_ident()]


# --- time_to_string ----------------------------------------------------------------

def test_time_to_string_boundaries():
    assert time_to_string(0.0005) == '500 μs'
    assert time_to_string(0.005)  == '5.000 ms'
    assert time_to_string(0.5)    == '500 ms'
    assert time_to_string(2.5)    == '2.500 sec'
    assert time_to_string(65)     == '1min 5sec'
    assert time_to_string(3661)   == '1h 1min 1sec'


# --- tree construction -------------------------------------------------------------

def test_nested_timers_build_tree(time_enabled, fake_clock):
    with Timer('outer'):
        fake_clock.tick(1.)
        with Timer('inner'):
            fake_clock.tick(0.5)

    tree = _main_tree()
    assert list(tree) == ['outer']
    outer = tree['outer']
    assert outer['runs'] == [1.5]
    assert list(outer['children']) == ['inner']
    assert outer['children']['inner']['runs'] == [0.5]


def test_disabled_level_is_a_noop():
    logging.getLogger().setLevel(logging.INFO)   # INFO (20) > TIME_LEVEL (15)

    @timer
    def foo():
        return 42

    assert foo() == 42
    assert not _root_timer._timers
    assert not is_timer_running()


def test_debug_timers_require_time_debug_level(time_enabled):
    # TIME_LEVEL (15) enabled but not TIME_DEBUG_LEVEL (13)
    logging.getLogger().setLevel(TIME_LEVEL)

    with Timer('regular') as regular:
        with Timer('detail', debug = True) as detail:
            pass

    assert regular._timer is not None
    assert detail._timer is None
    assert 'children' not in _main_tree()['regular']


# --- the `timer` decorator ---------------------------------------------------------

def test_timer_decorator_bare_logs_and_resets(time_enabled, caplog):
    @timer
    def foo(x):
        return x * 2

    with caplog.at_level(TIME_LEVEL):
        assert foo(3) == 6

    # the root timer logs its report... and resets
    assert 'foo' in caplog.text
    assert not _root_timer._timers


def test_timer_decorator_named_and_nested(time_enabled, fake_clock):
    @timer(name = 'leaf', log_if_root = False)
    def inner():
        fake_clock.tick(0.25)

    @timer(log_if_root = False)
    def outer():
        inner()
        inner()

    outer()

    tree = _main_tree()
    assert list(tree) == ['outer']
    leaf = tree['outer']['children']['leaf']
    assert leaf['runs'] == [0.25, 0.25]


def test_timer_decorator_stops_on_exception(time_enabled):
    @timer(log_if_root = False)
    def boom():
        raise ValueError('boom')

    with pytest.raises(ValueError):
        boom()

    assert not is_timer_running()
    assert len(_main_tree()['boom']['runs']) == 1


def test_disable_timers_returns_undecorated_function(monkeypatch):
    # `_DISABLE_TIMERS` is read from `LOGGERS_DISABLE_TIMERS` at import time : the module
    # attribute is patched directly here
    from loggers import time_logging
    monkeypatch.setattr(time_logging, '_DISABLE_TIMERS', True)

    def foo():
        return 42

    assert timer(foo) is foo
    assert timer(name = 'foo', debug = True)(foo) is foo


def test_multiple_runs_are_aggregated(time_enabled, fake_clock):
    @timer(log_if_root = False)
    def foo():
        fake_clock.tick(0.125)

    for _ in range(3): foo()

    node = _main_tree()['foo']
    assert node['runs'] == [0.125] * 3
    assert 'executed 3 times' in _timer_to_str(node)


# --- start / stop errors -----------------------------------------------------------

def test_stop_without_running_timer_is_tolerated(time_enabled):
    # a concurrent `log_time` (from another thread) may reset the timers between a
    # `start` and its `stop` : stopping with nothing running must not raise
    assert stop_timer('never_started') is None


def test_stop_after_concurrent_reset_is_tolerated(time_enabled):
    @timer
    def foo():
        _root_timer.reset()   # simulates a concurrent `log_time` from another thread
        return 42

    assert foo() == 42        # regression : the final `stop_timer` used to raise
    assert not is_timer_running()


def test_stop_wrong_name_raises_without_corrupting_state(time_enabled):
    start_timer('a', TIME_LEVEL)
    with pytest.raises(RuntimeError):
        stop_timer('b')

    # the mismatched stop left the running timer untouched : it can still be stopped
    assert is_timer_running()
    stop_timer('a')
    assert len(_main_tree()['a']['runs']) == 1


# --- the `Timer` class -------------------------------------------------------------

def test_timers_property_without_children(time_enabled, fake_clock):
    with Timer('solo') as t:
        fake_clock.tick(0.25)

    assert t.timers == {'solo' : [0.25]}


def test_timers_property_with_children(time_enabled, fake_clock):
    with Timer('parent') as t:
        with Timer('child'):
            fake_clock.tick(0.5)

    assert t.timers == {'parent' : [0.5], 'child' : [0.5]}


def test_timer_save(time_enabled, fake_clock, tmp_path):
    import json

    with Timer('saved') as t:
        fake_clock.tick(0.5)

    path = tmp_path / 'timers.json'
    t.save(str(path))
    assert json.loads(path.read_text()) == {'saved' : [0.5]}

    # `overwrite = False` (default) must not touch an existing file
    path.write_text('{}')
    t.save(str(path))
    assert path.read_text() == '{}'

    t.save(str(path), overwrite = True)
    assert json.loads(path.read_text()) == {'saved' : [0.5]}


# --- rendering / reset -------------------------------------------------------------

def test_str_renders_and_resets(time_enabled, fake_clock):
    with Timer('tracked'):
        fake_clock.tick(0.5)

    des = str(_root_timer)
    assert 'tracked' in des
    assert '500 ms' in des
    # displaying the report resets the timers
    assert not _root_timer._timers


def test_str_while_running_in_another_thread(time_enabled):
    started, release = threading.Event(), threading.Event()

    def worker():
        start_timer('worker_task', TIME_LEVEL)
        started.set()
        release.wait()
        stop_timer('worker_task')

    th = threading.Thread(target = worker)
    th.start()
    started.wait()
    try:
        des = str(_root_timer)   # regression : used to crash on the missing `self.name`
        assert 'running' in des
        assert 'worker_task' in des
        assert _root_timer._timers   # not reset while a timer is running
    finally:
        release.set()
        th.join()


def test_threads_have_separate_trees(time_enabled):
    def worker(tag):
        with Timer(tag):
            pass

    threads = [
        threading.Thread(target = worker, args = ('task_{}'.format(i),), name = 'Worker-{}'.format(i))
        for i in range(2)
    ]
    for th in threads: th.start()
    for th in threads: th.join()

    trees = _root_timer._timers
    assert len(trees) == 2
    assert sorted(name for tree in trees.values() for name in tree) == ['task_0', 'task_1']

    des = str(_root_timer)
    assert 'Worker-0' in des and 'Worker-1' in des


# --- `logging` shortcuts -----------------------------------------------------------

def test_logging_log_time_is_callable(time_enabled, caplog, fake_clock):
    with Timer('tracked'):
        fake_clock.tick(0.5)

    with caplog.at_level(TIME_LEVEL):
        logging.log_time()   # regression : used to raise `TypeError` (descriptor re-binding)
    assert 'tracked' in caplog.text


def test_logger_shortcuts(time_enabled, caplog):
    lg = logging.getLogger('tests.loggers.shortcuts')

    with lg.timer('via_logger'):
        pass
    assert 'via_logger' in _main_tree()

    with caplog.at_level(TIME_LEVEL):
        lg.log_time()   # regression : used to receive the logger instance as `level`
    assert 'via_logger' in caplog.text
