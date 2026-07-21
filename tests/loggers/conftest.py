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

""" pytest fixtures for the `loggers` test-package.

Importing `loggers` mutates the global `logging` state (root handlers replaced by one
strict handler per level, module-level shortcuts patched in). These fixtures make sure
every test starts from — and leaves behind — a clean state :
  - the root logger level and its handlers (levels included) are snapshotted / restored ;
  - the shared `RootTimer` singleton is emptied before and after each test ;
  - `fake_clock` replaces `time.perf_counter` with a manually-advanced clock so run
    durations are deterministic (use binary-exact ticks — 0.5, 0.25, ... — to avoid
    float noise) ;
  - `fresh_logger` provides a per-test, isolated `logging.Logger`.
"""

import time
import logging

import pytest

from loggers import TIME_DEBUG_LEVEL
from loggers.time_logging import _root_timer


@pytest.fixture(autouse = True)
def clean_logging_state():
    root     = logging.getLogger()
    level    = root.level
    handlers = list(root.handlers)
    handler_levels = [h.level for h in handlers]

    _root_timer.reset()
    yield
    _root_timer.reset()

    root.setLevel(level)
    for h in list(root.handlers):
        if h not in handlers: root.removeHandler(h)
    for h, lvl in zip(handlers, handler_levels):
        h.setLevel(lvl)
        if h not in root.handlers: root.addHandler(h)


@pytest.fixture
def time_enabled():
    """ Enables timer tracking for the test (restored by `clean_logging_state`) """
    logging.getLogger().setLevel(TIME_DEBUG_LEVEL)


class FakeClock:
    """ Manually advanced clock replacing `time.perf_counter` (see the `fake_clock` fixture) """
    def __init__(self, start = 1000.):
        self.now = start

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(time, 'perf_counter', clock)
    return clock


@pytest.fixture
def fresh_logger(request):
    logger = logging.getLogger('tests.loggers.{}'.format(request.node.name))
    yield logger
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
