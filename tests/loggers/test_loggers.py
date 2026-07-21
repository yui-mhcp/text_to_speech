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

""" Pure-``pytest`` tests for the :mod:`loggers` package helpers.

Covers the formatter / handler factories (`get_formatter`, `get_handler`, `add_handler`,
`add_file_handler`), the level-based routing (`LevelFilter` modes, `LevelRouter`,
`LevelBasedFormatter`, `setup_loggers`) and the level helpers (`add_level`, `set_level`,
`set_format`) with their error paths. The behaviors driven by import-time environment
variables (`LOG_LEVEL`, `LOGGERS_AUTO_SETUP`) are not tested : they would require
re-importing `loggers`, which re-mutates the global `logging` state.
"""

import logging

import pytest

import loggers
from loggers import (
    LevelFilter, LevelRouter, LevelBasedFormatter, setup_loggers,
    add_level, set_level, set_format,
    get_formatter, get_handler, add_handler, add_file_handler
)


def _record(level, msg = 'msg'):
    return logging.LogRecord('test', level, __file__, 0, msg, None, None)


# --- LevelFilter -------------------------------------------------------------------

def test_level_filter_strict_accepts_exact_level_only():
    f = LevelFilter(logging.INFO)   # default mode : 'strict'
    assert f.filter(_record(logging.INFO))
    assert not f.filter(_record(logging.DEBUG))
    assert not f.filter(_record(logging.WARNING))


def test_level_filter_min_and_max_modes():
    f_min = LevelFilter(logging.INFO, mode = 'min')
    assert f_min.filter(_record(logging.ERROR))
    assert f_min.filter(_record(logging.INFO))
    assert not f_min.filter(_record(logging.DEBUG))

    f_max = LevelFilter(logging.INFO, mode = 'max')
    assert f_max.filter(_record(logging.DEBUG))
    assert f_max.filter(_record(logging.INFO))
    assert not f_max.filter(_record(logging.ERROR))


def test_level_filter_invalid_mode():
    with pytest.raises(ValueError):
        LevelFilter(logging.INFO, mode = 'unknown')


# --- LevelRouter / LevelBasedFormatter ----------------------------------------------

def test_level_router_nearest_routing():
    config = {10 : ('basic', 'stdout'), 30 : ('extended', 'stderr')}
    stdout_router = LevelRouter(config, 'stdout')
    stderr_router = LevelRouter(config, 'stderr')

    # each record is routed to exactly one destination, based on its closest configured level
    assert stdout_router.filter(_record(logging.INFO))       # 20 → nearest is 10 → stdout
    assert not stderr_router.filter(_record(logging.INFO))
    assert stderr_router.filter(_record(logging.ERROR))      # 40 → nearest is 30 → stderr
    assert not stdout_router.filter(_record(logging.ERROR))
    # below every configured level → follows the lowest entry (never lost)
    assert stdout_router.filter(_record(5))
    assert not stderr_router.filter(_record(5))


def test_level_router_honors_levels_added_afterwards():
    config = {10 : ('basic', 'stdout'), 30 : ('extended', 'stderr')}
    stdout_router = LevelRouter(config, 'stdout')
    stderr_router = LevelRouter(config, 'stderr')
    assert stdout_router.filter(_record(logging.INFO))

    # the mapping is shared by reference : a level added later re-routes immediately
    config[20] = ('basic', 'stderr')
    assert stderr_router.filter(_record(logging.INFO))
    assert not stdout_router.filter(_record(logging.INFO))


def test_level_based_formatter_selects_format_per_level():
    config = {10 : ('{levelname} - {message}', 'stdout'), 30 : ('extended', 'stderr')}
    formatter = LevelBasedFormatter(config)

    assert formatter.format(_record(logging.DEBUG)) == 'DEBUG - msg'
    assert formatter.format(_record(logging.INFO))  == 'INFO - msg'   # nearest : 10
    assert '[ERROR]' in formatter.format(_record(logging.ERROR))      # the 'extended' style


def test_setup_loggers_installs_one_handler_per_destination():
    setup_loggers()   # the previous root handlers are restored by `clean_logging_state`

    root = logging.getLogger()
    keys = {entry[1] for entry in loggers._level_to_format.values()}
    routed = [f.key for h in root.handlers for f in h.filters if isinstance(f, LevelRouter)]

    assert sorted(routed) == sorted(keys)
    assert len(root.handlers) == len(keys)
    assert all(isinstance(h.formatter, LevelBasedFormatter) for h in root.handlers)
    assert root.level == loggers._default_level


# --- get_formatter -----------------------------------------------------------------

def test_get_formatter_brace_style():
    fmt = get_formatter('[{levelname}] {message}')
    assert fmt.format(_record(logging.INFO)) == '[INFO] msg'


def test_get_formatter_percent_style():
    fmt = get_formatter('%(levelname)s :: %(message)s')
    assert fmt.format(_record(logging.WARNING)) == 'WARNING :: msg'


def test_get_formatter_named_style():
    fmt = get_formatter('basic')
    assert fmt.format(_record(logging.INFO)) == 'msg'


def test_get_formatter_dict_and_passthrough():
    fmt = get_formatter({'fmt' : '{message}', 'style' : '{'})
    assert fmt.format(_record(logging.INFO)) == 'msg'
    assert get_formatter(fmt) is fmt


def test_get_formatter_invalid():
    with pytest.raises(ValueError):
        get_formatter(42)


# --- get_handler / add_handler -----------------------------------------------------

def test_get_handler_unknown_name():
    with pytest.raises(ValueError) as exc:
        get_handler('does_not_exist')
    # regression : the accepted handlers (not the requested one) are listed as `Accepted`
    assert 'stdout' in str(exc.value)


def test_get_handler_resolves_str_level_and_strict_filter():
    handler = get_handler('stdout', level = 'info', filter_mode = 'strict')
    assert handler.level == logging.INFO
    assert any(
        isinstance(f, LevelFilter) and f.level == logging.INFO for f in handler.filters
    )


def test_get_handler_min_mode_uses_no_filter():
    handler = get_handler('stdout', level = 'info')   # default `filter_mode = 'min'`
    assert handler.level == logging.INFO
    assert handler.filters == []


def test_get_handler_max_mode_keeps_handler_level_open():
    handler = get_handler('stdout', level = 'info', filter_mode = 'max')
    assert handler.level == logging.NOTSET
    assert any(
        isinstance(f, LevelFilter) and f.mode == 'max' and f.level == logging.INFO
        for f in handler.filters
    )


def test_get_handler_invalid_filter_mode():
    with pytest.raises(ValueError):
        get_handler('stdout', level = 'info', filter_mode = 'exact')


def test_get_handler_without_level():
    handler = get_handler('stdout')   # regression : `level = None` used to crash
    assert handler.level == logging.NOTSET
    assert handler.filters == []


def test_add_handler_ignores_none_handler(monkeypatch, fresh_logger):
    monkeypatch.setitem(loggers._handlers, 'broken', lambda * a, ** kw: None)
    # regression : a factory returning `None` (e.g. the `tts` fallback) used to crash
    add_handler('broken', logger = fresh_logger, level = 'info')
    assert fresh_logger.handlers == []


def test_add_handler_lowers_logger_level(fresh_logger):
    fresh_logger.setLevel(logging.WARNING)
    add_handler('stdout', logger = fresh_logger, level = 'info')
    assert len(fresh_logger.handlers) == 1
    assert fresh_logger.level == logging.INFO


def test_add_file_handler(tmp_path, fresh_logger):
    path = tmp_path / 'logs.log'
    # regression : the default `level = None` used to raise a `TypeError`
    add_file_handler(logger = fresh_logger, filename = str(path))
    fresh_logger.warning('hello file')

    for handler in fresh_logger.handlers: handler.close()
    content = path.read_text(encoding = 'utf-8')
    assert 'hello file' in content
    assert '[WARNING]' in content   # the 'extended' format


# --- add_level ---------------------------------------------------------------------

def test_add_level_registers_custom_level(caplog):
    root = logging.getLogger()
    level_before, handlers_before = root.level, list(root.handlers)

    add_level(26, 'notice')
    try:
        assert loggers._levels['notice'] == 26
        assert logging.NOTICE == 26
        assert logging.getLevelName(26) == 'NOTICE'

        # without env vars, no config entry / handler is needed : the level follows the
        # routing of its closest configured level, and the root state is untouched
        assert 26 not in loggers._level_to_format
        assert root.level == level_before
        assert root.handlers == handlers_before

        # both the module-level shortcut and the `Logger` method work
        with caplog.at_level(26):
            logging.getLogger('tests.loggers.custom').notice('custom level message')
        assert 'custom level message' in caplog.text
    finally:
        loggers._levels.pop('notice', None)
        for attr in ('NOTICE', 'notice'):
            if hasattr(logging, attr): delattr(logging, attr)
        if hasattr(logging.Logger, 'notice'): delattr(logging.Logger, 'notice')


def test_add_level_env_overrides(monkeypatch):
    monkeypatch.setenv('AUDIT_HANDLER', 'stderr')
    add_level(21, 'audit')
    try:
        assert loggers._levels['audit'] == 21
        # the env var creates a config entry re-routing the level
        assert loggers._level_to_format[21][1] == 'stderr'
    finally:
        loggers._levels.pop('audit', None)
        loggers._level_to_format.pop(21, None)
        for attr in ('AUDIT', 'audit'):
            if hasattr(logging, attr): delattr(logging, attr)
        if hasattr(logging.Logger, 'audit'): delattr(logging.Logger, 'audit')


# --- set_level / set_format --------------------------------------------------------

def test_set_level_accepts_str_and_int(fresh_logger):
    set_level('warning', fresh_logger)
    assert fresh_logger.level == logging.WARNING
    set_level(logging.DEBUG, fresh_logger)
    assert fresh_logger.level == logging.DEBUG


def test_set_level_accepts_logger_name(fresh_logger):
    set_level('error', fresh_logger.name)
    assert fresh_logger.level == logging.ERROR


def test_set_level_with_handler_kwargs(fresh_logger):
    set_level('info', fresh_logger, handler = 'stream')
    assert fresh_logger.level == logging.INFO
    assert len(fresh_logger.handlers) == 1
    assert fresh_logger.handlers[0].level == logging.INFO


def test_set_format(fresh_logger):
    handler = logging.StreamHandler()
    fresh_logger.addHandler(handler)
    set_format('debug', fresh_logger)
    assert handler.formatter._fmt == loggers._styles['debug']
