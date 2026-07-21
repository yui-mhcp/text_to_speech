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
import sys
import logging
import logging.handlers

from functools import partial, partialmethod

from .routing import (
    _styles, get_formatter, _nearest_config, LevelFilter, LevelRouter, LevelBasedFormatter
)
from .time_logging import TIME_LEVEL, TIME_DEBUG_LEVEL, Timer, timer, time_logger
from .telegram_handler import TelegramHandler

logger  = logging.getLogger(__name__)

DEV_LEVEL   = 11
RETRACING_LEVEL = 18

_levels = {
    'debug' : logging.DEBUG,
    'dev'   : DEV_LEVEL,
    'time_debug'    : TIME_DEBUG_LEVEL,
    'time'  : TIME_LEVEL,
    'retracing' : RETRACING_LEVEL,
    'info'      : logging.INFO,
    'warning'   : logging.WARNING,
    'error'     : logging.ERROR,
    'critical'  : logging.CRITICAL
}

_default_level  = os.environ.get('LOG_LEVEL', 'info').lower()
if _default_level.isdigit():
    _default_level = int(_default_level)
elif _default_level in _levels:
    _default_level = _levels[_default_level]
else:
    logger.warning('Unknown `LOG_LEVEL` `{}` : defaulting to `info`'.format(_default_level))
    _default_level = logging.INFO
_default_format = os.environ.get('LOG_FORMAT', 'basic').lower()
_default_format = _styles.get(_default_format, _default_format)
_default_stream = 'stdout'

_level_to_format    = {
    logging.DEBUG   : ('extended', 'stdout'),
    logging.WARNING : ('extended', 'stderr'),
    logging.ERROR   : ('extended', 'stderr'),
    logging.CRITICAL    : ('extended', 'stderr')
}
for name, level in _levels.items():
    if level in _level_to_format:
        default_format, default_stream = _level_to_format[level]
    else:
        default_format, default_stream = _default_format, _default_stream
    
    _level_to_format[level] = (
        os.environ.get('{}_FORMAT'.format(name.upper()), default_format),
        os.environ.get('{}_HANDLER'.format(name.upper()), default_stream)
    )
del name, level, default_format, default_stream

def _try_tts_handler(* args, ** kwargs):
    try:
        from .tts_handler import TTSHandler
        return TTSHandler(* args, ** kwargs)
    except ImportError as e:
        logger.error("An error occured while initializing `TTSHandler` : {}".format(e))
        return None

_handlers   = {
    'stdout'    : partial(logging.StreamHandler, sys.stdout),
    'stderr'    : partial(logging.StreamHandler, sys.stderr),
    'stream'    : logging.StreamHandler,
    'file'      : logging.FileHandler,
    'smtp'      : logging.handlers.SMTPHandler,
    'telegram'  : TelegramHandler,
    'tts'       : _try_tts_handler
}

root_logger = logging.getLogger()

def add_level(value, name):
    """
        Adds a new level to the logging module
        
        Arguments :
            - value : the log level value (e.g., logging.DEBUG = 10, logging.INFO = 20, ...)
            - name  : the level name
        
        Example :
        ```python
        # add a 'dev' level just above the debug level
        add_level(11, 'dev')
        # Now it is possible to set the level with the `set_level` method
        set_level('dev')
        # log a message with the new `.dev` method
        logging.dev('This is a test !')
        # logging.getLogger(__name__).dev('This will also work !')
        ```

        Note : without `{NAME}_FORMAT` / `{NAME}_HANDLER` env variables, no configuration
        is needed : the new level follows the routing / format of its closest configured
        level (see `LevelRouter`)
    """
    name = name.lower()
    if name not in _levels:
        _levels[name] = value

        fmt    = os.environ.get('{}_FORMAT'.format(name.upper()))
        stream = os.environ.get('{}_HANDLER'.format(name.upper()))
        if fmt or stream:
            nearest = _nearest_config(_level_to_format, value)
            if not fmt:     fmt    = nearest[0] if nearest else _default_format
            if not stream:  stream = nearest[1] if nearest else _default_stream
            _level_to_format[value] = (fmt, stream)

            routed = {
                f.key for h in root_logger.handlers
                for f in h.filters if isinstance(f, LevelRouter)
            }
            if routed and stream not in routed:
                _add_routed_handler(stream)

    logging.addLevelName(value, name.upper())
    if not hasattr(logging, name.upper()):
        setattr(logging, name.upper(), value)
    if not hasattr(logging, name):
        setattr(logging, name, partial(logging.log, value))
    if not hasattr(logging.Logger, name):
        setattr(logging.Logger, name, partialmethod(logging.Logger.log, value))

def set_level(level, logger = None, ** kwargs):
    """ Sets the `logger` level to `level` """
    if isinstance(level, str):  level = _levels[level.lower()]
    if isinstance(logger, str) or logger is None: logger = logging.getLogger(logger)

    logger.setLevel(level)

    if kwargs:
        if 'handler' not in kwargs: kwargs['handler'] = _default_stream
        add_handler(level = level, logger = logger, ** kwargs)
    elif logger is not root_logger:
        # the root handlers are level-agnostic (`LevelRouter`) : the root level is the only gate
        for handler in logger.handlers: handler.setLevel(level)

def set_format(format, logger = None):
    """ Sets the logging style to `logger` (root logger if None) """
    formatter = get_formatter(format)
    
    if isinstance(logger, str) or logger is None: logger = logging.getLogger(logger)
    for handler in logger.handlers:
        handler.setFormatter(formatter)

def get_handler(handler, * args, level = None, filter_mode = 'min', ** kwargs):
    """
        Builds (if needed) and configures a `logging.Handler`

        Arguments :
            - handler : a `logging.Handler`, or a name from `_handlers` (then built with `args` / `kwargs`)
            - level   : the handler level (name or value)
            - filter_mode : how `level` restricts the emitted records :
                - 'min'    : records with `levelno >= level` (standard `logging` behavior)
                - 'max'    : records with `levelno <= level`
                - 'strict' : only records with `levelno == level`
            - kwargs  : may contain `format` (see `get_formatter`), the rest is forwarded
                        to the handler constructor

        Note : it is named `filter_mode` (not `mode`) to avoid collisions with handler
        constructor kwargs (e.g., the `mode` of `logging.FileHandler`)
    """
    if filter_mode not in ('min', 'max', 'strict'):
        raise ValueError('Unknown `filter_mode` !\n  Accepted : (min, max, strict)\n  Got : {}'.format(
            filter_mode
        ))

    format = kwargs.pop('format', _default_format)
    if isinstance(level, str): level = _levels[level.lower()]

    if isinstance(handler, str):
        if handler not in _handlers:
            raise ValueError('Unknown handler !\n  Accepted : {}\n  Got : {}'.format(
                tuple(_handlers.keys()), handler
            ))

        handler = _handlers[handler](* args, ** kwargs)
        if handler is None: return None

    if level is not None:
        if filter_mode != 'max': handler.setLevel(level)
        if filter_mode != 'min': handler.addFilter(LevelFilter(level, mode = filter_mode))
    handler.setFormatter(get_formatter(format))

    return handler

def add_handler(handler, * args, logger = None, ** kwargs):
    if logger is None or isinstance(logger, str): logger = logging.getLogger(logger)

    handler = get_handler(handler, * args, ** kwargs)
    if handler is None: return

    logger.addHandler(handler)
    if handler.level > 0 and (logger.level == 0 or handler.level < logger.level):
        logger.setLevel(handler.level)
    
add_file_handler    = partial(
    add_handler, 'file', filename = 'logs.log', encoding = 'utf-8', format = 'extended'
)

def _add_routed_handler(key, config = None):
    """
        Adds a *routed* handler to the root logger for the destination `key` (e.g., 'stdout')

        A routed handler is passive and level-agnostic (level `NOTSET`) : every decision is
        taken per-record by two objects sharing `config` (`{level : (format, handler_key)}`) :
            - `LevelRouter` : accepts the record iff the entry of its closest configured
              level points to `key` — the routing is *exclusive*, each record is emitted
              by exactly one destination
            - `LevelBasedFormatter` : formats the record with the format of that same entry

        `config` is stored by **reference** (not copied) : it is the single source of truth
        shared by all the routed handlers, so levels added afterwards (see `add_level`) are
        honored immediately, without rebuilding any handler.

        This differs from `add_handler`, which is *additive* and *static* : it attaches one
        extra handler whose level / filter / format are fixed at creation time (e.g., a
        strict `TelegramHandler` emitting on top of the console). This function is only
        meaningful as a building block of `setup_loggers` (and of `add_level`, when an env
        variable routes a new level to a not-yet-routed destination).
    """
    if config is None: config = _level_to_format

    handler = get_handler(key)
    if handler is None: return

    handler.setFormatter(LevelBasedFormatter(config, default = _default_format))
    handler.addFilter(LevelRouter(config, key))
    root_logger.addHandler(handler)

def setup_loggers(level = None, config = None):
    """
        (Re)configures the root logger : its current handlers are removed, then one handler
        per destination in `config` (`{level : (format, handler_key)}`) is installed, with
        level-based routing (`LevelRouter`) and formatting (`LevelBasedFormatter`)

        The routed handlers are level-agnostic : the root logger level is the only gate,
        which makes `set_level` a simple `logger.setLevel`

        This function is called at import time, unless the `LOGGERS_AUTO_SETUP` env
        variable is set to false
    """
    if level is None:   level = _default_level
    if config is None:  config = _level_to_format

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    for key in dict.fromkeys(entry[1] for entry in config.values()):
        _add_routed_handler(key, config)

    root_logger.setLevel(level)

for name, val in _levels.items():
    add_level(val, name)
del name, val

if os.environ.get('LOGGERS_AUTO_SETUP', 'true').lower() not in ('0', 'false'):
    setup_loggers()
