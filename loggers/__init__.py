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

from .time_logging import TIME_LEVEL, TIME_DEBUG_LEVEL, Timer, timer, time_logger
from .telegram_handler import TelegramHandler

logger  = logging.getLogger(__name__)

DEV_LEVEL   = 11
RETRACING_LEVEL = 18

_styles = {
    'basic' : '{message}',
    'time'  : '[{asctime}] {message}',
    'extended'  : '[{asctime}] [{levelname}] {message}',
    'debug'     : '[{asctime}] [{levelname}] [{module} {funcName}:{lineno}] {message}'
}
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

_default_style  = os.environ.get('LOG_STYLE', 'basic').lower()
_default_level  = os.environ.get('LOG_LEVEL', 'info').lower()
_default_format = _styles.get(_default_style, _default_style)

_default_level  = int(_default_level) if _default_level.isdigit() else _levels[_default_level]
logging.basicConfig(
    level   = _default_level,
    stream  = sys.stdout,
    format  = _default_format,
    style   = '%' if '%' in _default_format else '{'
)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.setLevel(_default_level)
del handler

class LevelFilter(logging.Filter):
    """Filtre qui accepte uniquement un niveau de log spécifique"""
    def __init__(self, level):
        super().__init__()
        self.level = level
    
    def filter(self, record):
        return record.levelno == self.level

def add_level(value, name):
    """
        Adds a new level to the logging module
        
        Arguments :
            - value : the log level value (e.g., logging.DEBUG = 10, logging.INFO = 20, ...)
            - name  : the level name
        
        Example :
        ```python
        # add a 'dev' level just above the debug level
        add_level('dev', 11)
        # Now it is possible to set the level with the `set_level` method
        set_level('dev')
        # log a message with the new `.dev` method
        logging.dev('This is a test !')
        # logging.getLogger(__name__).dev('This will also work !')
        ```
    """
    name = name.lower()
    if name not in _levels:
        set_level.dispatch(name, value)

    logging.addLevelName(value, name.upper())
    if not hasattr(logging, name.upper()):
        setattr(logging, name.upper(), value)
    if not hasattr(logging, name):
        setattr(logging, name, partial(logging.log, value))
    if not hasattr(logging.Logger, name):
        setattr(logging.Logger, name, partialmethod(logging.Logger.log, value))

def set_level(level, logger = None, ** kwargs):
    """ Sets the `logger` level to `level` """
    if not kwargs:
        if isinstance(level, str):  level = _levels[level.lower()]
        if isinstance(logger, str) or logger is None: logger = logging.getLogger(logger)
        
        logger.setLevel(level)
        if logger.handlers:
            for handler in logger.handlers: handler.setLevel(level)
        elif level < root_logger.level:
            for handler in root_logger.handlers: handler.setLevel(level)
    else:
        if 'handler' not in kwargs: kwargs['handler'] = 'stdout'
        add_handler(level = level, logger = logger, ** kwargs)

def set_format(format, logger = None):
    """ Sets the logging style to `logger` (root logger if None) """
    formatter = get_formatter(format)
    
    if isinstance(logger, str) or logger is None: logger = logging.getLogger(logger)
    for handler in logger.handlers:
        handler.setFormatter(formatter)


def get_formatter(format, ** kwargs):
    if isinstance(format, str):
        if format in _styles: format = _styles[format]
        return logging.Formatter(format, style = '%' if '%' in format else '{')
    elif isinstance(format, dict):
        return logging.Formatter(** format)
    elif isinstance(format, logging.Formatter):
        return format
    else:
        raise ValueError('Unknown format : {}'.format(format))

def add_handler(handler, * args, logger = None, level = None, add_formatter = True, ** kwargs):
    global _default_style
    
    if isinstance(logger, str) or logger is None: logger = logging.getLogger(logger)
    if isinstance(level, str): level = _levels[level.lower()]
    
    fmt = kwargs.pop('format', _default_style)
    
    if isinstance(handler, str):
        if handler not in _handlers:
            raise ValueError('Unknown handler !\n  Accepted : {}\n  Got : {}'.format(
                handler, tuple(_handlers.keys())
            ))
        
        handler = _handlers[handler](* args, ** kwargs)
        if handler is None: return
    
    if level is not None:
        if level < logger.level or logger.level == 0:
            original_level = logger.level
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(original_level)
        
        handler.setLevel(level)
        handler.addFilter(LevelFilter(level))
    
    if add_formatter and fmt:
        handler.setFormatter(get_formatter(fmt))

    logger.addHandler(handler)
    return handler

add_basic_handler   = partial(
    add_handler, 'stream', sys.stdout, format = 'basic'
)
add_file_handler    = partial(
    add_handler, 'file', filename = 'logs.log', encoding = 'utf-8', format = 'extended'
)

def try_tts_handler(* args, ** kwargs):
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
    'tts'       : try_tts_handler
}

for name, val in _levels.items(): add_level(val, name)
