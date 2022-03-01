
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
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

from logging.handlers import SMTPHandler

try:
    from utils.generic_utils import get_object
except ImportError as e:
    from loggers.utils import get_object
from loggers.time_logger import TIME_LEVEL, timer
from loggers.telegram_handler import TelegramHandler

DEV     = 11

_styles = {
    'basic' : '{message}',
    'extended'  : '{asctime} : {levelname} : {message}',
    'dev'       : '{asctime} : {levelname} : {module} ({funcName}, {lineno}) : {message}'
}
_levels = {
    'debug' : logging.DEBUG,
    'time'  : TIME_LEVEL,
    'info'  : logging.INFO,
    'warning'   : logging.WARNING,
    'error'     : logging.ERROR,
    'critical'  : logging.CRITICAL
}

_default_style  = os.environ.get('LOG_STYLE', 'basic').lower()
_default_level  = os.environ.get('LOG_LEVEL', 'info').lower()
_default_format = _styles.get(_default_style, _default_style)

logging.basicConfig(
    level = _levels.get(_default_level, _default_level), stream = sys.stdout,
    format = _default_format, style = '%' if '%' in _default_format else '{'
)

def add_level(value, name):
    def _special_log(* args, ** kwargs):
        return logging.log(value, * args, ** kwargs)
    
    global _levels
    _levels[name.lower()] = value

    logging.addLevelName(value, name.upper())
    setattr(logging, name.lower(), _special_log)

def set_style(style, logger = None):
    global _default_style
    _default_style = style
    
    formatter = get_formatter(style)
    
    for handler in logging.getLogger(logger).handlers:
        handler.setFormatter(formatter)

def set_level(level, logger = None):
    global _levels
    if isinstance(level, str): level = level.lower()
    logging.getLogger(logger).setLevel(_levels.get(level, level))

def get_formatter(format = _default_style, style = None, datefmt = None, ** kwargs):
    if isinstance(format, str): format = {'fmt' : _styles.get(format, format)}
    if isinstance(format, dict): format.setdefault('style', '%' if '%' in format['fmt'] else '{')
    
    return format if not isinstance(format, dict) else logging.Formatter(** format)

def add_handler(handler_name, * args, logger = None, level = None,
                add_formatter = True, ** kwargs):
    global _default_style, _levels
    if logger is None: logger = logging.getLogger()
    elif isinstance(logger, str): logger = logging.getLogger(logger)
    
    if isinstance(level, str): level = _levels[level]
    
    fmt = kwargs.pop('format', _default_style)
    
    handler = get_object(_handlers, handler_name, * args, ** kwargs) if isinstance(handler_name, str) else handler_name
    if isinstance(handler, str) or handler is None: return
    if level is not None: handler.setLevel(level)
    
    if add_formatter and fmt is not None:
        formatter = get_formatter(fmt)
        
        handler.setFormatter(formatter)
    
    if level is not None and logger.level > handler.level: logger.setLevel(level)

    logger.addHandler(handler)
    return handler

def add_basic_handler(format = 'basic', ** kwargs):
    return add_handler('stream', sys.stdout, format = format, ** kwargs)

def add_file_handler(filename = 'logs.log', encoding = 'utf-8', format = 'extended', ** kwargs):
    return add_handler('file', filename = filename, encoding = encoding, format = format, ** kwargs)


def try_tts_handler(* args, ** kwargs):
    try:
        from loggers.tts_handler import TTSHandler
        return TTSHandler(* args, ** kwargs)
    except ImportError as e:
        logging.error("Error when adding TTSHandler : {}".format(e))
        return None

_handlers   = {
    'stream'    : logging.StreamHandler,
    'file'      : logging.FileHandler,
    'smtp'      : SMTPHandler,
    'tts'       : try_tts_handler,
    'telegram'  : TelegramHandler
}

add_level(DEV, 'DEV')
