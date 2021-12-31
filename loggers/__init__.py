import os
import sys
import logging

from logging.handlers import SMTPHandler

from utils.generic_utils import get_object
from loggers.time_logger import TIME_LEVEL, timer
from loggers.telegram_handler import TelegramHandler

DEV     = 11
logging.addLevelName(DEV, 'DEV')

_styles = {
    'basic' : '{message}',
    'extended'  : '{asctime} : {levelname} : {message}',
    'dev'       : '{asctime} : {levelname} : {module} ({funcName}, {lineno}) : {message}'
}
_levels = {
    'debug' : logging.DEBUG,
    'dev'   : DEV,
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

def set_style(style, logger = None):
    global default_style
    _default_style = style
    
    formatter = get_formatter(style)
    
    for handler in logging.getLogger(logger).handlers:
        handler.setFormatter(formatter)

def set_level(level, logger = None):
    logging.getLogger(logger).setLevel(_levels.get(level, level))

def get_formatter(format = _default_style, style = None, datefmt = None, ** kwargs):
    if isinstance(format, str): format = {'fmt' : _styles.get(format, format)}
    if isinstance(format, dict): format.setdefault('style', '%' if '%' in format['fmt'] else '{')
    
    return format if not isinstance(format, dict) else logging.Formatter(** format)

def add_handler(handler_name, * args, logger = None, level = None,
                add_formatter = True, ** kwargs):
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
