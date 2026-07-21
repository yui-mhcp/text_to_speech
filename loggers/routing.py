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

""" Level-based routing / formatting primitives for the `loggers` package.

This module answers, for a given record, "where should it go and how should it look".
It is stateless and only depends on the stdlib `logging`, so it can be used (and tested)
in isolation. The shared vocabulary is a `config` mapping `{level : (format, handler_key)}`
(see `LevelRouter`), owned by the package `__init__` (`_level_to_format`).
"""

import logging

_styles = {
    'basic' : '{message}',
    'time'  : '[{asctime}] {message}',
    'extended'  : '[{asctime}] [{levelname}] {message}',
    'debug'     : '[{asctime}] [{levelname}] [{module} {funcName}:{lineno}] {message}'
}

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

class LevelFilter(logging.Filter):
    """
        Filter that restricts the accepted records based on their level

        `mode` controls the comparison with `level` :
            - 'min'    : accepts records with `levelno >= level` (standard `logging` behavior)
            - 'max'    : accepts records with `levelno <= level`
            - 'strict' : only accepts records with `levelno == level`
    """
    def __init__(self, level, mode = 'strict'):
        super().__init__()
        if mode not in ('min', 'max', 'strict'):
            raise ValueError('Unknown `mode` !\n  Accepted : (min, max, strict)\n  Got : {}'.format(mode))

        self.level = level
        self.mode  = mode

    def filter(self, record):
        if self.mode == 'min':      return record.levelno >= self.level
        elif self.mode == 'max':    return record.levelno <= self.level
        return record.levelno == self.level

def _nearest_config(config, levelno):
    """ Returns the entry of the highest configured level `<= levelno` (or the lowest configured one) """
    nearest = None
    for level in sorted(config):
        if nearest is None or level <= levelno: nearest = level
        else: break
    return config[nearest] if nearest is not None else None

class LevelRouter(logging.Filter):
    """
        Filter that routes each record to a single destination based on its level

        `config` is a `{level : (format, handler_key)}` mapping, kept by **reference** :
        levels added afterwards (e.g., by `add_level`) are automatically honored. A record
        is accepted iff the entry of its closest configured level (see `_nearest_config`)
        points to `key`. Records with an unknown level are therefore never lost : they
        follow the routing of their closest configured level.

        Note : the per-level decision is cached, and invalidated when `len(config)` changes,
        meaning that entries should only be added to `config`, not modified in-place.
    """
    def __init__(self, config, key):
        super().__init__()
        self.config = config
        self.key    = key

        self._cache = {}
        self._n     = -1

    def filter(self, record):
        if self._n != len(self.config):
            self._cache, self._n = {}, len(self.config)

        accept = self._cache.get(record.levelno)
        if accept is None:
            entry  = _nearest_config(self.config, record.levelno)
            accept = self._cache[record.levelno] = (entry is not None and entry[1] == self.key)
        return accept

class LevelBasedFormatter(logging.Formatter):
    """
        Formatter that selects its format based on the record level

        `config` is the same `{level : (format, handler_key)}` mapping as `LevelRouter`
        (also kept by reference, with the same caching strategy). Records with an unknown
        level use the format of their closest configured level, and `default` if `config`
        is empty.
    """
    def __init__(self, config, default = 'basic'):
        super().__init__()
        self.config  = config
        self.default = default

        self._cache = {}
        self._n     = -1

    def format(self, record):
        if self._n != len(self.config):
            self._cache, self._n = {}, len(self.config)

        formatter = self._cache.get(record.levelno)
        if formatter is None:
            entry     = _nearest_config(self.config, record.levelno)
            formatter = self._cache[record.levelno] = get_formatter(
                entry[0] if entry is not None else self.default
            )
        return formatter.format(record)
