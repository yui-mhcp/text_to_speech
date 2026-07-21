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

""" Pure-``pytest`` tests for :mod:`utils.generic_utils`.

Written as plain module-level functions + ``@pytest.mark.parametrize`` (no
``unittest.TestCase`` / ``absl.parameterized``). Pure helpers use plain
``assert``; tolerant numpy comparisons go through the ``asserts`` fixture
(``tests.asserts``).

Covers, following the mirror-tree convention (each helper is tested next to the
module it lives in, ``utils/generic_utils.py``) :
  - data conversion : ``to_json``, ``convert_to_str``, ``time_to_string`` ;
  - iteration       : ``create_iterable`` (moved here from ``test_stream``) ;
  - ``inspect`` utilities : ``get_fn_name`` / ``get_args`` / ``get_kwargs`` /
    ``get_annotations`` / ``has_args`` / ``has_kwargs`` / ``signature_to_str`` /
    ``get_enum_item``.

Out of scope for now (environment-dependent or pandas-heavy) : ``timestamp_to_str``,
``get_module_version``, ``import_submodules``, ``set_display_options`` and the
DataFrame helpers (``filter_df`` / ``sample_df`` / ``aggregate_df``).
"""

import enum
import json
import uuid
import queue
import logging
import argparse
import datetime
import inspect
import multiprocessing
import numpy as np
import pandas as pd
import pytest

from functools import partial
from dataclasses import dataclass

from utils.generic_utils import (
    to_json, convert_to_str, time_to_string, create_iterable,
    get_fn_name, get_args, get_kwargs, get_annotations,
    has_args, has_kwargs, signature_to_str, get_enum_item,
)
from utils.file_utils import dump_data  # used as a `function` sample for to_json


# --- shared samples ----------------------------------------------------------------

@dataclass
class User:
    name        : str
    age         : int
    metadata    : dict

class _WithConfig:
    """ Mimics a keras object : serialized through its `get_config()`. """
    def get_config(self):
        return {'units' : 4, 'name' : 'layer'}

class _Unknown:
    """ Matches no `to_json` case -> falls back to `str(...)`. """
    def __str__(self):
        return 'unknown-object'

class _Named:
    name = 'custom_name'

class _Anonymous:
    pass

class _Color(enum.Enum):
    RED   = 1
    GREEN = 2

def _sample_fn(a, b, c = 3, * args, d = 4, ** kwargs):
    pass

def _no_varargs(a, b = 2):
    pass

def _annotated(a : int, b : str = 'x') -> bool:
    return True

def _generator():
    for i in range(1, 5):
        yield i

def _gen_kwargs(n = 0):
    yield from range(n)

# fixed UUID so its expected serialization can be precomputed at collection time
_UUID = uuid.uuid4()


# --- to_json -----------------------------------------------------------------------

@pytest.mark.parametrize('data,target', [
    # native JSON-compatible types are returned unchanged (target=None -> data)
    pytest.param(None, None, id = 'none'),
    pytest.param(True, None, id = 'bool'),
    pytest.param(2, None, id = 'int'),
    pytest.param(10., None, id = 'float'),
    pytest.param('Hello World !', None, id = 'str'),
    pytest.param(__file__, __file__.replace('\\', '/'), id = 'file'),

    pytest.param(b'hello', 'hello', id = 'bytes'),

    pytest.param([1, 2., 'Hello World !', True], None, id = 'list'),
    pytest.param((1, 2), [1, 2], id = 'tuple'),
    pytest.param({1}, [1], id = 'set'),
    pytest.param({'a' : 1, 'b' : 2., 'c' : True}, None, id = 'dict'),
    pytest.param([1, {'b' : 2., 'c' : [3, 'Hello World !']}], None, id = 'nested'),

    # numpy scalars collapse to native Python scalars
    pytest.param(np.array(True), True, id = 'np_bool_single'),
    pytest.param(np.array(1, dtype = 'uint8'), 1, id = 'np_uint8_single'),
    pytest.param(np.array(1, dtype = 'int32'), 1, id = 'np_int32_single'),
    pytest.param(np.array(1, dtype = 'int64'), 1, id = 'np_int64_single'),
    pytest.param(np.array(1, dtype = 'float32'), 1., id = 'np_float32_single'),
    pytest.param(np.array(1, dtype = 'float64'), 1., id = 'np_float64_single'),

    # numpy arrays collapse to native Python lists
    pytest.param(np.array([True, False]), [True, False], id = 'np_bool'),
    pytest.param(np.array([1, 2], dtype = 'uint8'), [1, 2], id = 'np_uint8'),
    pytest.param(np.array([1, 2], dtype = 'int32'), [1, 2], id = 'np_int32'),
    pytest.param(np.array([1, 2], dtype = 'int64'), [1, 2], id = 'np_int64'),
    pytest.param(np.array([1, 2], dtype = 'float32'), [1., 2.], id = 'np_float32'),
    pytest.param(np.array([1, 2], dtype = 'float64'), [1., 2.], id = 'np_float64'),

    pytest.param(_UUID, str(_UUID), id = 'uuid'),
    pytest.param(datetime.datetime(2026, 6, 28, 12, 30, 0), '2026-06-28 12:30:00', id = 'datetime'),
    pytest.param(argparse.Namespace(a = 1, b = 'x'), {'a' : 1, 'b' : 'x'}, id = 'namespace'),
    pytest.param(User('test', 16, {'a' : 1}), {'name' : 'test', 'age' : 16, 'metadata' : {'a' : 1}}, id = 'dataclass'),
    pytest.param(_WithConfig(), {'units' : 4, 'name' : 'layer'}, id = 'get_config'),
    pytest.param(_Unknown(), 'unknown-object', id = 'unknown'),
    pytest.param(dump_data, 'utils.file_utils.dump_data', id = 'function'),
])
def test_to_json(data, target):
    if target is None and data is not None:
        target = data

    result = to_json(data)
    assert result == target

    # the result must be JSON-serializable
    try:
        json.dumps(result)
    except TypeError as e:
        pytest.fail('to_json result {!r} (type {}) is not JSON-serializable : {}'.format(
            result, type(result), e
        ))


# --- convert_to_str ----------------------------------------------------------------

@pytest.mark.parametrize('value,target', [
    pytest.param(None, None, id = 'none'),
    pytest.param('hello', 'hello', id = 'str'),
    pytest.param(b'hello', 'hello', id = 'bytes'),
    pytest.param([b'a', 'b'], ['a', 'b'], id = 'list'),
    pytest.param((b'a', b'b'), ['a', 'b'], id = 'tuple'),       # tuple -> list
    pytest.param({b'a'}, ['a'], id = 'set'),                    # set   -> list
    pytest.param({'x' : b'y'}, {'x' : 'y'}, id = 'dict'),       # only values are converted
    pytest.param(np.array(['a', 'b']), ['a', 'b'], id = 'str_array'),
    pytest.param(np.array([b'a', b'b']), ['a', 'b'], id = 'bytes_array'),
    # regression : a `bytes` array (dtype 'S') with UTF-8 content used to hit
    # `astype(str)` -> ASCII decode and raise on the 0xc3 byte of `à`
    pytest.param(np.array([b'Bonjour \xc3\xa0 tous']), ['Bonjour à tous'], id = 'utf8_bytes_array'),
    pytest.param(np.array(b'caf\xc3\xa9'), 'café', id = 'utf8_bytes_scalar'),   # 0-D array
])
def test_convert_to_str(value, target):
    assert convert_to_str(value) == target


def test_convert_to_str_numeric_array_warns(caplog, asserts):
    """ A numerical array cannot be stringified : it is returned unchanged + a warning. """
    arr = np.array([1, 2, 3])
    with caplog.at_level(logging.WARNING):
        result = convert_to_str(arr)

    asserts.assert_array(result)
    asserts.assert_equal(arr, result)
    assert any('not been converted' in record.message for record in caplog.records)


# --- time_to_string ----------------------------------------------------------------

@pytest.mark.parametrize('seconds,target', [
    pytest.param(0.0005, '500 µs', id = 'microseconds'),  # µ = micro-sign, mirrors the source escape
    pytest.param(0.001,  '1.000 ms',       id = 'ms_boundary'),
    pytest.param(0.005,  '5.000 ms',       id = 'ms_fractional'),
    pytest.param(0.5,    '500 ms',         id = 'ms_integer'),
    pytest.param(5.0,    '5.000 sec',      id = 'seconds_only'),
    pytest.param(65,     '1min 5sec',      id = 'minutes'),
    pytest.param(3661,   '1h 1min 1sec',   id = 'hours'),
])
def test_time_to_string(seconds, target):
    assert time_to_string(seconds) == target


# --- inspect utilities -------------------------------------------------------------

@pytest.mark.parametrize('include_args,target', [
    pytest.param(True,  ['a', 'b', 'c', 'args'], id = 'with_varargs'),
    pytest.param(False, ['a', 'b', 'c'],         id = 'without_varargs'),
])
def test_get_args(include_args, target):
    # keyword-only (`d`) and `**kwargs` are never positional -> excluded
    assert get_args(_sample_fn, include_args = include_args) == target


def test_get_kwargs():
    # only parameters carrying a default value are returned
    assert get_kwargs(_sample_fn) == {'c' : 3, 'd' : 4}
    assert get_kwargs(_no_varargs) == {'b' : 2}


def test_get_annotations():
    assert get_annotations(_annotated) == {'a' : int, 'b' : str, 'return' : bool}


def test_has_args():
    assert has_args(_sample_fn) is True
    assert has_args(_no_varargs) is False


def test_has_kwargs():
    assert has_kwargs(_sample_fn) is True
    assert has_kwargs(_sample_fn, name = 'kwargs') is True   # var-keyword param is named `kwargs`
    assert has_kwargs(_sample_fn, name = 'other') is False
    assert has_kwargs(_no_varargs) is False


def test_get_fn_name():
    assert get_fn_name(_sample_fn) == '_sample_fn'
    assert get_fn_name(partial(_sample_fn, 1)) == '_sample_fn'  # unwraps `.func`
    assert get_fn_name(_Named()) == 'custom_name'              # `.name` attribute wins
    assert get_fn_name(_Anonymous()) == '_Anonymous'          # falls back to class name


def test_signature_to_str():
    assert signature_to_str(_no_varargs) == '_no_varargs(a, b=2)'


@pytest.mark.parametrize('value', [
    pytest.param(_Color.RED, id = 'enum_member'),
    pytest.param('red',      id = 'lower_str'),    # `upper_names = True` by default
    pytest.param('RED',      id = 'upper_str'),
    pytest.param(1,          id = 'raw_value'),
])
def test_get_enum_item(value):
    assert get_enum_item(value, _Color) is _Color.RED


def test_get_enum_item_invalid():
    with pytest.raises(KeyError):
        get_enum_item('blue', _Color)


# --- create_iterable ---------------------------------------------------------------

@pytest.mark.parametrize('data', [
    pytest.param([1, 2, 3, 4], id = 'list'),
    pytest.param((1, 2, 3, 4), id = 'tuple'),
    pytest.param({1, 2, 3, 4}, id = 'set'),
    pytest.param(range(1, 5), id = 'range'),
    pytest.param(np.arange(1, 5), id = 'array'),
    pytest.param(_generator, id = 'function'),
    pytest.param(_generator(), id = 'generator'),
])
def test_create_iterable_simple(data):
    iterable = create_iterable(data)
    assert inspect.isgenerator(iterable)
    # `set` iteration order is unspecified -> compare order-independently
    assert sorted(int(item) for item in iterable) == [1, 2, 3, 4]


@pytest.mark.parametrize('make_queue', [
    pytest.param(queue.Queue, id = 'queue'),
    pytest.param(multiprocessing.Queue, id = 'multiprocessing_queue'),
])
def test_create_iterable_queue(make_queue):
    q = make_queue()
    for i in range(1, 5):
        q.put(i)
    q.put(None)  # `None` is the sentinel that stops the iteration

    assert [int(item) for item in create_iterable(q)] == [1, 2, 3, 4]


def test_create_iterable_dataframe():
    df = pd.DataFrame([{'x' : 1}, {'x' : 2}, {'x' : 3}])
    rows = list(create_iterable(df))
    assert [int(row['x']) for row in rows] == [1, 2, 3]


def test_create_iterable_forwards_kwargs():
    # only signature-matching kwargs are forwarded to a generator function
    assert list(create_iterable(_gen_kwargs, n = 3)) == [0, 1, 2]
    assert list(create_iterable(_gen_kwargs, n = 2, unknown = 'ignored')) == [0, 1]
