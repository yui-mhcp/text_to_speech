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

""" Pure-``pytest`` tests for :mod:`utils.file_utils`.

Written as plain module-level functions + ``@pytest.mark.parametrize`` (no
``unittest.TestCase`` / ``absl.parameterized``). Tolerant comparisons for numpy
arrays / DataFrames go through the ``asserts`` fixture (``tests.asserts``), while
booleans / strings use plain ``assert``. Files are written to the ``tmp_path``
fixture so the test-data directory is never polluted.
"""

import os
import glob
import pytest
import hashlib
import numpy as np
import pandas as pd

from tests import data_dir
from utils.file_utils import (
    is_path, path_to_unix, expand_path, hash_file,
    contains_index_format, get_path_index, format_path_index,
    sort_files, remove_path_prefix, load_data, dump_data,
)

# data shared between the load fixtures and the round-trip cases
_JSON_DATA  = {"a" : 1, "b" : 2, "c" : True, "d" : "Hello World !", "e" : {"aa" : 1.}, "f" : [1, 2., "3"]}
_DF_DATA    = pd.DataFrame([{'a' : 1, 'b' : 2}, {'a' : 2, 'b' : 3}])


# --- path predicates / helpers -----------------------------------------------------

@pytest.mark.parametrize('path,target', [
    pytest.param('/', os.path.exists('/'), id = 'root'),
    pytest.param('.', True, id = 'current'),
    pytest.param('..', True, id = 'parent'),
    pytest.param('~', True, id = 'home'),
    pytest.param('~/*', True, id = 'home_formatted'),
    pytest.param(data_dir, True, id = 'folder'),
    pytest.param(os.path.join(data_dir, 'audio_test.wav'), True, id = 'file'),
    pytest.param(os.path.join(data_dir, '*.wav'), True, id = 'unix_format'),

    pytest.param(None, False, id = 'none'),
    pytest.param('', False, id = 'empty'),
    pytest.param(' ', False, id = 'wrong'),
    pytest.param('?%%µ*%`"', False, id = 'wrong_symbol'),
    pytest.param(os.path.join(data_dir, 'audio_{}.wav'), False, id = 'py_format'),
])
def test_is_path(path, target):
    assert is_path(path) == target


@pytest.mark.parametrize('value,target', [
    pytest.param('a\\b\\c', 'a/b/c', id = 'backslashes'),
    pytest.param('already/unix', 'already/unix', id = 'unix'),
    pytest.param(None, None, id = 'none'),
    pytest.param(123, 123, id = 'non_str'),
])
def test_path_to_unix(value, target):
    assert path_to_unix(value) == target


@pytest.mark.parametrize('path,target', [
    pytest.param('out_{}.png', True, id = 'empty_braces'),
    pytest.param('out_{i}.png', True, id = 'i'),
    pytest.param('out_{i:02d}.png', True, id = 'i_padded'),
    pytest.param('out_{:02d}.png', True, id = 'padded'),
    pytest.param('out_{}_{}.png', True, id = 'double_braces'),

    pytest.param('plain.png', False, id = 'plain'),
    pytest.param('no_format_at_all', False, id = 'no_format'),
])
def test_contains_index_format(path, target):
    assert contains_index_format(path) == target


@pytest.mark.parametrize('files,target', [
    # sorted by (len, name) : shorter names first, alphabetical within a length
    pytest.param(['bbb', 'a', 'cc', 'ab'], ['a', 'ab', 'cc', 'bbb'], id = 'len_then_name'),
    # length-aware, so 'file10' comes after 'file2' (not lexicographically before)
    pytest.param(['file2', 'file10', 'file1'], ['file1', 'file2', 'file10'], id = 'length_aware'),
])
def test_sort_files(files, target):
    assert sort_files(files) == target


# --- index-formatted paths (need real files) ---------------------------------------

def test_get_path_index(tmp_path):
    pattern = os.path.join(str(tmp_path), 'file_{}.txt')
    assert get_path_index(pattern) == 0

    for i in range(2):
        with open(os.path.join(str(tmp_path), 'file_{}.txt'.format(i)), 'w') as f:
            f.write('x')

    assert get_path_index(pattern) == 2


def test_format_path_index(tmp_path):
    pattern = os.path.join(str(tmp_path), 'file_{}.txt')
    assert format_path_index(pattern) == os.path.join(str(tmp_path), 'file_0.txt')

    with open(os.path.join(str(tmp_path), 'file_0.txt'), 'w') as f:
        f.write('x')

    assert format_path_index(pattern) == os.path.join(str(tmp_path), 'file_1.txt')


def test_format_path_index_zero_padded(tmp_path):
    pattern = os.path.join(str(tmp_path), 'img_{i:02d}.png')
    assert get_path_index(pattern) == 0
    assert format_path_index(pattern) == os.path.join(str(tmp_path), 'img_00.png')


# --- remove_path_prefix ------------------------------------------------------------

def test_remove_path_prefix_str():
    # the prefix is normalized, so a trailing '/' is optional : both forms strip cleanly
    assert remove_path_prefix('/d/audio/x.wav', '/d/audio/') == 'x.wav'
    assert remove_path_prefix('/d/audio/x.wav', '/d/audio') == 'x.wav'
    # non-matching prefix : path is returned (unix-normalized) unchanged
    assert remove_path_prefix('/other/x.wav', '/d/') == '/other/x.wav'


def test_remove_path_prefix_list():
    assert remove_path_prefix(['/d/a.wav', '/d/b.wav'], '/d/') == ['a.wav', 'b.wav']


def test_remove_path_prefix_dict():
    # only keys whose name contains 'filename' are rewritten
    data = {'filename' : '/d/a.wav', 'label' : '/d/keep'}
    assert remove_path_prefix(data, '/d/') == {'filename' : 'a.wav', 'label' : '/d/keep'}


def test_remove_path_prefix_dataframe(asserts):
    df = pd.DataFrame({'filename' : ['/d/a.wav', '/d/b.wav'], 'label' : [0, 1]})
    # the DataFrame branch mutates in place and returns the same (now stripped) df
    result = remove_path_prefix(df, '/d/')
    assert result is df
    assert df['filename'].tolist() == ['a.wav', 'b.wav']
    assert df['label'].tolist() == [0, 1]


# --- hash_file ---------------------------------------------------------------------

def test_hash_file():
    path  = os.path.join(data_dir, 'audio_test.wav')
    other = os.path.join(data_dir, 'files', 'test.json')

    with open(path, 'rb') as f:
        expected = hashlib.sha256(f.read()).hexdigest()

    # matches an independently computed SHA-256
    assert hash_file(path) == expected
    # invariant to the reading block size
    assert hash_file(path) == hash_file(path, block_size = 1024)
    # different files produce different hashes
    assert hash_file(path) != hash_file(other)


# --- expand_path -------------------------------------------------------------------

def test_expand_path_empty():
    assert expand_path(None) == []
    assert expand_path('') == []


def test_expand_path():
    files     = [f for f in glob.glob(os.path.join(data_dir, '*')) if os.path.isfile(f)]
    files_rec = sorted(files + glob.glob(os.path.join(data_dir, '**', '*')))

    assert expand_path(data_dir, recursive = False, unix = False) == files
    assert expand_path(data_dir, recursive = False, unix = True) == [
        f.replace(os.path.sep, '/') for f in files
    ]
    assert set(expand_path(data_dir, recursive = True, unix = True)) == set(
        f.replace(os.path.sep, '/') for f in files_rec
    )


# --- load_data : committed fixture files -------------------------------------------

@pytest.mark.parametrize('ext,target', [
    pytest.param('txt', None, id = 'txt'),
    pytest.param('md', None, id = 'md'),
    pytest.param('py', None, id = 'py'),
    pytest.param('json', _JSON_DATA, id = 'json'),
    pytest.param('npy', np.arange(5, dtype = 'int32'), id = 'npy'),
    pytest.param('csv', _DF_DATA, id = 'csv'),
    pytest.param('tsv', _DF_DATA, id = 'tsv'),
])
def test_load(ext, target, asserts):
    path = os.path.join(data_dir, 'files', 'test.' + ext)
    if target is None:
        with open(path, 'r', encoding = 'utf-8') as file:
            target = file.read()

    asserts.assert_equal(target, load_data(path))


# --- dump_data / load_data round-trips ---------------------------------------------

@pytest.mark.parametrize('ext,data,kwargs', [
    pytest.param('txt', 'Hello World !\nsecond line', {}, id = 'txt'),
    pytest.param('json', _JSON_DATA, {}, id = 'json'),
    pytest.param('jsonl', [{'a' : 1}, {'b' : 2, 'c' : [1, 2, 3]}], {}, id = 'jsonl'),
    pytest.param('npy', np.arange(5, dtype = 'int32'), {}, id = 'npy'),
    pytest.param('npz', {'a' : np.arange(3, dtype = 'int32'), 'b' : np.ones((2, 2), dtype = 'float32')}, {}, id = 'npz'),
    pytest.param('pkl', {'x' : [1, 2, 3], 'y' : 'hello', 'z' : (1, 2)}, {}, id = 'pkl'),
    pytest.param('csv', _DF_DATA, {}, id = 'csv'),
    pytest.param('tsv', _DF_DATA, {}, id = 'tsv'),
    pytest.param('pdpkl', _DF_DATA, {}, id = 'pdpkl'),
])
def test_dump_roundtrip(ext, data, kwargs, tmp_path, asserts):
    path = os.path.join(str(tmp_path), 'test.' + ext)

    dump_data(path, data, ** kwargs)
    assert os.path.exists(path)
    asserts.assert_equal(data, load_data(path))


def test_dump_roundtrip_h5(tmp_path, asserts):
    pytest.importorskip('h5py')
    path = os.path.join(str(tmp_path), 'test.h5')
    data = {'a' : np.arange(3, dtype = 'int32'), 'b' : np.array([1., 2., 3.])}

    dump_data(path, data)
    asserts.assert_equal(data, load_data(path))


def test_dump_roundtrip_xlsx(tmp_path, asserts):
    pytest.importorskip('openpyxl')
    path = os.path.join(str(tmp_path), 'test.xlsx')

    # `index = False` so no spurious 'Unnamed: 0' column comes back on read
    dump_data(path, _DF_DATA, index = False)
    asserts.assert_equal(_DF_DATA, load_data(path))


# --- jsonl-specific behaviour (append + corrupt-line resilience) -------------------

def test_load_jsonl(tmp_path, asserts):
    path = os.path.join(str(tmp_path), 'test.jsonl')
    with open(path, 'w', encoding = 'utf-8') as f:
        f.write('{"a": 1}\n{"b": 2, "c": [1, 2, 3]}\n')

    asserts.assert_equal([{'a' : 1}, {'b' : 2, 'c' : [1, 2, 3]}], load_data(path))


def test_dump_jsonl_append(tmp_path, asserts):
    path = os.path.join(str(tmp_path), 'test.jsonl')

    dump_data(path, [{'a' : 1}], mode = 'a')
    dump_data(path, [{'b' : 2}, {'c' : 3}], mode = 'a')

    asserts.assert_equal([{'a' : 1}, {'b' : 2}, {'c' : 3}], load_data(path))


def test_dump_jsonl_overwrite(tmp_path, asserts):
    path = os.path.join(str(tmp_path), 'test.jsonl')

    dump_data(path, [{'a' : 1}, {'b' : 2}])
    dump_data(path, [{'c' : 3}])  # default mode = 'w' rewrites the whole file

    asserts.assert_equal([{'c' : 3}], load_data(path))


def test_load_jsonl_skips_corrupt_lines(tmp_path, asserts):
    path = os.path.join(str(tmp_path), 'test.jsonl')
    # blank line + a truncated trailing append (simulating a crash mid-write)
    with open(path, 'w', encoding = 'utf-8') as f:
        f.write('{"a": 1}\n\n{"b": 2}\n{"c": ')

    asserts.assert_equal([{'a' : 1}, {'b' : 2}], load_data(path))


# --- load-only formats (no matching `dump_*` dispatcher) ---------------------------

def test_load_yaml(tmp_path, asserts):
    path = os.path.join(str(tmp_path), 'test.yaml')
    with open(path, 'w', encoding = 'utf-8') as f:
        f.write('a: 1\nb:\n  - 1\n  - 2\n  - 3\nc: hello\n')

    asserts.assert_equal({'a' : 1, 'b' : [1, 2, 3], 'c' : 'hello'}, load_data(path))
