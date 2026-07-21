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

""" Pure-``pytest`` tests for :mod:`utils.callbacks.file_saver`.

Covers ``FileSaver`` filename formatting / indexing, the "output already holds a
path" short-circuit, thread-safe index allocation, and ``join`` (regression : it
used to ``KeyError`` on a registry that was never populated). Also covers
``JSONSaver`` merge / idempotency and its previously-``None`` mutex.

A recording ``save_fn`` keeps everything backend-free (``save_in_parallel=0`` makes
the underlying ``Stream`` synchronous, so assertions are deterministic).
"""

import os
import json
import threading

import pytest

from utils.callbacks import FileSaver, JSONSaver
from ._helpers import RecordingSaveFn


def _make_saver(tmp_path, file_format = 'out-{}.txt', ** kwargs):
    save_fn = RecordingSaveFn()
    saver   = FileSaver(
        'data', os.path.join(str(tmp_path), file_format), save_fn = save_fn, ** kwargs
    )
    return saver, save_fn


# --- declarative capabilities ------------------------------------------------------

def test_saver_capabilities():
    assert FileSaver.saves_to_disk is True
    assert JSONSaver.saves_to_disk is True     # inherited from FileSaver
    assert JSONSaver.provides_entry is True
    assert FileSaver.provides_entry is False


# --- basic save + indexing ---------------------------------------------------------

def test_saves_payload_with_indexed_filename(tmp_path):
    saver, save_fn = _make_saver(tmp_path)

    infos = {}
    saver(infos, {'data': [1, 2, 3]})

    assert infos['data'].endswith('out-0.txt')
    assert len(save_fn.calls) == 1
    filename, data, _ = save_fn.calls[0]
    assert filename == infos['data']
    assert data == [1, 2, 3]

def test_index_increments_across_calls(tmp_path):
    saver, save_fn = _make_saver(tmp_path)

    first, second = {}, {}
    saver(first, {'data': [0]})
    saver(second, {'data': [1]})

    assert first['data'].endswith('out-0.txt')
    assert second['data'].endswith('out-1.txt')

def test_index_resumes_from_existing_files(tmp_path):
    # two files already on disk -> the next allocated index must be 2
    for i in range(2):
        open(os.path.join(str(tmp_path), 'out-{}.txt'.format(i)), 'w').close()

    saver, _ = _make_saver(tmp_path)
    infos = {}
    saver(infos, {'data': [0]})
    assert infos['data'].endswith('out-2.txt')

def test_non_indexed_format_is_constant(tmp_path):
    saver, save_fn = _make_saver(tmp_path, file_format = 'fixed.txt')
    assert not saver.use_index

    infos = {}
    saver(infos, {'data': [0]})
    assert infos['data'].endswith('fixed.txt')


# --- "output already holds a path" short-circuit -----------------------------------

def test_str_output_is_treated_as_existing_path(tmp_path):
    saver, save_fn = _make_saver(tmp_path)

    infos = {}
    saver(infos, {'data': 'already-there.txt'})

    assert infos['data'] == 'already-there.txt'
    assert save_fn.calls == [], 'a str payload must not be re-saved'


# --- thread-safe index allocation --------------------------------------------------

def test_concurrent_calls_allocate_unique_indices(tmp_path):
    saver, save_fn = _make_saver(tmp_path)
    saver.build()  # build once up-front so all threads share the same Stream / mutex

    n       = 32
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i):
        barrier.wait()
        infos = {}
        saver(infos, {'data': [i]})
        results[i] = infos['data']

    threads = [threading.Thread(target = worker, args = (i,)) for i in range(n)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(set(results)) == n, 'each concurrent call must get a distinct filename'
    expected = {'out-{}.txt'.format(i) for i in range(n)}
    assert {os.path.basename(r) for r in results} == expected


# --- join (regression : used to KeyError) ------------------------------------------

def test_join_after_use_does_not_raise(tmp_path):
    saver, _ = _make_saver(tmp_path)
    saver({}, {'data': [0]})
    saver.join()  # must not raise

def test_join_before_build_is_noop(tmp_path):
    saver, _ = _make_saver(tmp_path)
    assert not saver.built
    saver.join()  # never built -> no-op, must not raise


# --- JSONSaver ---------------------------------------------------------------------

def _make_json_saver(tmp_path):
    map_file = os.path.join(str(tmp_path), 'map.json')
    return JSONSaver(data = {}, filename = map_file, primary_key = 'text'), map_file

def test_json_saver_writes_and_merges_entries(tmp_path):
    saver, map_file = _make_json_saver(tmp_path)

    assert saver({'text': 'a', 'v': 1}, {}) == 'a'
    assert saver({'text': 'b', 'v': 2}, {}) == 'b'

    with open(map_file, 'r', encoding = 'utf-8') as f:
        stored = json.load(f)
    assert set(stored) == {'a', 'b'}
    assert stored['a']['v'] == 1 and stored['b']['v'] == 2

def test_json_saver_skips_non_str_and_missing_keys(tmp_path):
    saver, _ = _make_json_saver(tmp_path)
    assert saver({'text': 123}, {}) is None      # non-str primary key
    assert saver({'other': 'x'}, {}) is None      # primary key absent

def test_json_saver_does_not_rewrite_unchanged_entry(tmp_path, monkeypatch):
    import utils.callbacks.file_saver as fs_mod

    calls = {'n': 0}
    original = fs_mod.dump_json
    def counting_dump_json(* args, ** kwargs):
        calls['n'] += 1
        return original(* args, ** kwargs)
    monkeypatch.setattr(fs_mod, 'dump_json', counting_dump_json)

    saver, _ = _make_json_saver(tmp_path)
    infos = {'text': 'a', 'v': 1}
    saver(dict(infos), {})
    saver(dict(infos), {})  # identical -> must be a no-op write

    assert calls['n'] == 1, 'an unchanged entry must not trigger a second dump'
