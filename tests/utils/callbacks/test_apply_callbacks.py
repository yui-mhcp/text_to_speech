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

""" Pure-``pytest`` tests for :func:`utils.callbacks.apply_callbacks`.

Covers the generic orchestration : empty / ``None`` no-op, ``save=False`` skipping
``FileSaver`` instances, per-callback exception isolation, and ``JSONSaver`` entry
capture (which only works once ``Callback.__call__`` returns ``apply``'s result).
"""

import os
import json

from utils.callbacks import apply_callbacks, JSONSaver
from ._helpers import RecordingCallback, RaisingCallback, RecordingSaveFn


# --- no-op cases -------------------------------------------------------------------

def test_empty_and_none_are_noop():
    assert apply_callbacks([], {}, {}) is None
    assert apply_callbacks(None, {}, {}) is None


# --- save=False skips FileSaver ----------------------------------------------------

def test_save_false_skips_file_savers(tmp_path):
    from utils.callbacks import FileSaver

    save_fn = RecordingSaveFn()
    saver   = FileSaver(
        'data', os.path.join(str(tmp_path), 'out-{}.txt'), save_fn = save_fn
    )
    recorder = RecordingCallback()

    apply_callbacks([saver, recorder], {}, {'data': [1, 2, 3]}, save = False)

    assert save_fn.calls == [], 'FileSaver must be skipped when save=False'
    assert not saver.built, 'a skipped FileSaver must not build'
    assert len(recorder.calls) == 1, 'non-saver callbacks still run'


# --- exception isolation -----------------------------------------------------------

def test_exception_in_one_callback_does_not_stop_the_others():
    recorder = RecordingCallback()
    # the raising callback comes first : the recorder must still run
    apply_callbacks([RaisingCallback(), recorder], {}, {'x': 1})
    assert len(recorder.calls) == 1


# --- JSONSaver entry capture -------------------------------------------------------

def test_returns_json_saver_entry(tmp_path):
    map_file = os.path.join(str(tmp_path), 'map.json')
    saver    = JSONSaver(data = {}, filename = map_file, primary_key = 'text')

    entry = apply_callbacks([saver], {'text': 'hello', 'v': 1}, {}, save = True)

    assert entry == 'hello'
    assert os.path.exists(map_file)
    with open(map_file, 'r', encoding = 'utf-8') as f:
        stored = json.load(f)
    assert stored['hello']['v'] == 1


# --- capability-based dispatch (no isinstance coupling) -----------------------------

def test_default_capabilities_are_false():
    cb = RecordingCallback()
    assert cb.saves_to_disk is False
    assert cb.provides_entry is False

def test_saves_to_disk_capability_controls_skip():
    class DiskCallback(RecordingCallback):
        saves_to_disk = True

    disk, plain = DiskCallback(), RecordingCallback()
    apply_callbacks([disk, plain], {}, {'x': 1}, save = False)

    assert disk.calls == [], 'any saves_to_disk callback is skipped when save=False'
    assert len(plain.calls) == 1

def test_provides_entry_capability_is_captured():
    class EntryCallback(RecordingCallback):
        provides_entry = True

    entry_cb = EntryCallback(return_value = 'K')
    entry = apply_callbacks([RecordingCallback(return_value = 'ignored'), entry_cb], {}, {})
    assert entry == 'K', 'the entry comes from the provides_entry callback, not the others'
