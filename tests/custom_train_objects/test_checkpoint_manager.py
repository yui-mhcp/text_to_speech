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

""" Pure-``pytest`` tests for :mod:`custom_train_objects.checkpoint_manager`.

``CheckpointManager`` tracks weight files + a small json state (``checkpoint.json``)
and imports **no** ``keras`` (only ``utils`` for the json I/O), so these tests run
**without a deep-learning backend** and are left unmarked.

The parts that actually (de)serialize a model (``save`` / ``load``) need a keras
model and are therefore *not* covered here — only the keras-free surface is :
filename formatting, the ``infos`` / state views, and the ``clear`` bookkeeping.

A lightweight ``_FakeCore`` stands in for the ``BaseModel`` the manager wraps
(it only needs ``save_dir`` / ``epochs`` / ``steps``).
"""

import os
import pytest

from custom_train_objects.checkpoint_manager import (
    CheckpointManager, standardize_checkpoint_format, _empty_checkpoint_infos
)


class _FakeCore:
    def __init__(self, save_dir, epochs = 3, steps = 120):
        self.save_dir   = save_dir
        self.epochs     = epochs
        self.steps      = steps


@pytest.fixture
def manager(tmp_path):
    """ A manager rooted in an isolated (empty) directory. """
    return CheckpointManager(directory = str(tmp_path))


# --- standardize_checkpoint_format -------------------------------------------------

@pytest.mark.parametrize('fmt,target', [
    pytest.param('ckpt.weights.h5', 'ckpt-{counter:04d}.weights.h5', id = 'default'),
    pytest.param('model-{epoch}.keras', 'model-{epoch}-{counter:04d}.keras', id = 'epoch'),
    pytest.param('ckpt-{counter:02d}.weights.h5', 'ckpt-{counter:02d}.weights.h5', id = 'has_counter'),
    pytest.param('model', 'model-{counter:04d}.keras', id = 'no_extension'),
    pytest.param('epoch_{epoch}.weights.h5', 'epoch_{epoch}-{counter:04d}.weights.h5', id = 'prefix'),
])
def test_standardize_checkpoint_format(fmt, target):
    assert standardize_checkpoint_format(fmt) == target


# --- directory / infos plumbing ----------------------------------------------------

def test_directory_from_explicit_path(tmp_path, manager):
    assert manager.directory == str(tmp_path)
    assert manager.checkpoint_file == os.path.join(str(tmp_path), 'checkpoint.json')
    assert manager.best_checkpoint_path == os.path.join(str(tmp_path), 'best.weights.h5')

def test_infos_without_core(manager):
    # no `core` -> epoch / step default to -1, counter starts at 0
    assert manager.infos == {'epoch' : -1, 'step' : -1, 'counter' : 0}

def test_directory_and_infos_from_core(tmp_path):
    core = _FakeCore(save_dir = str(tmp_path), epochs = 7, steps = 42)
    cm   = CheckpointManager(core = core)
    assert cm.directory == str(tmp_path)
    assert cm.epoch == 7 and cm.step == 42
    assert cm.infos == {'epoch' : 7, 'step' : 42, 'counter' : 0}


# --- empty state -------------------------------------------------------------------

def test_empty_state(manager):
    assert len(manager) == 0
    assert manager.counter == 0
    assert manager.loaded == -1
    assert manager.checkpoints == []
    assert manager.latest_checkpoint is None
    assert manager.best_checkpoint is None

def test_repr_without_checkpoint(manager):
    assert 'no checkpoint created' in repr(manager)

def test_state_is_not_shared_between_managers(tmp_path):
    # the default state is deep-copied : mutating one must not leak into another
    a = CheckpointManager(directory = str(tmp_path / 'a'))
    b = CheckpointManager(directory = str(tmp_path / 'b'))
    a._state['checkpoints'].append({'epoch' : 0, 'step' : 0, 'counter' : 0})
    assert b._state['checkpoints'] == []
    assert _empty_checkpoint_infos['checkpoints'] == []


# --- get_filename ------------------------------------------------------------------

def test_get_filename_uses_standardized_format(tmp_path, manager):
    # default format 'ckpt.weights.h5' -> 'ckpt-{counter:04d}.weights.h5'
    assert manager.get_filename() == os.path.join(str(tmp_path), 'ckpt-0000.weights.h5')

def test_get_filename_with_explicit_infos(tmp_path, manager):
    infos = {'epoch' : 2, 'step' : 10, 'counter' : 5}
    assert manager.get_filename(infos) == os.path.join(str(tmp_path), 'ckpt-0005.weights.h5')


# --- state persistence -------------------------------------------------------------

def test_best_checkpoint_infos_roundtrip(tmp_path, manager):
    manager.set_best_checkpoint_infos(epoch = 2, logs = {'val_loss' : 0.5})
    assert os.path.exists(manager.checkpoint_file)

    # a fresh manager on the same directory must reload that state
    reloaded = CheckpointManager(directory = str(tmp_path))
    assert reloaded._state['best_checkpoint'] == {'val_loss' : 0.5, 'epoch' : 2}


# --- clear bookkeeping (regression for the `self.loaded` fix) -----------------------

def _register(cm, counter):
    """ Append a fake checkpoint + create its (empty) weight file. """
    infos = {'epoch' : counter, 'step' : counter * 10, 'counter' : counter}
    cm._state['checkpoints'].append(infos)
    open(cm.get_filename(infos), 'w').close()
    return infos

def test_clear_keeps_only_loaded(manager):
    """ ``clear`` must delete every checkpoint but the loaded one.

        Regression test : the old code referenced ``self.loaded_index`` (which does
        not exist) -> ``AttributeError``.
    """
    registered = [_register(manager, c) for c in range(3)]
    manager._state['loaded'] = 1

    manager.clear()

    assert len(manager) == 1
    assert manager[0] == registered[1]
    # the kept file is still on disk, the others were removed
    assert os.path.exists(manager.get_filename(registered[1]))
    assert not os.path.exists(manager.get_filename(registered[0]))
    assert not os.path.exists(manager.get_filename(registered[2]))
