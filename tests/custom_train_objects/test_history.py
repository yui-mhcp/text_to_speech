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

""" Pure-``pytest`` tests for :mod:`custom_train_objects.history`.

``History`` is a plain-Python training-history container / keras-callback shim :
it imports **no** ``keras`` (only ``utils`` for the json round-trip), so these
tests run **without a deep-learning backend** and are therefore left unmarked.

The class is essentially a small state-machine driven by the keras callback
hooks (``on_train_begin`` -> ``on_epoch_begin`` -> ``on_train_batch_end`` ->
``on_epoch_end`` -> ``on_train_end``, with an interleaved ``on_test_*`` cycle for
validation / evaluation). The helpers below replay those hooks; the tests then
assert on the derived views (``epochs`` / ``steps`` / ``metrics`` / ``history`` /
``get_best`` / indexing) and on the json ``save`` / ``load`` round-trip.
"""

import os
import pytest

from custom_train_objects.history import History, Phase


# --- hook-replay helpers -----------------------------------------------------------

def _train_epoch(h, epoch, batch_logs):
    """ Replay a full training epoch from a list of per-batch ``logs`` dicts. """
    h.on_epoch_begin(epoch)
    for batch, logs in enumerate(batch_logs):
        h.on_train_batch_end(batch, logs)
    h.on_epoch_end(epoch)

def _validate_epoch(h, batch_logs):
    """ Replay a validation pass (``on_test_*`` while training). """
    h.on_test_begin()
    for batch, logs in enumerate(batch_logs):
        h.on_test_batch_end(batch, logs)
    h.on_test_end()


@pytest.fixture
def trained():
    """ A two-epoch training : decreasing ``loss``, increasing ``acc``. """
    h = History()
    h.on_train_begin()
    _train_epoch(h, 0, [{'loss' : 2., 'acc' : 0.2}, {'loss' : 1., 'acc' : 0.4}])
    _train_epoch(h, 1, [{'loss' : 1., 'acc' : 0.6}, {'loss' : 0.5, 'acc' : 0.8}])
    h.on_train_end()
    return h


# --- empty / initial state ---------------------------------------------------------

def test_initial_state():
    h = History()
    assert len(h) == 0
    assert h.epochs == 0
    assert h.steps == 0
    assert h.phase == Phase.SLEEP
    assert not h.is_training
    assert not h.is_evaluating
    assert h.history == []
    assert h.metrics == {}

def test_get_best_empty_is_none():
    assert History().get_best('loss') is None

def test_str_index_on_empty_raises():
    with pytest.raises(RuntimeError):
        History()['loss']


# --- nominal training : derived views ----------------------------------------------

def test_epochs_and_steps(trained):
    assert trained.epochs == 2
    assert len(trained) == 2
    assert trained.steps == 4          # 2 batches * 2 epochs
    assert trained.phase == Phase.SLEEP

def test_history_keeps_last_batch_value(trained):
    # `history` exposes, per epoch, the *last* batch value of each metric
    assert trained.history == [
        {'loss' : 1.0, 'acc' : 0.4},
        {'loss' : 0.5, 'acc' : 0.8},
    ]

def test_metrics_view(trained):
    assert trained.metrics == {
        'loss' : {0 : 1.0, 1 : 0.5},
        'acc'  : {0 : 0.4, 1 : 0.8},
    }

def test_training_time_is_positive(trained):
    assert isinstance(trained.training_time, float)
    assert trained.training_time >= 0.


# --- indexing ----------------------------------------------------------------------

def test_int_index_returns_epoch(trained):
    assert trained[0] == {'loss' : 1.0, 'acc' : 0.4}
    assert trained[-1] == {'loss' : 0.5, 'acc' : 0.8}

def test_str_index_returns_metric_series(trained):
    assert trained['loss'] == {0 : 1.0, 1 : 0.5}

def test_invalid_index_type_raises(trained):
    with pytest.raises(ValueError):
        trained[1.5]


# --- get_best ----------------------------------------------------------------------

def test_get_best_loss_is_min(trained):
    assert trained.get_best('loss') == 0.5

def test_get_best_metric_is_max_when_increasing(trained):
    # `acc` increases over epochs -> the best is the maximum
    assert trained.get_best('acc') == 0.8

def test_get_best_unknown_metric_is_none(trained):
    assert trained.get_best('does_not_exist') is None

def test_get_best_single_epoch_returns_only_value():
    h = History()
    h.on_train_begin()
    _train_epoch(h, 0, [{'acc' : 0.3}])
    h.on_train_end()
    assert h.get_best('acc') == 0.3


# --- validation cycle --------------------------------------------------------------

def test_validation_metrics_merged_into_epoch():
    h = History()
    h.on_train_begin()
    h.on_epoch_begin(0)
    h.on_train_batch_end(0, {'loss' : 1.0})
    _validate_epoch(h, [{'val_loss' : 1.5}])
    h.on_epoch_end(0)
    h.on_train_end()

    assert h.history == [{'loss' : 1.0, 'val_loss' : 1.5}]

def test_phase_transitions_during_validation():
    h = History()
    h.on_train_begin()
    assert h.is_training and h.phase == Phase.TRAIN

    h.on_epoch_begin(0)
    h.on_test_begin()
    assert h.is_validating and h.is_evaluating and h.phase == Phase.VALID

    h.on_test_end()
    assert h.phase == Phase.TRAIN and not h.is_evaluating


# --- evaluation (test) cycle -------------------------------------------------------

def test_standalone_evaluation_phase(trained):
    trained.on_test_begin()
    assert trained.is_testing and trained.phase == Phase.TEST
    trained.on_test_batch_end(0, {'loss' : 0.1})
    trained.on_test_end()
    assert trained.phase == Phase.SLEEP


# --- guard rails -------------------------------------------------------------------

def test_train_batch_before_begin_raises():
    with pytest.raises(RuntimeError):
        History().on_train_batch_end(0, {'loss' : 1.})

def test_test_batch_before_begin_raises():
    with pytest.raises(RuntimeError):
        History().on_test_batch_end(0, {'loss' : 1.})

def test_test_end_before_begin_raises():
    with pytest.raises(RuntimeError):
        History().on_test_end()

def test_unexpected_epoch_raises():
    h = History()
    h.on_train_begin()
    with pytest.raises(RuntimeError):
        h.on_epoch_begin(3)        # expected is 0


# --- interrupted training (regression for the `on_train_end` fix) ------------------

def test_interrupted_training_commits_pending_epoch():
    """ Pending epoch history at ``on_train_end`` -> committed + flagged interrupted.

        Regression test : the old code raised ``AttributeError`` (``self.epoch``)
        and never set the ``interrupted`` flag (typo ``interrupdated``).
    """
    h = History()
    h.on_train_begin()
    h.on_epoch_begin(0)
    h.on_train_batch_end(0, {'loss' : 1.})
    h.on_train_end()               # no on_epoch_end -> interrupted

    assert h.epochs == 1
    assert h.history == [{'loss' : 1.0}]
    assert h.training_logs[0]['interrupted'] is True

def test_clean_training_not_flagged_interrupted(trained):
    assert trained.training_logs[0]['interrupted'] is False


# --- training config / infos -------------------------------------------------------

def test_set_config_exposed_after_train_begin():
    h = History()
    h.set_config(
        hparams = {'lr' : 1e-3},
        config  = {'epochs' : 5, 'initial_epoch' : 0},
        dataset_infos = {'name' : 'dummy'}
    )
    h.on_train_begin()

    assert h.training_config[0] == {'lr' : 1e-3, 'epochs' : 5, 'initial_epoch' : 0}
    assert h.training_infos[0]['dataset'] == {'name' : 'dummy'}


# --- json save / load round-trip ---------------------------------------------------

def test_save_load_roundtrip(trained, tmp_path, asserts):
    path = os.path.join(tmp_path, 'history.json')
    trained.save(path)
    assert os.path.exists(path)

    reloaded = History.load(path)
    assert reloaded.epochs == trained.epochs
    assert reloaded.steps == trained.steps
    asserts.assert_equal(trained.history, reloaded.history)
    asserts.assert_equal(trained.metrics, reloaded.metrics)

def test_load_missing_file_returns_empty(tmp_path):
    reloaded = History.load(os.path.join(tmp_path, 'does_not_exist.json'))
    assert len(reloaded) == 0
