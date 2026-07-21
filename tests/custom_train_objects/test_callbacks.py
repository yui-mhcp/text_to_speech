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

""" Pure-``pytest`` tests for :mod:`custom_train_objects.callbacks`.

Covers the ``get_callbacks`` factory (``None`` / instance / str / dict / list
resolution) and the ``HistoryCallback`` delegation to its wrapped ``History``.

Builds keras callbacks -> marked ``keras`` and skipped when the backend is missing.
"""

import pytest

keras = pytest.importorskip('keras')

from custom_train_objects.callbacks import get_callbacks, HistoryCallback
from custom_train_objects.history import History, Phase

pytestmark = pytest.mark.keras


# --- get_callbacks : resolution rules ----------------------------------------------

def test_get_callbacks_none():
    assert get_callbacks(None) is None
    assert get_callbacks() is None

def test_get_callbacks_instance_passthrough():
    cb = keras.callbacks.TerminateOnNaN()
    assert get_callbacks(cb) is cb

def test_get_callbacks_from_string():
    assert isinstance(get_callbacks('TerminateOnNaN'), keras.callbacks.TerminateOnNaN)

def test_get_callbacks_list():
    instance = keras.callbacks.TerminateOnNaN()
    result   = get_callbacks(['TerminateOnNaN', instance])
    assert isinstance(result[0], keras.callbacks.TerminateOnNaN)
    assert result[1] is instance

@pytest.mark.parametrize('spec', [
    pytest.param({'name' : 'TerminateOnNaN'}, id = 'name'),
    pytest.param({'class_name' : 'TerminateOnNaN', 'config' : {}}, id = 'class_name_config'),
])
def test_get_callbacks_from_dict(spec):
    assert isinstance(get_callbacks(spec), keras.callbacks.TerminateOnNaN)


# --- HistoryCallback : delegation --------------------------------------------------

def test_history_callback_delegates_hooks():
    history = History()
    callback = HistoryCallback(history)

    callback.on_train_begin()
    assert history.phase == Phase.TRAIN

    callback.on_epoch_begin(0)
    callback.on_train_batch_end(0, {'loss' : 1.0})
    callback.on_epoch_end(0)
    assert history.epochs == 1
    assert history.history == [{'loss' : 1.0}]
