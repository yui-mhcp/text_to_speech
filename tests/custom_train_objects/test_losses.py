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

""" Pure-``pytest`` tests for :mod:`custom_train_objects.losses`.

Covers the ``get_loss`` factory (registry resolution + config / serialization
round-trips for every custom loss) and a behavioural check that ``TextLoss``
ignores padded timesteps.

All custom losses build keras objects -> the module is marked ``keras`` and
skipped when the backend is missing.
"""

import numpy as np
import pytest

keras = pytest.importorskip('keras')

from custom_train_objects.losses import (
    get_loss, TextLoss, CTCLoss, TacotronLoss, YoloLoss, GE2ELoss,
)
from ._helpers import normalize_object

pytestmark = pytest.mark.keras


# instances are built once at collection (keras is available by then)
_CUSTOM_LOSSES = [
    pytest.param(
        'TextLoss',
        TextLoss(pad_value = 0, eos_value = -1, warmup_tokens = 5, from_logits = True),
        id = 'text_loss'
    ),
    pytest.param('CTCLoss', CTCLoss(pad_value = 0), id = 'ctc_loss'),
    pytest.param('TacotronLoss', TacotronLoss(), id = 'tacotron_loss'),
    pytest.param('YoloLoss', YoloLoss(), id = 'yolo_loss'),
    pytest.param(
        'GE2ELoss', GE2ELoss(mode = 'softmax', distance_metric = 'cosine'), id = 'ge2e_loss'
    ),
]


# --- registry round-trips ----------------------------------------------------------

@pytest.mark.parametrize('name,loss', _CUSTOM_LOSSES)
def test_loss_config_roundtrip(name, loss, asserts):
    config  = {k : v for k, v in loss.get_config().items() if k != 'fn'}
    rebuilt = get_loss(name, ** config)
    asserts.assert_equal(normalize_object(loss), normalize_object(rebuilt))

@pytest.mark.parametrize('name,loss', _CUSTOM_LOSSES)
def test_loss_serialization_roundtrip(name, loss, asserts):
    rebuilt = get_loss(keras.losses.serialize(loss))
    asserts.assert_equal(normalize_object(loss), normalize_object(rebuilt))


# --- get_loss : resolution rules ---------------------------------------------------

def test_get_loss_crossentropy_passthrough():
    assert get_loss('crossentropy') == 'crossentropy'

def test_get_loss_instance_passthrough():
    loss = TextLoss()
    assert get_loss(loss) is loss

def test_get_loss_unknown_raises():
    with pytest.raises(ValueError):
        get_loss('does_not_exist')


# --- TextLoss : padding is ignored -------------------------------------------------

def test_text_loss_ignores_padding(asserts):
    """ Changing predictions on padded timesteps must not change the loss. """
    K       = keras.ops
    loss    = TextLoss(pad_value = 0, eos_value = -1)        # pure `!= pad` mask
    y_true  = K.convert_to_tensor([[3, 1, 0, 0]], 'int32')  # 2 real tokens + padding

    base    = np.random.default_rng(0).normal(size = (1, 4, 5)).astype('float32')
    altered = base.copy()
    altered[0, 2:] = 100.                                    # only the padded steps

    asserts.assert_equal(
        loss(y_true, K.convert_to_tensor(base)),
        loss(y_true, K.convert_to_tensor(altered)),
        max_err = 1e-5
    )
