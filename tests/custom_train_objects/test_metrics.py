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

""" Pure-``pytest`` tests for :mod:`custom_train_objects.metrics`.

Covers the ``get_metrics`` factory (registry resolution + serialization round-trip)
and the behaviour of the two custom metrics : ``TextAccuracy`` (masked token
accuracy + exact-match) and ``EER`` (equal-error-rate on top of keras ``AUC``).

Builds keras metrics -> marked ``keras`` and skipped when the backend is missing.
"""

import numpy as np
import pytest

keras = pytest.importorskip('keras')

from custom_train_objects.metrics import get_metrics, TextAccuracy, EER
from ._helpers import normalize_object

pytestmark = pytest.mark.keras


# --- registry round-trips ----------------------------------------------------------

@pytest.mark.parametrize('name,metric', [
    pytest.param('TextAccuracy', TextAccuracy(pad_value = 0, eos_value = -1), id = 'text_accuracy'),
    pytest.param('EER', EER(), id = 'eer'),
])
def test_metric_serialization_roundtrip(name, metric, asserts):
    rebuilt = get_metrics(keras.metrics.serialize(metric))
    asserts.assert_equal(normalize_object(metric), normalize_object(rebuilt))

def test_text_accuracy_config_roundtrip(asserts):
    metric  = TextAccuracy(pad_value = 0, eos_value = 2)
    rebuilt = get_metrics('TextAccuracy', ** metric.get_config())
    asserts.assert_equal(normalize_object(metric), normalize_object(rebuilt))


# --- get_metrics : resolution rules ------------------------------------------------

def test_get_metrics_accuracy_passthrough():
    assert get_metrics('accuracy') == 'accuracy'

def test_get_metrics_instance_passthrough():
    metric = TextAccuracy()
    assert get_metrics(metric) is metric

def test_get_metrics_list():
    metrics = get_metrics(['accuracy', TextAccuracy()])
    assert metrics[0] == 'accuracy'
    assert isinstance(metrics[1], TextAccuracy)


# --- TextAccuracy : padding mask ---------------------------------------------------

def test_padding_mask_distinct_eos(asserts):
    # eos != pad -> pure `!= pad_value` mask
    metric  = TextAccuracy(pad_value = 0, eos_value = -1)
    mask    = metric.build_padding_mask(keras.ops.convert_to_tensor([[1, 2, 0, 0]], 'int32'))
    asserts.assert_equal(np.array([[True, True, False, False]]), np.asarray(mask))

def test_padding_mask_eos_equals_pad(asserts):
    # eos == pad -> length is `#non-pad + 1` (one eos slot kept)
    metric  = TextAccuracy(pad_value = 0, eos_value = 0)
    mask    = metric.build_padding_mask(keras.ops.convert_to_tensor([[1, 2, 0, 0]], 'int32'))
    asserts.assert_equal(np.array([[True, True, True, False]]), np.asarray(mask))


# --- TextAccuracy : accuracy / exact-match -----------------------------------------

def test_text_accuracy_perfect_match(asserts):
    metric  = TextAccuracy(pad_value = 0, eos_value = -1)
    y_true  = keras.ops.convert_to_tensor([[1, 2, 0]], 'int32')
    # argmax -> [1, 2, 0] ; position 2 is padding (ignored)
    y_pred  = keras.ops.convert_to_tensor(
        [[[0., 9., 0., 0.], [0., 0., 9., 0.], [9., 0., 0., 0.]]], 'float32'
    )
    metric.update_state(y_true, y_pred)
    res = metric.result()
    asserts.assert_equal(1.0, res['accuracy'])
    asserts.assert_equal(1.0, res['exact_match'])

def test_text_accuracy_partial_match(asserts):
    metric  = TextAccuracy(pad_value = 0, eos_value = -1)
    y_true  = keras.ops.convert_to_tensor([[1, 2, 0]], 'int32')
    # argmax -> [1, 3, 0] : the 2nd (unmasked) token is wrong -> 0.5 accuracy, no exact match
    y_pred  = keras.ops.convert_to_tensor(
        [[[0., 9., 0., 0.], [0., 0., 0., 9.], [9., 0., 0., 0.]]], 'float32'
    )
    metric.update_state(y_true, y_pred)
    res = metric.result()
    asserts.assert_equal(0.5, res['accuracy'])
    asserts.assert_equal(0.0, res['exact_match'])


# --- EER ---------------------------------------------------------------------------

def test_eer_perfect_separation(asserts):
    """ Perfectly separable scores -> EER ~ 0 and AUC ~ 1. """
    metric  = EER()
    y_true  = keras.ops.convert_to_tensor([0, 0, 1, 1], 'int32')
    y_pred  = keras.ops.convert_to_tensor([0.1, 0.2, 0.8, 0.9], 'float32')
    metric.update_state(y_true, y_pred)
    res = metric.result()

    assert float(keras.ops.convert_to_numpy(res['eer'])) < 0.1
    assert 0.9 < float(keras.ops.convert_to_numpy(res['auc'])) <= 1.0
