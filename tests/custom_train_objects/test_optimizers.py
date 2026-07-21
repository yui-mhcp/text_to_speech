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

""" Pure-``pytest`` tests for :mod:`custom_train_objects.optimizers`.

Covers the ``get_optimizer`` / ``get_lr_scheduler`` factories (registry resolution
+ config / serialization round-trips, ported from the old ``test_learning_rate``)
and the custom learning-rate schedulers (``__call__`` values, ``clip`` bounds and
``get_config`` round-trips).

Everything here builds keras objects (variables / schedules) -> the whole module is
marked ``keras`` and skipped when the backend is missing.
"""

import pytest

keras = pytest.importorskip('keras')

from custom_train_objects.optimizers import (
    get_optimizer, get_lr_scheduler,
    DivideByStep, ReduceEvery, WarmupScheduler, SinScheduler, TanhDecayScheduler,
)
from ._helpers import normalize_object

pytestmark = pytest.mark.keras


def _step(value):
    return keras.ops.convert_to_tensor(value, 'int32')

def _value(scheduler, step):
    return float(keras.ops.convert_to_numpy(scheduler(_step(step))))


# --- get_optimizer : learning-rate resolution (ported test_learning_rate) ----------

def test_optimizer_scalar_learning_rate(asserts):
    lr = get_optimizer('adam', lr = 1.).learning_rate
    asserts.assert_equal(1.0, float(keras.ops.convert_to_numpy(lr)))

def test_optimizer_keeps_scheduler_instance():
    scheduler = DivideByStep(factor = 5)
    optimizer = get_optimizer('adam', lr = scheduler)
    assert optimizer._learning_rate is scheduler

def test_optimizer_scheduler_from_dict(asserts):
    scheduler = DivideByStep(factor = 5)
    optimizer = get_optimizer('adam', lr = {'name' : 'DivideByStep', ** scheduler.get_config()})
    asserts.assert_equal(normalize_object(scheduler), normalize_object(optimizer._learning_rate))

def test_optimizer_config_roundtrip(asserts):
    optimizer = get_optimizer('adam', lr = DivideByStep(factor = 5))
    rebuilt   = get_optimizer('adam', ** optimizer.get_config())
    asserts.assert_equal(normalize_object(optimizer), normalize_object(rebuilt))

def test_optimizer_serialization_roundtrip(asserts):
    optimizer = get_optimizer('adam', lr = DivideByStep(factor = 5))
    rebuilt   = get_optimizer(keras.optimizers.serialize(optimizer))
    asserts.assert_equal(normalize_object(optimizer), normalize_object(rebuilt))


# --- get_optimizer : registry smoke over a few keras optimizers --------------------

@pytest.mark.parametrize('name', ['adam', 'sgd', 'rmsprop'])
def test_optimizer_registry_roundtrip(name, asserts):
    optimizer = get_optimizer(name, lr = 1e-3)
    rebuilt   = get_optimizer(keras.optimizers.serialize(optimizer))
    asserts.assert_equal(normalize_object(optimizer), normalize_object(rebuilt))

def test_optimizer_instance_passthrough():
    optimizer = keras.optimizers.Adam()
    assert get_optimizer(optimizer) is optimizer


# --- get_lr_scheduler --------------------------------------------------------------

def test_get_lr_scheduler_from_string():
    assert isinstance(get_lr_scheduler('DivideByStep'), DivideByStep)

def test_get_lr_scheduler_passthrough():
    scheduler = DivideByStep(factor = 5)
    assert get_lr_scheduler(scheduler) is scheduler


# --- schedulers : config round-trip ------------------------------------------------

_SCHEDULERS = [
    pytest.param(DivideByStep(factor = 5), id = 'divide_by_step'),
    pytest.param(ReduceEvery(base = 1e-3, step = 10, factor = 0.1), id = 'reduce_every'),
    pytest.param(WarmupScheduler(factor = 8, warmup_steps = 1000), id = 'warmup'),
    pytest.param(SinScheduler(period = 512, with_decay = True), id = 'sin'),
    pytest.param(TanhDecayScheduler(period = 1024), id = 'tanh_decay'),
]

@pytest.mark.parametrize('scheduler', _SCHEDULERS)
def test_scheduler_config_roundtrip(scheduler, asserts):
    rebuilt = get_lr_scheduler({'name' : type(scheduler).__name__, ** scheduler.get_config()})
    assert isinstance(rebuilt, type(scheduler))
    asserts.assert_equal(normalize_object(scheduler), normalize_object(rebuilt))


# --- schedulers : numeric behaviour ------------------------------------------------

@pytest.mark.parametrize('step,target', [
    pytest.param(1000, 0.005, id = 'in_range'),    # 5 / 1000
    pytest.param(1,    0.01,  id = 'clip_high'),    # 5 / 1   -> maxval
    pytest.param(10 ** 6, 1e-5, id = 'clip_low'),   # 5 / 1e6 -> minval
])
def test_divide_by_step_values(step, target, asserts):
    asserts.assert_equal(target, _value(DivideByStep(factor = 5), step), max_err = 1e-7)

@pytest.mark.parametrize('scheduler', _SCHEDULERS)
def test_scheduler_call_is_finite_positive(scheduler):
    """ Every scheduler must return a finite, positive scalar.

        Notably a regression for ``SinScheduler`` whose ``__init__`` used to raise
        ``NameError`` (``self.range = maxval - minval`` -> undefined names).
    """
    import math
    value = _value(scheduler, 100)
    assert math.isfinite(value)
    assert value > 0.
