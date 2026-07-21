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

""" Fixtures isolating the `models.core` tests from global / on-disk state.

All three fixtures are **autouse** : every test under `tests/models/core/` runs
hermetically, without touching the real `pretrained_models/` directory, leaking
instances through the singleton cache, or needing a real runtime.
"""

import pytest

from models.core.base_model import ModelInstances
from models.core.utils.saving import set_saving_dir
from utils.keras.runtimes import _runtimes

from tests.models.core._helpers import FakeRuntime


@pytest.fixture(autouse = True)
def isolated_saving_dir(tmp_path):
    """ Redirect the model root to an isolated temp dir so no test writes to `pretrained_models/`.

        Yields the (str) root so tests can assert on the derived `directory` / `config_file` paths.
    """
    previous = set_saving_dir(str(tmp_path))
    try:
        yield str(tmp_path)
    finally:
        set_saving_dir(previous)


@pytest.fixture(autouse = True)
def clean_registry():
    """ Clear the `ModelInstances` singleton cache around every test.

        The cache (`ModelInstances._instances`) is keyed by `name` and **shared across every**
        `BaseModel` subclass, so without this a name reused between tests would return a stale
        instance (and the `class_name`-mismatch guard could fire spuriously).
    """
    ModelInstances._instances.clear()
    try:
        yield
    finally:
        ModelInstances._instances.clear()


@pytest.fixture(autouse = True)
def register_fake_runtime():
    """ Expose `FakeRuntime` under the `'fake'` runtime key.

        This lets the *inference-only* code path of `BaseModel` (`runtime != 'keras'` ->
        `build_runtime(runtime, ...)`, keras-free `save`) be exercised without a real engine.
    """
    _runtimes['fake'] = FakeRuntime
    try:
        yield
    finally:
        _runtimes.pop('fake', None)
