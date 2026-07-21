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

""" pytest configuration for the test-suite.

Responsibilities :
  - make the repository root importable (so `from utils import ...` works whatever
    the invocation directory) ;
  - turn the `--strict-golden` flag into the env var read by `tests.asserts` ;
  - auto-skip tests marked `tensorflow` / `gpu` when the requirement is missing ;
  - expose the standalone assertions (`tests.asserts`) as fixtures so tests can be
    written as plain functions without subclassing `CustomTestCase`.
"""

import os
import sys

import pytest

# --- make the repo root importable -------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests import asserts as _asserts
from tests._helpers import (
    data_dir as _data_dir,
    temp_dir as _temp_dir,
    reproductibility_dir as _reproductibility_dir,
    is_tensorflow_available,
    is_keras_available,
    is_torch_available,
    is_cv2_available,
)


# --- CLI options -------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        '--strict-golden',
        action  = 'store_true',
        default = False,
        help    = 'Fail (instead of silently creating) when a reproducibility golden is missing.'
    )

def pytest_configure(config):
    if config.getoption('--strict-golden'):
        os.environ[_asserts.STRICT_GOLDEN_ENV] = '1'


# --- automatic skipping ------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    skip_tf     = pytest.mark.skip(reason = '`tensorflow` is not available')
    skip_keras  = pytest.mark.skip(reason = '`keras` is not available')
    skip_torch  = pytest.mark.skip(reason = '`torch` is not available')
    skip_cv2    = pytest.mark.skip(reason = '`cv2` (opencv) is not available')
    for item in items:
        if item.get_closest_marker('tensorflow') and not is_tensorflow_available():
            item.add_marker(skip_tf)
        if item.get_closest_marker('keras') and not is_keras_available():
            item.add_marker(skip_keras)
        if item.get_closest_marker('torch') and not is_torch_available():
            item.add_marker(skip_torch)
        if item.get_closest_marker('cv2') and not is_cv2_available():
            item.add_marker(skip_cv2)


# --- fixtures ----------------------------------------------------------------------
@pytest.fixture(scope = 'session')
def data_dir():
    return _data_dir

@pytest.fixture(scope = 'session')
def temp_dir():
    return _temp_dir

@pytest.fixture(scope = 'session')
def reproductibility_dir():
    return _reproductibility_dir

@pytest.fixture(scope = 'session')
def tf_available():
    return is_tensorflow_available()

@pytest.fixture(scope = 'session')
def keras_available():
    return is_keras_available()

@pytest.fixture(scope = 'session')
def torch_available():
    return is_torch_available()

@pytest.fixture(scope = 'session')
def cv2_available():
    return is_cv2_available()

@pytest.fixture(scope = 'session')
def asserts():
    """ The `tests.asserts` module — use e.g. `asserts.assert_equal(a, b)` in plain tests. """
    return _asserts
