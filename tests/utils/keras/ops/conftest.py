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

""" Local fixtures for the ``utils.keras.ops`` test-suite.

The custom operations have (up to) two execution paths : a pure-``numpy`` one
(taken when every input is a ``np.ndarray`` / python scalar, so ``keras`` is
never imported) and a ``keras`` one (taken as soon as one input is a tensor).

The ``backend`` fixture drives the same test over both paths :
  - ``numpy``  : pure ``np.ndarray`` inputs -> the op must stay keras-free and
    return an ``ndarray``. Left **unmarked** (runs without keras).
  - ``tensor`` : ``keras`` tensor inputs -> marked ``keras`` (auto-skipped by the
    root ``conftest`` when keras is missing).
"""

import pytest

_BACKENDS = [
    pytest.param('numpy',  id = 'numpy'),
    pytest.param('tensor', id = 'tensor', marks = pytest.mark.keras),
]

@pytest.fixture(params = _BACKENDS)
def backend(request):
    """ Parametrized execution path : ``'numpy'`` (keras-free) or ``'tensor'``. """
    return request.param
