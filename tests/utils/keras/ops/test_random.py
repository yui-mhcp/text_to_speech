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

""" Pure-``pytest`` tests for :mod:`utils.keras.ops.random`.

The random ops have ``numpy_fn is None`` (they always build a tensor) so the
value tests are inherently ``keras``-marked ; only output shapes are checked
(values are non-deterministic). Graph compatibility is ``tensorflow``-marked and
skipped on the ``torch`` backend, which does not support random ops in graph mode.
"""

import pytest

from utils.keras import ops

# (name, extra positional args) — all called as ``fn(shape, *args, seed=0)``
_CREATION = [
    pytest.param('beta',             (0., 1.),  id = 'beta'),
    pytest.param('binomial',         (10, 0.1), id = 'binomial'),
    pytest.param('gamma',            (0.5, ),   id = 'gamma'),
    pytest.param('normal',           (),        id = 'normal'),
    pytest.param('randint',          (5, 100),  id = 'randint'),
    pytest.param('truncated_normal', (),        id = 'truncated_normal'),
    pytest.param('uniform',          (),        id = 'uniform'),
]

# (name, target_shape, input_shape, extra args)
_MODIFICATION = [
    pytest.param('categorical', (4, 5), (4, 16), (5, ),  id = 'categorical'),
    pytest.param('dropout',     (5, 6), (5, 6),  (0.5, ), id = 'dropout'),
    pytest.param('shuffle',     (5, 6), (5, 6),  (0, ),  id = 'shuffle'),
]


def _skip_if_torch_graph():
    import keras
    if keras.backend.backend() == 'torch':
        pytest.skip('random ops are not usable in graph mode with the `torch` backend')


# --- creation ----------------------------------------------------------------------

@pytest.mark.keras
@pytest.mark.parametrize('name,args', _CREATION)
def test_random_creation(name, args):
    fn = getattr(ops.random, name)
    assert fn.numpy_fn is None, 'numpy should be disabled for random ops'

    assert tuple(fn((5, 6), * args, seed = 0).shape) == (5, 6)

@pytest.mark.tensorflow
@pytest.mark.parametrize('name,args', _CREATION)
def test_random_creation_graph(name, args, asserts):
    _skip_if_torch_graph()
    fn = getattr(ops.random, name)
    asserts.assert_graph_compatible(fn, (5, 6), * args, is_random = True, seed = 0)


# --- modification ------------------------------------------------------------------

@pytest.mark.keras
@pytest.mark.parametrize('name,target_shape,input_shape,args', _MODIFICATION)
def test_random_modification(name, target_shape, input_shape, args):
    import keras

    fn = getattr(ops.random, name)
    assert fn.numpy_fn is None, 'numpy should be disabled for random ops'

    inp = keras.random.normal(input_shape)
    assert tuple(fn(inp, * args, seed = 0).shape) == target_shape

@pytest.mark.tensorflow
@pytest.mark.parametrize('name,target_shape,input_shape,args', _MODIFICATION)
def test_random_modification_graph(name, target_shape, input_shape, args, asserts):
    import keras

    _skip_if_torch_graph()
    fn  = getattr(ops.random, name)
    inp = keras.random.normal(input_shape)
    asserts.assert_graph_compatible(fn, inp, * args, is_random = True, seed = 0)
