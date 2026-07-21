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

""" Pure-``pytest`` tests for :mod:`utils.image.image_normalization`.

Plain ``test_*`` functions + ``@pytest.mark.parametrize`` (no ``CustomTestCase`` /
``absl.parameterized``). The ``backend`` fixture mirrors the old
``@named_parameters(('array', ...), ('tensor', ...))`` class decorator :

  - ``numpy``  : pure ``np.ndarray`` input. ``utils.keras.ops`` dispatches numpy
    without importing keras, so reference targets (computed via ``utils.keras.ops``)
    stay keras-free and these cases are left unmarked.
  - ``tensor`` : a ``keras`` tensor -> marked ``keras`` (lazily imported).

Reference targets are computed in plain numpy from the shared seeded array, and
compared through the tolerant ``asserts`` fixture (so ``tensor`` outputs compare
fine against the numpy reference).
"""

import numpy as np
import pytest

from utils.image import get_image_normalization_fn
from utils.image.image_normalization import (
    _clip_means, _clip_std, _east_means, _east_std
)


# --- backend fixture ---------------------------------------------------------------

_BACKENDS = [
    pytest.param('numpy',  id = 'numpy'),
    pytest.param('tensor', id = 'tensor', marks = pytest.mark.keras),
]

@pytest.fixture(scope = 'module')
def norm_arr():
    """ Shared seeded reference image, ``float32`` (256, 256, 3). """
    return np.random.default_rng(0).uniform(size = (256, 256, 3)).astype('float32')

@pytest.fixture(params = _BACKENDS)
def norm_image(request, norm_arr):
    if request.param == 'numpy':
        return norm_arr
    import keras
    return keras.ops.convert_to_tensor(norm_arr)


# --- reference (numpy) implementations ---------------------------------------------

_VGG_MEANS = np.array([103.939, 116.779, 123.68])[None, None].astype('float32')

def _ref_01(a):
    t = a - a.min()
    return t / max(1e-3, t.max())

def _ref_normal(a):
    return (a - a.mean()) / a.std()

def _ref_mean_std(a, means, std):
    means = np.reshape(means, [-1]).astype('float32')[None, None]
    std   = np.reshape(std,   [-1]).astype('float32')[None, None]
    return (a - means) / std

_SIMPLE_TARGETS = {
    '01'        : _ref_01,
    'normal'    : _ref_normal,
    'tanh'      : lambda a: a * 2. - 1.,
    'mobilenet' : lambda a: a / 127.5 - 1.,
    'vgg'       : lambda a: a[..., ::-1] - _VGG_MEANS,
    'vgg16'     : lambda a: a[..., ::-1] - _VGG_MEANS,
    'vgg19'     : lambda a: a[..., ::-1] - _VGG_MEANS,
}


# --- shared assertion helper -------------------------------------------------------

def _check(norm_image, asserts, method, target):
    normalized = get_image_normalization_fn(method)(norm_image)
    asserts.assert_equal(target, normalized)
    if isinstance(norm_image, np.ndarray):  asserts.assert_array(normalized)
    else:                                   asserts.assert_tensor(normalized)


# --- value tests : the "simple" (no mean/std table) styles -------------------------

@pytest.mark.parametrize('method', list(_SIMPLE_TARGETS))
def test_normalization_simple(norm_image, norm_arr, asserts, method):
    _check(norm_image, asserts, method, _SIMPLE_TARGETS[method](norm_arr))


# --- value tests : the mean/std styles ---------------------------------------------

def test_normalization_clip(norm_image, norm_arr, asserts):
    _check(norm_image, asserts, 'clip', _ref_mean_std(norm_arr, _clip_means, _clip_std))

def test_normalization_east(norm_image, norm_arr, asserts):
    _check(norm_image, asserts, 'east', _ref_mean_std(norm_arr, _east_means, _east_std))

def test_normalization_easyocr(norm_image, norm_arr, asserts):
    # NB: the legacy test mistakenly passed `'east'` here ; `easyocr` is mean/std 0.5
    _check(norm_image, asserts, 'easyocr', _ref_mean_std(norm_arr, 0.5, 0.5))


# --- `vggface` : note it broadcasts to a 4-D output (reshape to [1, 1, 1, 3]) -------

def test_normalization_vggface(norm_image, norm_arr, asserts):
    vggface_vals = np.array([91.4953, 103.8827, 131.0912])
    target = norm_arr[..., ::-1] / 255. - np.reshape(vggface_vals, [1, 1, 1, 3]) / 255.
    target = target.astype(np.float32)
    _check(norm_image, asserts, 'vggface', target)


# --- agreement with `keras.applications.*.preprocess_input` (requires keras) -------

@pytest.mark.keras
@pytest.mark.parametrize('model', ['vgg16', 'vgg19', 'mobilenet'])
def test_normalization_matches_keras_applications(norm_image, asserts, model):
    import keras

    copy   = norm_image.copy() if isinstance(norm_image, np.ndarray) else norm_image
    target = getattr(keras.applications, model).preprocess_input(copy)

    normalized = get_image_normalization_fn(model)(norm_image)
    asserts.assert_equal(target, normalized)
    if isinstance(norm_image, np.ndarray):  asserts.assert_array(normalized)
    else:                                   asserts.assert_tensor(normalized)


# --- `get_image_normalization_fn` : resolution logic (numpy, keras-free) -----------

@pytest.mark.parametrize('method', [None, 'identity'])
def test_resolution_identity_is_none(method):
    assert get_image_normalization_fn(method) is None

def test_resolution_callable_passthrough():
    fn = lambda image: image
    assert get_image_normalization_fn(fn) is fn

def test_resolution_dict(norm_arr, asserts):
    spec = {'means' : [0.1, 0.2, 0.3], 'std' : [1., 2., 4.]}
    fn   = get_image_normalization_fn(spec)
    asserts.assert_equal(_ref_mean_std(norm_arr, spec['means'], spec['std']), fn(norm_arr))

@pytest.mark.parametrize('container', [list, tuple])
def test_resolution_sequence(norm_arr, asserts, container):
    means, std = [0.1, 0.2, 0.3], [1., 2., 4.]
    fn = get_image_normalization_fn(container([means, std]))
    asserts.assert_equal(_ref_mean_std(norm_arr, means, std), fn(norm_arr))

def test_resolution_unknown_raises():
    with pytest.raises(ValueError):
        get_image_normalization_fn('does_not_exist')


# --- graph / XLA compatibility (requires tensorflow) -------------------------------

@pytest.mark.tensorflow
@pytest.mark.parametrize('method', ['01', 'normal', 'tanh', 'mobilenet', 'clip', 'east'])
def test_normalization_graph_compatible(norm_arr, asserts, method):
    asserts.assert_graph_compatible(get_image_normalization_fn(method), norm_arr)
