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

""" Pure-``pytest`` tests for :mod:`utils.image.image_processing`.

Plain ``test_*`` functions + ``@pytest.mark.parametrize`` (no ``CustomTestCase`` /
``absl.parameterized``). Tolerant comparisons go through the ``asserts`` fixture.

``get_output_size`` / ``resize_image`` / ``pad_image`` run on pure ``np.ndarray``
inputs : ``utils.keras.ops`` dispatches numpy without importing keras, so these
stay keras-free (unmarked). Graph / XLA compatibility is split into dedicated
``tensorflow``-marked tests. ``rotate_image`` builds a ``keras`` layer -> marked
``keras``.
"""

import numpy as np
import pytest

from utils.image import get_output_size, resize_image, pad_image
from utils.image.image_processing import rotate_image


# --- `get_output_size` : value tests (numpy, keras-free) ---------------------------

@pytest.mark.parametrize('size,target', [
    (None,         None),
    ((256, 256),   (256, 256)),
    ((256, None),  (256, 512)),
    ((None, 256),  (512, 256)),
    ((None, None), (512, 512)),
])
def test_get_output_size_simple(fake_image, asserts, size, target):
    out_size = get_output_size(fake_image, size)
    if target is None:
        assert out_size is None
        return

    asserts.assert_equal(target, out_size)
    # fully-specified size -> returned as-is (a `tuple`); a partial / missing dim
    # forces a computed `np.ndarray`
    if size is not None and not any(s is None for s in size):
        assert isinstance(out_size, tuple), 'Got {}'.format(type(out_size))
    else:
        asserts.assert_array(out_size)

@pytest.mark.parametrize('size,multiples,target', [
    (None,        64,  (512, 512)),
    (None,        100, (600, 600)),
    ((256, None), 100, (256, 600)),
    ((None, 64),  100, (600, 64)),
    ((256, 256),  100, (256, 256)),
])
def test_get_output_size_multiples(fake_image, asserts, size, multiples, target):
    out_size = get_output_size(fake_image, size, multiples = multiples)
    asserts.assert_equal(target, out_size)
    if size and all(s for s in size):
        assert isinstance(out_size, tuple)
    else:
        asserts.assert_array(out_size)

@pytest.mark.parametrize('size,multiples,target', [
    (None,        None, None),
    ((256, 256),  None, (256, 256)),

    ((256, None), None, (256, 256)),
    ((None, 500), None, (500, 500)),

    (None,        64,   (512, 512)),
    (None,        100,  (600, 600)),
    ((256, None), 100,  (256, 300)),
    ((None, 64),  100,  (100, 64)),
])
def test_get_output_size_preserve_ratio(fake_image, asserts, size, multiples, target):
    out_size = get_output_size(
        fake_image, size, multiples = multiples, preserve_aspect_ratio = True
    )
    if target is None:
        assert out_size is None
    else:
        asserts.assert_equal(target, out_size)


# --- `resize_image` ----------------------------------------------------------------

def test_resize_identity(fake_image):
    """ A no-op resize must return the very same object (no copy). """
    assert resize_image(fake_image) is fake_image
    assert resize_image(fake_image, fake_image.shape[:2]) is fake_image
    assert resize_image(fake_image, multiples = 16) is fake_image
    assert resize_image(fake_image, max_shape = 1024) is fake_image

@pytest.mark.keras   # an actual resize calls `ops.image_resize`, which has no numpy path
def test_resize_simple(fake_image, asserts):
    asserts.assert_equal((256, 256, 3), tuple(resize_image(fake_image, (256, 256)).shape))
    asserts.assert_equal((256, 512, 3), tuple(resize_image(fake_image, (256, None)).shape))

@pytest.mark.keras
def test_resize_kwargs(fake_image, asserts):
    asserts.assert_equal((256, 256, 3), tuple(resize_image(
        fake_image, (256, None), preserve_aspect_ratio = True
    ).shape))
    asserts.assert_equal((256, 256, 3), tuple(resize_image(
        fake_image, max_shape = 256
    ).shape))
    asserts.assert_equal((600, 600, 3), tuple(resize_image(
        fake_image, multiples = 100
    ).shape))

@pytest.mark.keras
@pytest.mark.parametrize('pad_value', [0., 1.])
def test_resize_and_pad(fake_image, asserts, pad_value):
    padded = resize_image(
        fake_image, (256, 512), preserve_aspect_ratio = True, pad_value = pad_value
    )
    asserts.assert_equal((256, 512, 3), tuple(padded.shape))
    asserts.assert_equal(np.full((256, 256, 3), pad_value, 'float32'), padded[:, 256 :])


# --- `pad_image` (pure numpy) ------------------------------------------------------

@pytest.fixture
def small_image():
    """ A (2, 2, 3) image of ones, so padding regions (zeros) are distinguishable. """
    return np.ones((2, 2, 3), dtype = 'float32')

def test_pad_after(small_image, asserts):
    padded = pad_image(small_image, (4, 4), pad_mode = 'after')
    asserts.assert_equal((4, 4, 3), tuple(padded.shape))
    asserts.assert_equal(small_image, padded[:2, :2])           # original is top-left
    asserts.assert_equal(np.zeros((2, 4, 3), 'float32'), padded[2:])     # bottom rows padded
    asserts.assert_equal(np.zeros((4, 2, 3), 'float32'), padded[:, 2:])  # right cols padded

def test_pad_before(small_image, asserts):
    padded = pad_image(small_image, (4, 4), pad_mode = 'before')
    asserts.assert_equal((4, 4, 3), tuple(padded.shape))
    asserts.assert_equal(small_image, padded[2:, 2:])           # original is bottom-right
    asserts.assert_equal(np.zeros((2, 4, 3), 'float32'), padded[:2])

def test_pad_even(small_image, asserts):
    padded = pad_image(small_image, (4, 4), pad_mode = 'even')
    asserts.assert_equal((4, 4, 3), tuple(padded.shape))
    asserts.assert_equal(small_image, padded[1:3, 1:3])         # original is centered

def test_pad_repeat_last(asserts):
    # columns valued 0, 1, 2 along the width so the repeated (last) one is identifiable
    col    = np.arange(3, dtype = 'float32')
    image  = np.broadcast_to(col[None, :, None], (3, 3, 1)).copy()
    padded = pad_image(image, (3, 5), pad_mode = 'repeat_last')
    asserts.assert_equal((3, 5, 1), tuple(padded.shape))
    asserts.assert_equal(image, padded[:, :3])              # original preserved
    # appended columns (3, 4) repeat the last original column (value 2, not 0)
    asserts.assert_equal(np.full((3, 1), 2., 'float32'), padded[:, 3])
    asserts.assert_equal(np.full((3, 1), 2., 'float32'), padded[:, 4])

def test_pad_value(small_image, asserts):
    padded = pad_image(small_image, (4, 4), pad_mode = 'after', pad_value = 5.)
    asserts.assert_equal(np.full((2, 4, 3), 5., 'float32'), padded[2:])

def test_pad_noop_returns_same(small_image):
    """ Padding to a size already reached is a no-op (returns the same object). """
    assert pad_image(small_image, (2, 2)) is small_image
    assert pad_image(small_image, (1, 1)) is small_image  # smaller target -> no padding


# --- `get_output_size` : graph / XLA compatibility (requires tensorflow) -----------

# only the cases going through the ops (partial / computed sizes -> `np.ndarray`,
# which becomes a tensor in graph mode) are checked : a fully-specified `(h, w)`
# returns a plain python tuple early and is not an interesting graph case.
@pytest.mark.tensorflow
@pytest.mark.parametrize('size', [(256, None), (None, 256)])
def test_get_output_size_graph_compatible(fake_image, asserts, size):
    asserts.assert_graph_compatible(get_output_size, fake_image, size)

@pytest.mark.tensorflow
def test_get_output_size_multiples_graph_compatible(fake_image, asserts):
    asserts.assert_graph_compatible(get_output_size, fake_image, None, multiples = 100)


# --- `rotate_image` (requires keras) -----------------------------------------------

@pytest.mark.keras
@pytest.mark.parametrize('angle', [0., 45., 90.])
def test_rotate_image_preserves_shape(angle, asserts):
    image   = np.random.default_rng(0).uniform(size = (16, 16, 3)).astype('float32')
    rotated = rotate_image(image, angle)
    asserts.assert_equal(image.shape, tuple(rotated.shape))

@pytest.mark.keras
def test_rotate_image_zero_angle_is_identity(asserts):
    image   = np.random.default_rng(0).uniform(size = (16, 16, 3)).astype('float32')
    rotated = rotate_image(image, 0.)
    asserts.assert_equal(image, rotated, max_err = 1e-5)
