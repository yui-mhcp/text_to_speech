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

""" Pure-``pytest`` tests for :mod:`utils.image.image_io`.

Plain ``test_*`` functions (no ``CustomTestCase`` / ``absl`` / ``unittest.skipIf``).
The reference image is provided by the ``lena`` / ``lena_path`` fixtures (see the
package ``conftest.py``), which skip the tests when ``data/lena.jpg`` is missing.

Marker notes :
  - ``get_image_size`` (filename / array / dict) and the metadata-only branches use
    no keras op -> unmarked.
  - building a `keras` tensor (``get_image_size`` tensor case) -> ``keras``.
  - ``load_image`` with ``channels = 1`` (``ops.rgb_to_grayscale``) or an actual
    resize (``ops.image_resize``) hits ops that have **no numpy path** -> ``keras``.
  - ``save_image`` needs ``cv2`` (``cv2.imwrite``) -> ``cv2``.

Streaming / video helpers (`stream_camera`, `frame_generator`, `build_gif`,
`build_sprite`, ...) require a camera / extra deps and are intentionally left out.
"""

import os

import numpy as np
import pytest

from utils.image import load_image, save_image, get_image_size, convert_to_uint8


# --- `get_image_size` --------------------------------------------------------------

def test_get_image_size_filename(lena, lena_path, asserts):
    asserts.assert_equal(lena.shape[:2], get_image_size(lena_path))

def test_get_image_size_array(lena, asserts):
    asserts.assert_equal(lena.shape[:2], get_image_size(lena))

@pytest.mark.keras
def test_get_image_size_tensor(lena, asserts):
    import keras
    asserts.assert_equal(lena.shape[:2], get_image_size(keras.ops.convert_to_tensor(lena)))

def test_get_image_size_dict_height_width(asserts):
    asserts.assert_equal((12, 34), get_image_size({'height' : 12, 'width' : 34}))

def test_get_image_size_dict_filename(lena, lena_path, asserts):
    asserts.assert_equal(lena.shape[:2], get_image_size({'filename' : lena_path}))

def test_get_image_size_dict_image(lena, asserts):
    asserts.assert_equal(lena.shape[:2], get_image_size({'image' : lena}))


# --- `load_image` : the existing paths ---------------------------------------------

def test_load_image_filename(lena, lena_path, asserts):
    asserts.assert_equal(lena.shape, tuple(load_image(lena_path).shape))

def test_load_image_array(lena, asserts):
    asserts.assert_equal(lena.shape, tuple(load_image(lena).shape))

def test_load_image_array_identity(lena):
    """ A raw array, with no transformation requested, is returned untouched. """
    assert load_image(lena, to_tensor = False) is lena


# --- `load_image` : dict inputs ----------------------------------------------------

def test_load_image_dict_filename(lena, lena_path, asserts):
    asserts.assert_equal(lena.shape, tuple(load_image({'filename' : lena_path}).shape))

def test_load_image_dict_image(lena, asserts):
    asserts.assert_equal(lena, load_image({'image' : lena}, to_tensor = False))


# --- `load_image` : dtype conversion (numpy-castable, keras-free) ------------------

def test_load_image_dtype(lena, asserts):
    """ ``uint8 -> float32`` rescales to [0, 1] (``convert_data_dtype``). """
    out = load_image(lena, dtype = 'float32', to_tensor = False)
    asserts.assert_array(out)
    assert out.dtype == np.float32
    asserts.assert_equal(lena.astype('float32') / 255., out)


# --- `load_image` : channels / resize (require keras ops) --------------------------

@pytest.mark.keras
def test_load_image_grayscale(lena, asserts):
    gray = load_image(lena, channels = 1, to_tensor = False)
    asserts.assert_equal(lena.shape[:2], tuple(gray.shape[:2]))
    assert tuple(gray.shape)[-1] == 1

@pytest.mark.keras
def test_load_image_resize(lena, asserts):
    out = load_image(lena, size = (64, 64))
    asserts.assert_equal((64, 64, 3), tuple(out.shape))


# --- `convert_to_uint8` ------------------------------------------------------------

def test_convert_to_uint8_passthrough(lena, asserts):
    """ An already-``uint8`` image is returned unchanged (as a numpy array). """
    out = convert_to_uint8(lena)
    asserts.assert_array(out)
    assert out.dtype == np.uint8
    asserts.assert_equal(lena, out)

def test_convert_to_uint8_from_float(asserts):
    """ A float image in [0, 1] is rescaled to [0, 255] ``uint8``. """
    arr = np.linspace(0., 1., 12, dtype = 'float32').reshape(2, 2, 3)
    out = convert_to_uint8(arr)
    asserts.assert_array(out)
    assert out.dtype == np.uint8
    asserts.assert_equal((arr * 255).astype('uint8'), out)


# --- `save_image` round-trip (requires cv2) ----------------------------------------

@pytest.mark.cv2
def test_save_image_roundtrip(lena, temp_dir, asserts):
    # PNG is lossless, so a uint8 RGB image survives the save -> reload round-trip
    path = os.path.join(temp_dir, 'lena_roundtrip.png')
    returned = save_image(path, lena)

    assert returned == path
    assert os.path.exists(path)
    asserts.assert_equal(lena, load_image(path, to_tensor = False))
