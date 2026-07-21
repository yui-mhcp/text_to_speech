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

""" Shared fixtures for the `utils.image` test-suite.

Kept dependency-light : `numpy` only at import time. `keras` / `PIL` are imported
lazily inside the fixtures so the keras-free (numpy) tests never pull a backend.
"""

import os

import numpy as np
import pytest


@pytest.fixture(scope = 'module')
def lena_path(data_dir):
    """ Path to the reference `lena.jpg`, skipping the test if it is missing. """
    path = os.path.join(data_dir, 'lena.jpg')
    if not os.path.exists(path):
        pytest.skip('{} does not exist'.format(path))
    return path

@pytest.fixture(scope = 'module')
def lena(lena_path):
    """ The reference image as a raw `np.uint8` `np.ndarray` (H, W, 3). """
    from PIL import Image
    return np.array(Image.open(lena_path))

@pytest.fixture
def fake_image():
    """ Deterministic (seeded) `float32` image, shape (512, 512, 3). """
    return np.random.default_rng(0).uniform(size = (512, 512, 3)).astype('float32')
