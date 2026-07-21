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

""" Shared fixtures for the `utils.audio` test-suite.

Kept dependency-light : only `os` / `numpy` / `scipy` at import time (`keras` /
`tensorflow` stay out so the numpy tests never pull a backend). The reference
audio is `data/audio_test.wav`, read once per module ; every fixture that needs
it depends on `audio_path`, which `pytest.skip`s the test when the file is
missing (mirrors the `lena` / `lena_path` pattern of the image suite).
"""

import os

import pytest


@pytest.fixture(scope = 'module')
def audio_path(data_dir):
    """ Path to the reference `audio_test.wav`, skipping the test if it is missing. """
    path = os.path.join(data_dir, 'audio_test.wav')
    if not os.path.exists(path):
        pytest.skip('{} does not exist'.format(path))
    return path

@pytest.fixture(scope = 'module')
def _wav(audio_path):
    """ The raw `(rate, audio)` pair, read once via `scipy.io.wavfile.read`. """
    from scipy.io.wavfile import read
    return read(audio_path)

@pytest.fixture
def rate(_wav):
    """ The native sampling rate of the reference audio (`int`). """
    return int(_wav[0])

@pytest.fixture
def audio(_wav):
    """ The raw reference audio as a 1D `np.ndarray` (native rate, `int16`). """
    return _wav[1]
