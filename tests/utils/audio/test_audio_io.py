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

""" Pure-``pytest`` tests for :mod:`utils.audio.audio_io`.

Plain ``test_*`` functions (no ``CustomTestCase`` / ``absl`` / ``unittest.skipIf``).
The reference audio comes from the ``audio_path`` / ``audio`` / ``rate`` fixtures
(see the package ``conftest.py``), which skip when ``data/audio_test.wav`` is
missing.

Marker notes :
  - ``read_audio`` is ``execute_eagerly(numpy = True)`` and every processing step
    (`normalize_audio`, `resample_audio`, ...) is pure numpy, so the functional
    cases run **without keras** and are left unmarked.
  - graph / XLA is exercised through ``asserts.assert_graph_compatible`` behind the
    ``tensorflow`` marker (auto-skipped when TF is missing) : ``read_audio`` is
    explicitly built to stay callable inside a ``tf.function``.

Non-wav formats (mp3 / flac via pydub / librosa, video via ffmpeg) need optional
dependencies and extra data files -> intentionally left out for this pass.
"""

import os

from types import SimpleNamespace

import numpy as np
import pytest

from utils.audio import load_audio, read_audio, write_audio, load_mel, resample_file


# --- `load_audio` : the input variants (faithful port of the original cases) -------

@pytest.mark.parametrize('case', ['file', 'dict_file', 'resampled', 'array', 'audio_dict'])
def test_load_audio(case, audio_path, audio, rate, asserts):
    if case == 'file':
        data = audio_path
    elif case == 'dict_file':
        data = {'filename' : audio_path, 'text' : 'Hello World !'}
    elif case == 'resampled':
        # the real file is exposed under the rate-specific key, so the (fake)
        # `filename` is never read -> no resampling happens.
        data = {
            'filename'              : audio_path.replace('.wav', '-fake.wav'),
            'wavs_{}'.format(rate)  : audio_path,
            'text'                  : 'Hello World !'
        }
    elif case == 'array':
        data = audio
    elif case == 'audio_dict':
        data = {'audio' : audio, 'rate' : rate, 'text' : 'Hello World !'}

    asserts.assert_equal(audio, load_audio(data, rate, normalize = False))


# --- `read_audio` ------------------------------------------------------------------

def test_read_audio(audio_path, audio, rate, asserts):
    out_rate, out_audio = read_audio(audio_path, normalize = False)
    asserts.assert_equal(rate, out_rate)
    asserts.assert_equal(audio, out_audio)


# --- `write_audio` round-trip ------------------------------------------------------

def test_write_audio_roundtrip(audio, rate, temp_dir, asserts):
    path = os.path.join(temp_dir, 'audio_roundtrip.wav')
    returned = write_audio(path, audio, rate, normalize = False)

    assert returned == path
    assert os.path.exists(path)
    asserts.assert_equal(audio, load_audio(path, rate, normalize = False))


# --- `resample_file` round-trip ----------------------------------------------------

def test_resample_file_roundtrip(audio_path, rate, temp_dir):
    target  = rate // 2
    out     = os.path.join(temp_dir, 'audio_resampled.wav')

    result = resample_file(audio_path, target, filename_out = out)

    assert result == out
    assert os.path.exists(out)
    assert read_audio(out, normalize = False)[0] == target


# --- `load_mel` : pass-through branches (keras-free) -------------------------------

def test_load_mel_dict_passthrough():
    """ A dict already carrying a `mel` is returned untouched (stft_fn unused). """
    mel = np.zeros((10, 80), dtype = 'float32')
    assert load_mel({'mel' : mel}, stft_fn = None) is mel

def test_load_mel_array_passthrough():
    """ A 2D array whose last dim matches `n_mel_channels` is treated as a mel. """
    mel     = np.zeros((10, 80), dtype = 'float32')
    stft_fn = SimpleNamespace(n_mel_channels = 80)
    assert load_mel(mel, stft_fn) is mel


# --- graph compatibility (requires tensorflow) -------------------------------------

@pytest.mark.tensorflow
def test_read_audio_graph_compatible(audio, rate, asserts):
    # raw-audio path : no file IO inside the graph, just the numpy processing
    # pipeline wrapped by `execute_eagerly` -> must run inside a `tf.function`.
    asserts.assert_graph_compatible(read_audio, audio, rate = rate)
