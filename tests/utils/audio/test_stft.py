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

""" Pure-``pytest`` tests for :mod:`utils.audio.stft`.

Parametrized over the concrete ``MelSTFT`` subclasses (``_mel_classes`` minus the
abstract base). Each class is exposed by the ``stft_class`` fixture.

Marker notes :
  - construction + ``save`` / ``load_from_file`` / ``get_config`` only touch numpy
    / librosa / scipy (``graph_compile`` merely *wraps* ``mel_spectrogram`` at
    decoration time) -> **keras-free**, left unmarked.
  - actually computing a mel (``mel_fn(audio)`` / ``load_mel``) goes through
    ``ops.conv1d`` / ``atan2`` / ``matmul`` which have no numpy path -> ``keras``.
  - graph / XLA compatibility of the mel computation -> ``tensorflow`` (the mel
    reproducibility golden uses ``max_err = 2e-3``, so accumulation is sensitive ;
    eager-vs-graph should match on the same device but stays behind the marker).
"""

import os
import math

import pytest

from utils.audio import load_audio, load_mel
from utils.audio.stft import MelSTFT, _mel_classes


_STFT_CLASSES = [cls for name, cls in _mel_classes.items() if name != 'MelSTFT']

@pytest.fixture(params = _STFT_CLASSES, ids = lambda cls: cls.__name__)
def stft_class(request):
    return request.param


# --- config save / load round-trip (keras-free) ------------------------------------

def test_stft_save_load_roundtrip(stft_class, temp_dir, asserts):
    mel_fn = stft_class()
    path   = os.path.join(temp_dir, '{}.json'.format(stft_class.__name__))

    mel_fn.save(path)
    assert os.path.exists(path), 'The saving failed !'

    asserts.assert_equal(mel_fn, MelSTFT.load_from_file(path))


# --- length helpers (pure arithmetic) ----------------------------------------------

def test_stft_lengths(stft_class):
    mel_fn = stft_class()

    audio_length = mel_fn.rate  # 1 second of audio
    mel_length   = mel_fn.get_mel_length(audio_length)

    assert mel_length == int(math.ceil(
        max(mel_fn.filter_length, audio_length) / mel_fn.hop_length
    ))
    assert mel_fn.get_audio_length(mel_length) == mel_length * mel_fn.hop_length


# --- mel computation (requires keras ops) ------------------------------------------

@pytest.mark.keras
def test_stft_mel_matches_load_mel(stft_class, audio_path, asserts):
    mel_fn = stft_class()

    asserts.assert_equal(
        mel_fn(load_audio(audio_path, mel_fn.rate))[0], load_mel(audio_path, mel_fn)
    )

@pytest.mark.keras
def test_stft_mel_matches_load_mel_trimmed(stft_class, audio_path, asserts):
    mel_fn = stft_class()

    asserts.assert_equal(
        mel_fn(load_audio(audio_path, mel_fn.rate, trim_silence = True))[0],
        load_mel(audio_path, mel_fn, trim_silence = True)
    )

@pytest.mark.keras
def test_stft_mel_reproducible(stft_class, audio_path, asserts):
    mel_fn = stft_class()

    asserts.assert_reproducible(
        load_mel(audio_path, mel_fn),
        'stft-{}.npy'.format(stft_class.__name__),
        max_err = 2e-3
    )


# --- graph compatibility (requires tensorflow) -------------------------------------

@pytest.mark.tensorflow
def test_stft_mel_graph_compatible(stft_class, audio_path, asserts):
    mel_fn = stft_class()
    audio  = load_audio(audio_path, mel_fn.rate)

    asserts.assert_graph_compatible(mel_fn, audio)
