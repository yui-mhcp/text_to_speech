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

""" Pure-``pytest`` tests for :mod:`utils.audio.audio_processing`.

Plain ``test_*`` functions. Everything here is pure numpy (``utils.audio`` never
imports keras for these paths), so the whole module is **keras-free** and left
unmarked.

Two flavours :
  - reproducibility goldens driven through ``load_audio`` (the original
    ``TestAudioProcessing`` coverage : resample / reduce_noise / trim_silence) ;
  - direct unit tests of the building blocks (``normalize_audio``,
    ``convert_audio_dtype``, ``resample_audio``, ``trim_silence`` dispatch).
"""

import numpy as np
import pytest

from utils.audio import (
    load_audio, normalize_audio, convert_audio_dtype, resample_audio, trim_silence
)


# --- reproducibility goldens (through `load_audio`) --------------------------------

def test_load_audio_resample_reproducible(audio_path, asserts):
    asserts.assert_reproducible(load_audio(audio_path, rate = 22050), 'audio_resample.npy')

def test_load_audio_reduce_noise_reproducible(audio_path, asserts):
    asserts.assert_reproducible(
        load_audio(audio_path, rate = None, reduce_noise = True), 'audio_reduce_noise.npy'
    )

def test_load_audio_trim_silence_reproducible(audio_path, asserts):
    asserts.assert_reproducible(
        load_audio(audio_path, rate = None, trim_silence = True, method = 'window'),
        'audio_trim_silence.npy'
    )
    asserts.assert_reproducible(
        load_audio(audio_path, rate = None, trim_silence = True, method = 'window'),
        'audio_trim_silence-window.npy'
    )

def test_trim_silence_consistency(audio_path, audio, rate, asserts):
    """ `load_audio(..., trim_silence = True)` == manual `normalize -> trim`. """
    normalized = normalize_audio(audio, max_val = 1.)
    trimmed    = trim_silence(normalized, rate = rate, method = 'window')

    assert len(trimmed) < len(audio), \
        'trimmed audio ({}) should be shorter than original ({})'.format(len(trimmed), len(audio))

    loaded = load_audio(audio_path, rate, trim_silence = True, method = 'window')
    asserts.assert_equal(loaded, trimmed)


# --- `normalize_audio` -------------------------------------------------------------

def test_normalize_audio_float(asserts):
    """ `max_val = 1.` -> float32, zero-mean, unit peak. """
    a   = np.random.default_rng(0).normal(size = 16000).astype('float32')
    out = normalize_audio(a, max_val = 1.)

    assert out.dtype == np.float32
    asserts.assert_equal(0., float(np.mean(out)),        max_err = 1e-5)
    asserts.assert_equal(1., float(np.max(np.abs(out))), max_err = 1e-5)

def test_normalize_audio_int16():
    """ Default `max_val = 32767` -> int16 scaled so the peak hits the max value. """
    a   = np.random.default_rng(0).normal(size = 16000).astype('float32')
    out = normalize_audio(a)

    assert out.dtype == np.int16
    # the peak is scaled to `max_val` then truncated by `astype` -> within 1 LSB
    assert abs(int(np.max(np.abs(out))) - 32767) <= 1


# --- `convert_audio_dtype` ---------------------------------------------------------

def test_convert_audio_dtype_identity():
    """ Same dtype in / out -> the very same array is returned. """
    a = np.zeros(10, dtype = 'float32')
    assert convert_audio_dtype(a, np.float32) is a

def test_convert_audio_dtype_float_to_int(asserts):
    a   = np.linspace(-1., 1., 100).astype('float32')
    out = convert_audio_dtype(a, np.int16)

    assert out.dtype == np.int16
    asserts.assert_equal((a * np.iinfo(np.int16).max).astype('int16'), out)

def test_convert_audio_dtype_int_to_float(asserts):
    a   = (np.linspace(-1., 1., 100) * np.iinfo(np.int16).max).astype('int16')
    out = convert_audio_dtype(a, np.float32)

    assert out.dtype == np.float32
    asserts.assert_equal((a / np.iinfo(np.int16).max).astype(np.float32), out, max_err = 1e-4)


# --- `resample_audio` --------------------------------------------------------------

def test_resample_audio_noop(audio, rate):
    """ Resampling to the same rate is a no-op (same array, same rate). """
    out, out_rate = resample_audio(audio, rate, rate)
    assert out is audio
    assert out_rate == rate

def test_resample_audio_changes_length(audio, rate, asserts):
    target        = rate // 2
    out, out_rate = resample_audio(audio, rate, target)

    assert out_rate == target
    asserts.assert_equal(int(len(audio) / rate * target), len(out))


# --- `trim_silence` : every numpy dispatch keeps the 1D / no-longer invariant -------

@pytest.mark.parametrize('method', ['window', 'rms', 'remove', 'threshold'])
def test_trim_silence_methods(audio, rate, method):
    normalized = normalize_audio(audio, max_val = 1.)
    trimmed    = trim_silence(normalized, rate = rate, method = method)

    assert trimmed.ndim == 1
    assert len(trimmed) <= len(normalized)
