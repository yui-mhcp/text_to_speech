
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from utils.audio import *
from utils.audio.stft import _mel_classes

from unitest import Test, assert_function, assert_equal, assert_smaller, _out_test_dir

_filename = os.path.join('unitest', '__datas', 'audio_en.wav')

@Test
def test_audio_io():
    rate, audio = read_audio(_filename)
    
    assert_equal(16000, rate)
    assert_equal(64880, len(audio))
    assert_equal(audio, load_audio(_filename, rate))
    
    assert_function(load_audio, _filename, rate = 22050)
    assert_function(load_audio, _filename, rate = rate, reduce_noise = True)
    assert_function(load_audio, _filename, rate = rate, trim_silence = True)
    assert_function(load_audio, _filename, rate = rate, trim_silence = True, method = 'window')
    
    trimmed = trim_silence(audio, rate = rate, method = 'window')
    
    assert_smaller(len(audio), len(trimmed))
    
    loaded_trimmed = load_audio(_filename, rate, trim_silence = True, method = 'window')
    
    assert_equal(loaded_trimmed, trimmed)

    new_filename = os.path.join(_out_test_dir, os.path.basename(_filename))
    write_audio(audio, filename = new_filename, rate = rate, factor = 1.)

    assert_equal(audio, load_audio(new_filename, None), max_err = 1e-6)

@Test
def test_stft():
    for name, mel_class in _mel_classes.items():
        mel_fn = mel_class() if 'Jasper' not in name else mel_class(dither = 0.)

        audio = load_audio(_filename, mel_fn.rate)
        audio_trimmed = trim_silence(audio, rate = mel_fn.rate, method = 'window')

        assert_function(mel_fn, audio, name = 'test_{}'.format(name))
        assert_function(mel_fn, audio_trimmed, name = 'test_{}'.format(name))
    
        original      = mel_fn(audio)[0]
        trimmed       = mel_fn(audio_trimmed)[0]

        assert_equal(original, load_mel, _filename, mel_fn, name = 'test_load_{}'.format(name))
        assert_equal(trimmed, load_mel, _filename, mel_fn, trim_silence = True, method = 'window', name = 'test_load_{}'.format(name))
    
        stft_filename = os.path.join(_out_test_dir,  '{}_config.json'.format(name))

        mel_fn.save_to_file(stft_filename)
        restored = mel_class.load_from_file(stft_filename)

        assert_equal(mel_fn.get_config(), restored.get_config(), name = 'test_{}_config'.format(name))
