
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

from utils.audio import load_audio, read_audio, write_audio, trim_silence

from unitest import Test, assert_function, assert_equal, assert_smaller

_filename = os.path.join('test', '__datas', 'audio_en.wav')

@Test
def test_audio_io():
    rate, audio = read_audio(_filename)
    
    assert_equal(16000, rate)
    assert_equal(64880, len(audio))
    
    trimmed = trim_silence(audio, rate = rate, method = 'window')
    
    assert_smaller(len(audio), len(trimmed))
    
    loaded_trimmed = load_audio(_filename, rate, trim_silence = True, method = 'window')
    
    assert_equal(loaded_trimmed, trimmed)

    new_filename = _filename.replace('.wav', '_test.wav')
    write_audio(audio, filename = new_filename, rate = rate, factor = 1.)

    assert_equal(audio, load_audio(new_filename, None), max_err = 1e-3)

@Test
def test_stft():
    from utils.audio import TacotronSTFT, load_audio, load_mel
    from utils.audio import trim_silence, reduce_noise

    stft = TacotronSTFT()

    audio = load_audio(_filename, stft.rate)
    audio_trimmed = trim_silence(audio, method = 'window')
    
    assert_function(stft, audio)
    assert_function(stft, audio_trimmed)
    
    original      = stft(audio)[0]
    trimmed       = stft(audio_trimmed)[0]
    
    assert_equal(original, load_mel, _filename, stft)
    assert_equal(trimmed, load_mel, _filename, stft, trim_silence = True)
    
    stft_filename = os.path.join('test', '__outputs',  'stft_config.json')

    stft.save_to_file(stft_filename)
    restored = TacotronSTFT.load_from_file(stft_filename)
    
    assert_equal(stft.get_config(), restored.get_config())
