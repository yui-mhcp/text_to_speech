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

import os
import unittest

from absl.testing import parameterized
from scipy.io.wavfile import read

from utils.audio import *
from utils.audio.stft import _mel_classes
from . import CustomTestCase, data_dir, reproductibility_dir

filename = os.path.join(data_dir, 'audio_test.wav')
_rate, _audio = None, None
if os.path.exists(filename):
    _rate, _audio = read(filename)

@unittest.skipIf(not os.path.exists(filename), '{} does not exist'.format(filename))
class TestAudioIO(CustomTestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ('file',        filename),
        ('dict_file',   {'filename' : filename, 'text' : 'Hello World !'}),
        ('resampled',   {
            'filename'  : filename.replace('.wav', '-fake.wav'),
            'wavs_{}'.format(_rate) : filename,
            'text'      : 'Hello World !'
        }),
        ('audio',       _audio),
        ('audio_dict', {'audio' : _audio, 'rate' : _rate, 'text' : 'Hello World !'})
    )
    def test_loading(self, data):
        self.assertEqual(_audio, load_audio(data, _rate, normalize = False))
    
    def test_read(self):
        rate, audio = read_audio(filename, normalize = False)
        
        self.assertEqual(_rate, rate)
        self.assertEqual(_audio, audio)

    def test_saving(self):
        try:
            save_file = filename.replace('.wav', '-tmp.wav')
            write_audio(save_file, _audio, _rate, normalize = False)
            
            self.assertTrue(os.path.exists(save_file))
            self.assertEqual(_audio, load_audio(save_file, _rate, normalize = False))
        finally:
            if os.path.exists(save_file): os.remove(save_file)

@unittest.skipIf(not os.path.exists(filename), '{} does not exist'.format(filename))
class TestAudioProcessing(CustomTestCase):
    def test_processing(self):
        self.assertReproductible(
            load_audio(filename, rate = 22050), 'audio_resample.npy'
        )
        self.assertReproductible(
            load_audio(filename, rate = None, reduce_noise = True), 'audio_reduce_noise.npy'
        )
        self.assertReproductible(
            load_audio(filename, rate = None, trim_silence = True, method = 'window'),
            'audio_trim_silence.npy'
        )
        self.assertReproductible(
            load_audio(filename, rate = None, trim_silence = True, method = 'window'),
            'audio_trim_silence-window.npy'
        )

        normalized = normalize_audio(_audio, max_val = 1.)
        trimmed = trim_silence(normalized, rate = _rate, method = 'window')
        self.assertTrue(len(_audio) > len(trimmed), 'trimmed audio ({}) should be shorter than original ({}) length'.format(len(trimmed), len(_audio)))

        loaded_trimmed = load_audio(filename, _rate, trim_silence = True, method = 'window')
        self.assertEqual(loaded_trimmed, trimmed)

@unittest.skipIf(not os.path.exists(filename), '{} does not exist'.format(filename))
class TestSTFT(CustomTestCase, parameterized.TestCase):
    @parameterized.named_parameters([
        (name.lower()[:-4], name) for name in _mel_classes.keys() if name != 'MelSTFT'
    ])
    def test_stft(self, name):
        mel_fn  = _mel_classes[name]()

        try:
            dump_file = os.path.join(reproductibility_dir, '{}.json'.format(name))
            
            mel_fn.save(dump_file)
            self.assertTrue(os.path.exists(dump_file), 'The saving failed !')
            
            self.assertEqual(mel_fn, MelSTFT.load_from_file(dump_file))
        finally:
            if os.path.exists(dump_file): os.remove(dump_file)

        self.assertEqual(
            mel_fn(load_audio(filename, mel_fn.rate))[0], load_mel(filename, mel_fn)
        )
        self.assertEqual(
            mel_fn(load_audio(filename, mel_fn.rate, trim_silence = True))[0],
            load_mel(filename, mel_fn, trim_silence = True)
        )
        self.assertReproductible(
            load_mel(filename, mel_fn), 'stft-{}.npy'.format(name), max_err = 2e-3
        )

