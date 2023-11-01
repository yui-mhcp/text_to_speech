
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
import unittest
import numpy as np
import tensorflow as tf

from utils.audio import *
from utils.audio.stft import _mel_classes
from unitests import CustomTestCase, data_dir, reproductibility_dir

filename = os.path.join(data_dir, 'audio_test.wav')

@unittest.skipIf(not os.path.exists(filename), '{} does not exist'.format(filename))
class TestAudio(CustomTestCase):
    def setUp(self):
        self.rate, self.audio = read_audio(filename)
        
    def test_audio_io(self):
        self.assertEqual(self.rate, 16000)
        self.assertEqual(len(self.audio), 64880)
        self.assertEqual(load_audio(filename, self.rate), self.audio)

    def test_processing(self):
        self.assertReproductible(
            load_audio(filename, rate = 22050), 'audio_resample.npy'
        )
        self.assertReproductible(
            load_audio(filename, rate = None, reduce_noise = True), 'reduce_noise.npy'
        )
        self.assertReproductible(
            load_audio(filename, rate = None, trim_silence = True), 'trim_silence.npy'
        )
        self.assertReproductible(
            load_audio(filename, rate = None, trim_silence = True, method = 'window'),
            'trim_silence-window.npy'
        )

        trimmed = trim_silence(self.audio, rate = self.rate, method = 'window')
        self.assertTrue(len(self.audio) > len(trimmed))

        loaded_trimmed = load_audio(filename, self.rate, trim_silence = True, method = 'window')
        self.assertEqual(loaded_trimmed, trimmed)

        self.assertEqual(load_audio(write_audio(
            self.audio, os.path.join(reproductibility_dir, 'write_audio.wav'), self.rate, normalize = False
        ), None, normalize = False), self.audio, max_err = 1e-6)

    def test_stft(self):
        for name, mel_class in _mel_classes.items():
            with self.subTest(name = name):
                mel_fn = mel_class() if 'Jasper' not in name else mel_class(dither = 0.)
                
                self.assertEqual(MelSTFT.load_from_file(mel_fn.save_to_file(
                    os.path.join(reproductibility_dir, '{}.json'.format(name))
                )), mel_fn)
                
                self.assertEqual(
                    load_mel(filename, mel_fn), mel_fn(load_audio(filename, mel_fn.rate))[0]
                )
                self.assertEqual(
                    load_mel(filename, mel_fn, trim_silence = True),
                    mel_fn(load_audio(filename, mel_fn.rate, trim_silence = True))[0]
                )
                self.assertReproductible(
                    load_mel(filename, mel_fn), 'stft-{}.npy'.format(name)
                )
