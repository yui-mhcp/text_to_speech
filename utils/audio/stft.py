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
import json
import math
import librosa
import numpy as np

from scipy.linalg import pinv
from scipy.signal import get_window
from librosa.util import pad_center
from librosa.filters import mel as librosa_mel_fn

from loggers import timer
from ..file_utils import dump_json, load_json
from ..keras import TensorSpec, ops, graph_compile

class MelSTFT(object):
    def __init__(self,
                 sampling_rate,
                 n_mel_channels = 80,
                 
                 *,
                 
                 win_length = 1024,
                 hop_length = 256,
                 filter_length  = 1024,

                 mel_fmin       = 0.0,
                 mel_fmax       = 8000.0,
                 
                 normalize_mode = None,
                 pre_emph       = 0.,
                 
                 ** kwargs
                ):
        assert normalize_mode in (None, 'per_feature', 'all_feature')
        
        self.n_mel_channels = n_mel_channels
        self.sampling_rate  = sampling_rate

        self.win_length     = win_length if win_length > 1. else int(win_length * sampling_rate)
        self.hop_length     = hop_length if hop_length > 1. else int(hop_length * sampling_rate)
        self.filter_length  = filter_length if filter_length > 1. else int(
            filter_length * sampling_rate
        )

        self.mel_fmin       = mel_fmin
        self.mel_fmax       = mel_fmax
        
        self.pre_emph   = pre_emph
        self.normalize_mode = normalize_mode
        
        self.mel_basis = None
        if self.use_mel_basis:
            mel_basis = librosa_mel_fn(
                sr      = self.sampling_rate, 
                n_fft   = self.filter_length, 
                n_mels  = self.n_mel_channels, 
                fmin    = self.mel_fmin, 
                fmax    = self.mel_fmax
            )
            self.mel_basis = np.expand_dims(mel_basis.T.astype(np.float32), axis = 0)
        
        if not self.run_eagerly:
            self.mel_spectrogram    = graph_compile(
                fn = self.mel_spectrogram,
                input_signature = TensorSpec(shape = (None, None), dtype = 'float32'),
                support_xla     = False
            )
    
    @property
    def run_eagerly(self):
        return False
    
    @property
    def rate(self):
        return self.sampling_rate
    
    @property
    def use_mel_basis(self):
        return True

    def __str__(self):
        config = self.get_config()
        des = "\n========== {} ==========\n".format(config.pop('class_name'))
        for k, v in config.items():
            des += "{}\t: {}\n".format(k, v)
        return des
    
    @timer(name = 'mel spectrogram')
    def __call__(self, audio, ** kwargs):
        """
            Compute the mel spectrogram of audio. 
            Arguments : 
                - audio : the audio to compute (shape = [length] or [1, length])
            Return :
                - mel   : the mel spectrogram (shape = [1, mel_length, n_mel_channels])
        """
        if not (self.run_eagerly or kwargs.get('run_eagerly', False)):
            audio = ops.convert_to_tensor(audio, 'float32')
        else:
            kwargs = {}

        if len(ops.shape(audio)) == 1: audio = ops.expand_dims(audio, axis = 0)
        if ops.shape(audio)[1] < self.win_length:
            audio = ops.pad(audio, [[0,0], [0, self.win_length - ops.shape(audio)[1]]])
        
        if self.pre_emph > 0.:
            audio = ops.concatenate([
                audio[:, :1], audio[:, 1:] - self.pre_emph * audio[:, :-1]
            ], axis = 1)
        
        if self.run_eagerly: kwargs = {}
        return self.mel_spectrogram(audio, ** kwargs)
    
    def get_mel_length(self, audio_length):
        """ Return expected mel_length given the audio_length """
        return int(math.ceil(max(self.filter_length, audio_length) / self.hop_length))
    
    def get_audio_length(self, mel_length):
        return mel_length * self.hop_length
    
    def mel_spectrogram(self, audio):
        """
            Computes mel-spectrograms from a batch of waves
            Arguments :
                - audio : Tensor (or ndarray) with shape (batch_size, samples) in range [-1, 1]

            Return :
                - mel_output : Tensor of shape (batch_size, mel_frames, n_mel_channels)
        """
        raise NotImplementedError()
    
    def normalize(self, mel):
        if self.normalize_mode is None: return mel
        
        kwargs = {'axis' : 1, 'keepdims' : True} if self.normalize_mode == 'per_feature' else {}
        return ops.divide_no_nan(mel - ops.mean(mel, ** kwargs), ops.std(mel, ** kwargs))
    
    def get_config(self):
        return {
            'class_name'    : self.__class__.__name__,
            
            'n_mel_channels'    : self.n_mel_channels,
            'sampling_rate'     : self.sampling_rate,
            
            'win_length'    : self.win_length,
            'hop_length'    : self.hop_length,
            'filter_length' : self.filter_length,

            'mel_fmin'      : self.mel_fmin,
            'mel_fmax'      : self.mel_fmax,
            
            'pre_emph'      : self.pre_emph,
            'normalize_mode'    : self.normalize_mode
        }
    
    def save(self, filename):
        if not filename.endswith('.json'): filename += '.json'
        
        return dump_json(filename, self.get_config(), indent = 4)

    save_to_file    = save

    @classmethod
    def load_from_file(cls, filename):
        return MelSTFT.create(filename)

    @staticmethod
    def create(class_name, * args, ** kwargs):
        if class_name in _mel_classes:
            return _mel_classes[class_name](* args, ** kwargs)
        elif os.path.isfile(class_name):
            return MelSTFT.create(** load_json(class_name))
        else:
            raise ValueError("Unknown Mel STFT class !\n  Accepted : {}\n  Got : {}".format(tuple(_mel_classes.keys()), class_name))

class STFT(object):
    """ 
        Tensorflow adaptation of Prem Seetharaman's https://github.com/pseeth/pytorch-stft
        Equivalent of torch.stft (if to_magnitude == False)
        This class has been modified to be tensorflow-compatible. Many tests have been done to be sure that this produces the exact same output as the pytorch version
    """
    def __init__(self, 
                 filter_length  = 800, 
                 hop_length     = 200, 
                 win_length     = 800,
                 window         = 'hann',
                 to_magnitude   = True,
                 periodic       = True
                ):
        self.filter_length  = filter_length
        self.hop_length     = hop_length
        self.win_length     = win_length
        self.window         = window
        self.to_magnitude   = to_magnitude
        self.periodic       = periodic
        
        self._scale     = self.filter_length / self.hop_length
        self._cutoff    = self.filter_length // 2 + 1
        
        fourier_basis   = np.fft.fft(np.eye(self.filter_length))

        fourier_basis = np.vstack([
            np.real(fourier_basis[: self._cutoff, :]),
            np.imag(fourier_basis[: self._cutoff, :])
        ])

        forward_basis = np.expand_dims(fourier_basis, 1).astype(np.float32)
        inverse_basis = np.expand_dims(
            np.transpose(pinv(self._scale * fourier_basis)), 1
        ).astype(np.float32)

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins = periodic)
            fft_window = pad_center(fft_window, size = filter_length)
            fft_window = fft_window

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window
            
        self.forward_basis = np.transpose(forward_basis, [2, 1, 0])
        self.inverse_basis = np.transpose(inverse_basis, [2, 1, 0])
    
    def __call__(self, audio):
        if len(audio.shape) == 1: audio = audio[None, :]
        return self.transform(audio)[0]

    def transform(self, audio):
        """
            Applyes STFT on audio
                - audio Tensor wit shape (batch_size, num_samples)
            Output : (magnitude, phase)
                - magnitude Tensor with shape : 
                    (batch_size, mel_frames, filter_length / 2 + 1)
                - magnitude Tensor with shape : 
                    (batch_size, mel_frames, filter_length / 2 + 1)
        """
        batch_size, num_samples = audio.shape[0], audio.shape[1]

        # similar to librosa, reflect-pad the input
        audio = ops.pad(
            audio,
            [(0, 0), (self.filter_length // 2, self.filter_length // 2)],
            mode = 'reflect'
        )[:, :, None]

        forward_transform = ops.conv1d(
            audio, self.forward_basis, self.hop_length, padding = 'valid'
        )
        
        real_part = forward_transform[:, :, :self._cutoff]
        imag_part = forward_transform[:, :, self._cutoff:]
        
        phase = ops.atan2(imag_part, real_part)
        if self.to_magnitude:
            magnitude = ops.sqrt(real_part ** 2 + imag_part ** 2)
        else:
            magnitude = ops.stack([real_part, imag_part], axis = -1)
        
        return magnitude, phase

    def get_config(self):
        return {
            'filter_length' : self.filter_length,
            'hop_length'    : self.hop_length,
            'win_length'    : self.win_length,
            'window'        : self.window,
            'to_magnitude'  : self.to_magnitude,
            'periodic'      : self.periodic
        }

class TacotronSTFT(MelSTFT):
    def __init__(self,
                 sampling_rate  = 22050,
                 n_mel_channels = 80,

                 *,
                 
                 window = 'hann',
                 periodic   = True,
                 
                 ** kwargs
                ):
        super().__init__(sampling_rate = sampling_rate, n_mel_channels = n_mel_channels, ** kwargs)
        self.stft_fn = STFT(
            filter_length   = self.filter_length,
            hop_length  = self.hop_length,
            win_length  = self.win_length,
            periodic    = periodic,
            window  = window
        )
        
    def spectral_normalize(self, magnitudes, clip_val = 1e-5):
        return ops.log(ops.maximum(magnitudes, clip_val))

    def mel_spectrogram(self, audio):
        magnitudes, phases = self.stft_fn.transform(audio)
        mel_output = magnitudes @ ops.convert_to_tensor(self.mel_basis, magnitudes.dtype)
        mel_output = self.spectral_normalize(mel_output)
        return self.normalize(mel_output)
    
    def get_config(self):
        config = super().get_config()
        config.update(self.stft_fn.get_config())
        return config
    
class WhisperSTFT(TacotronSTFT):
    def __init__(self,
                 sampling_rate  = 16000,
                 n_mel_channels = 80,
                 
                 *,
                 
                 win_length     = 400,
                 hop_length     = 160, 
                 filter_length  = 400, 

                 mel_fmin       = 0.0,
                 mel_fmax       = 8000.0,
                 
                 ** kwargs
                ):
        super().__init__(
            sampling_rate  = sampling_rate, 
            n_mel_channels = n_mel_channels, 

            win_length     = win_length,
            hop_length     = hop_length, 
            filter_length  = filter_length, 

            mel_fmin       = mel_fmin,
            mel_fmax       = mel_fmax,
            ** kwargs
        )

    def mel_spectrogram(self, audio):
        magnitudes, phases = self.stft_fn.transform(audio)
        magnitudes = ops.abs(magnitudes[:, :-1])
        
        mel_output = ops.matmul(
            magnitudes, ops.convert_to_tensor(self.mel_basis, magnitudes.dtype)
        )
        mel_output = ops.log10(ops.maximum(mel_output, 1e-10))

        mel_output = ops.maximum(
            mel_output, ops.max(mel_output, axis = [1, 2], keepdims = True) - 8.0
        )
        mel_output = (mel_output + 4.0) / 4.0

        return mel_output
    
_mel_classes = {k : v for k, v in globals().items() if isinstance(v, type) and issubclass(v, MelSTFT)}
