# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
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
import keras
import librosa
import numpy as np

from scipy.linalg import pinv
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn

from loggers import timer
from utils import dump_json, load_json
from utils.keras_utils import TensorSpec, ops, graph_compile, execute_eagerly
from .audio_processing import window_sumsquare
from .audio_processing import dynamic_range_compression, dynamic_range_decompression

class MelSTFT(object):
    def __init__(self, 
                 filter_length  = 1024, 
                 hop_length     = 256, 
                 win_length     = 1024,
                 n_mel_channels = 80, 
                 sampling_rate  = 22050, 
                 mel_fmin       = 0.0,
                 mel_fmax       = 8000.0,
                 normalize_mode = None,
                 pre_emph       = 0.,
                 ** kwargs
                ):
        assert normalize_mode in (None, 'per_feature', 'all_feature')
        self.filter_length  = filter_length if filter_length > 1. else int(filter_length * sampling_rate)
        self.hop_length     = hop_length if hop_length > 1. else int(hop_length * sampling_rate)
        self.win_length     = win_length if win_length > 1. else int(win_length * sampling_rate)
        self.n_mel_channels = n_mel_channels
        self.sampling_rate  = sampling_rate
        self.mel_fmin       = mel_fmin
        self.mel_fmax       = mel_fmax
        self.normalize_mode = normalize_mode
        self.pre_emph   = pre_emph
        
        if self.use_mel_basis:
            mel_basis = librosa_mel_fn(
                sr      = self.sampling_rate, 
                n_fft   = self.filter_length, 
                n_mels  = self.n_mel_channels, 
                fmin    = self.mel_fmin, 
                fmax    = self.mel_fmax
            )
            self.mel_basis = mel_basis.T.astype(np.float32)
        
        self.min_length = self.filter_length
        
        if not self.run_eagerly:
            self.mel_spectrogram    = timer(graph_compile(
                self.mel_spectrogram,
                input_signature = TensorSpec(shape = (None, None), dtype = 'float32')
            ))
    
    @property
    def run_eagerly(self):
        return False
    
    @property
    def rate(self):
        return self.sampling_rate
    
    @property
    def dir_name(self):
        return 'mels_{}_chann-{}_filt-{}_hop-{}_win-{}_norm-{}'.format(
            self.sampling_rate,
            self.n_mel_channels,
            self.filter_length,
            self.hop_length,
            self.win_length,
            self.normalize_mode
        )
        
    @property
    def use_mel_basis(self):
        return True

    def __str__(self):
        config = self.get_config()
        des = "\n========== {} ==========\n".format(config.pop('class_name'))
        for k, v in config.items():
            des += "{}\t: {}\n".format(k, v)
        return des
    
    def __call__(self, audio, ** kwargs):
        """
            Compute the mel spectrogram of audio. 
            Arguments : 
                - audio : the audio to compute (shape = [length] or [1, length])
            Return :
                - mel   : the mel spectrogram (shape = [1, mel_length, n_mel_channels])
        """
        audio = ops.convert_to_tensor(audio, 'float32')
        
        if len(ops.shape(audio)) == 1: audio = ops.expand_dims(audio, axis = 0)
        if ops.shape(audio)[1] < self.min_length:
            audio = ops.pad(audio, [[0,0], [0, self.min_length - ops.shape(audio)[1]]])
        
        if self.pre_emph > 0.:
            audio = ops.concat([
                audio[:, :1], audio[:, 1:] - self.pre_emph * audio[:, :-1]
            ], axis = 1)
        
        if self.run_eagerly: kwargs = {}
        return self.mel_spectrogram(audio, ** kwargs)
    
    def get_length(self, audio_length):
        """ Return expected mel_length given the audio_length """
        if ops.executing_eagerly():
            return int(math.ceil(max(self.filter_length, audio_length) / self.hop_length))
        return ops.cast(ops.ceil(
            ops.maximum(self.filter_length, audio_length) / self.hop_length
        ), 'int32')
    
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
        return ops.divide_no_nan(mel - ops.mean(mel, ** kwargs), _std(mel, ** kwargs))
    
    def get_config(self):
        config = {
            'class_name'    : self.__class__.__name__,
            'filter_length' : self.filter_length,
            'hop_length'    : self.hop_length,
            'win_length'    : self.win_length,
            'n_mel_channels'    : self.n_mel_channels,
            'sampling_rate' : self.sampling_rate,
            'mel_fmin'      : self.mel_fmin,
            'mel_fmax'      : self.mel_fmax,
            'normalize_mode'    : self.normalize_mode,
            'pre_emph'      : self.pre_emph
        }
        
        return config
    
    def save_to_file(self, filename):
        if '.json' not in filename:
            filename += '.json'
        
        dump_json(filename, self.get_config(), indent = 4)

        return filename

    @staticmethod
    def create(class_name, * args, ** kwargs):
        if class_name in _mel_classes:
            return _mel_classes[class_name](* args, ** kwargs)
        elif os.path.isfile(class_name):
            return MelSTFT.load_from_file(class_name)
        else:
            raise ValueError("Unknown Mel STFT class !\n  Accepted : {}\n  Got : {}".format(tuple(_mel_classes.keys()), class_name))
    
    @classmethod
    def load_from_file(cls, filename):
        config = load_json(filename)
        
        return cls.create(** config)

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
        
        scale   = self.filter_length / self.hop_length
        cutoff  = self.filter_length // 2 + 1
        
        fourier_basis   = np.fft.fft(np.eye(self.filter_length))

        fourier_basis = np.vstack([
            np.real(fourier_basis[:cutoff, :]),
            np.imag(fourier_basis[:cutoff, :])
        ])

        forward_basis = np.expand_dims(fourier_basis, 1).astype(np.float32)
        inverse_basis = np.expand_dims(
            np.transpose(pinv(scale * fourier_basis)), 1
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
        num_batches = audio.shape[0]
        num_samples = audio.shape[1]

        # similar to librosa, reflect-pad the input
        audio = ops.pad(
            audio,
            [(0,0), (self.filter_length // 2, self.filter_length // 2)],
            mode = 'reflect'
        )[:, :, None]

        forward_transform = ops.conv1d(
            audio, self.forward_basis, self.hop_length, padding = 'valid'
        )
        
        cutoff = self.filter_length // 2 + 1
        real_part = forward_transform[:, :, :cutoff]
        imag_part = forward_transform[:, :, cutoff:]
        
        phase = ops.atan2(imag_part, real_part)
        if self.to_magnitude:
            magnitude = ops.sqrt(real_part**2 + imag_part**2)
        else:
            magnitude = ops.stack([real_part, imag_part], axis = -1)
        
        return magnitude, phase

    def inverse(self, magnitude, phase):
        """
            Applyes ISTFT on [magnitude, phase]
                - magnitude shape   : (batch_size, mel_frames, filter_length / 2 + 1)
                - magnitude shape   : (batch_size, mel_frames, filter_length / 2 + 1)
            Output : inverse_transform
                - inverse_transform : (batch_size, num_samples)
        """
        batch_size  = magnitude.shape[0]
        mel_frames  = magnitude.shape[1]
        
        recombine_magnitude_phase = ops.concat([
            magnitude * ops.cos(phase), magnitude * ops.sin(phase)
        ], axis = -1)

        stride = self.hop_length
        out_length = (mel_frames -1) * self.hop_length + self.inverse_basis.shape[0]
        out_shape = (batch_size, out_length, self.inverse_basis.shape[1])

        inverse_transform = ops.conv1d_transpose(
            recombine_magnitude_phase,
            filters     = self.inverse_basis,
            output_shape    = out_shape,
            strides     = self.hop_length,
            padding     = 'VALID'
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.shape[1], 
                hop_length  = self.hop_length,
                win_length  = self.win_length, 
                n_fft       = self.filter_length,
                dtype   = np.float32
            )

            # remove modulation effects
            approx_nonzero_indices = np.where(window_sum > tiny(window_sum))[0]

            inverse_transform = np.array(np.transpose(inverse_transform, [0, 2, 1]))
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return ops.squeeze(inverse_transform, 1)

    def __call__(self, audio):
        audio = ops.convert_to_tensor(audio, 'float32')
        if len(audio.shape) == 1: audio = audio[None, :]
        return self.transform(audio)[0]

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
    def __init__(self, * args, window = 'hann', periodic = True, ** kwargs):
        super(TacotronSTFT, self).__init__(* args, ** kwargs)
        self.stft_fn = STFT(
            self.filter_length, self.hop_length, self.win_length,
            window = window, periodic = periodic
        )

        self.mel_basis = self.mel_basis[None]
        
    def spectral_normalize(self, magnitudes):
        return dynamic_range_compression(magnitudes)

    def spectral_de_normalize(self, magnitudes):
        return dynamic_range_decompression(magnitudes)

    def mel_spectrogram(self, audio):
        magnitudes, phases = self.stft_fn.transform(audio)
        mel_output = magnitudes @ ops.convert_to_tensor(self.mel_basis, magnitudes.dtype)
        mel_output = self.spectral_normalize(mel_output)
        return self.normalize(mel_output)
    
    def get_config(self):
        config = super(TacotronSTFT, self).get_config()
        config.update(self.stft_fn.get_config())
        return config
    
class ConformerSTFT(TacotronSTFT):
    def __init__(self,
                 * args,
                 mag_power = 1.,
                 log    = True,
                 log_zero_guard_type    = 'add',
                 log_zero_guard_value   = 2 ** -24,
                 ** kwargs
                ):
        super().__init__(* args, ** kwargs)
        self.log    = log
        self.log_zero_guard_type    = log_zero_guard_type
        self.log_zero_guard_value   = log_zero_guard_value
        self.mag_power  = mag_power
    
    def mel_spectrogram(self, audio):
        magnitudes, phases = self.stft_fn.transform(audio)

        if self.mag_power != 1.:
            magnitudes = magnitudes ** self.mag_power

        mel_output = magnitudes @ ops.convert_to_tensor(self.mel_basis, magnitudes.dtype)
        
        if self.log:
            if self.log_zero_guard_type == "add":
                mel_output = ops.log(mel_output + self.log_zero_guard_value)
            elif self.log_zero_guard_type == "clamp":
                raise NotImplementedError()
            else:
                raise ValueError("log_zero_guard_type was not understood")

        return self.normalize(mel_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'log'   : self.log,
            'mag_power' : self.mag_power,
            'log_zero_guard_type'   : self.log_zero_guard_type,
            'log_zero_guard_value'  : self.log_zero_guard_value
        })
        return config
    

class WhisperSTFT(TacotronSTFT):
    def __init__(self,
                 filter_length  = 400, 
                 hop_length     = 160, 
                 win_length     = 400,
                 n_mel_channels = 80, 
                 sampling_rate  = 16000, 
                 mel_fmin       = 0.0,
                 mel_fmax       = 8000.0,
                 ** kwargs
                ):
        super().__init__(
            filter_length  = filter_length, 
            hop_length     = hop_length, 
            win_length     = win_length,
            n_mel_channels = n_mel_channels, 
            sampling_rate  = sampling_rate, 
            mel_fmin       = mel_fmin,
            mel_fmax       = mel_fmax,
            ** kwargs
        )

    def mel_spectrogram(self, audio):
        magnitudes, phases = self.stft_fn.transform(audio)
        
        magnitudes = ops.squeeze(ops.abs(magnitudes[:, :-1]))
        
        mel_output = ops.matmul(
            magnitudes, ops.convert_to_tensor(self.mel_basis, magnitudes.dtype)
        )
        mel_output = ops.log10(ops.maximum(mel_output, 1e-10))

        mel_output = ops.maximum(mel_output, ops.max(mel_output) - 8.0)
        mel_output = (mel_output + 4.0) / 4.0

        return mel_output

class SpeechNetSTFT(MelSTFT):
    def mel_spectrogram(self, audio):
        stfts = ops.stft(
            audio, self.win_length, self.hop_length, self.filter_length, center = False
        )
        # the `tf.signal.stft` returns the `complex Tensor`, while `K.stft` returns `(real, imag)` 
        if isinstance(stfts, (list, tuple)): stfts = ops.complex(* stfts)
        power_spectrograms = ops.real(stfts * ops.conj(stfts))
        
        mel_output = ops.tensordot(
            power_spectrograms, ops.convert_to_tensor(self.mel_basis, power_spectrograms.dtype), 1
        )
        mel_output = ops.log(mel_output + 1e-6)
        return self.normalize(mel_output)
    
class DeepSpeechSTFT(MelSTFT):
    @property
    def run_eagerly(self):
        return True
    
    @execute_eagerly(
        signature = TensorSpec(shape = (None, None), dtype = 'float32'), numpy = True
    )
    def make_features(self, audio):
        import python_speech_features
        
        audio = np.array(audio * np.iinfo(np.int16).max, dtype = np.int16)
        frames = python_speech_features.sigproc.framesig(
            audio, self.win_length, self.hop_length, np.hanning
        )
        mel = python_speech_features.sigproc.logpowspec(
            frames, self.win_length, norm = True
        )
        return mel[:, :self.n_mel_channels].astype(np.float32)
        
    def mel_spectrogram(self, audio):
        audio = ops.divide_no_nan(audio, ops.max(ops.abs(audio)))
        mel_output = self.make_features(
            audio[0], shape = (None, self.n_mel_channels)
        )[None]
        
        return self.normalize(mel_output)

class JasperSTFT(MelSTFT):
    def __init__(self, dither = 1e-5, preemph = 0.97, log = True, pad_to = 0, ** kwargs):
        super(JasperSTFT, self).__init__(** kwargs)
        self.preemph    = preemph
        self.dither = dither
        self.pad_to = pad_to
        self.log    = log
        
        self.stft_fn = STFT(
            self.filter_length, self.hop_length, self.win_length, to_magnitude = False
        )
        
    def get_seq_len(self, audio):
        if ops.executing_eagerly():
            return int(math.ceil(max(self.filter_length, audio.shape[1]) / self.hop_length))
        return ops.cast(ops.ceil(ops.shape(audio)[1] / self.hop_length), 'int32')
    
    def mel_spectrogram(self, audio):
        if self.dither > 0:
            audio += self.dither * ops.random.normal(ops.shape(audio), dtype = audio.dtype)
        
        if self.preemph is not None:
            audio = ops.concat([
                audio[:, :1], audio[:, 1:] - self.preemph * audio[:, :-1]
            ], axis = 1)
                
        x = self.stft_fn(audio)
        x = ops.reduce_sum(ops.square(x), axis = -1)
        
        mel_output = ops.matmul(x, ops.convert_to_tensor(self.mel_basis, x.dtype))
        
        if self.log:
            mel_output = ops.log(mel_output + 1e-20)
        
        mel_output = self.normalize(mel_output)
        
        mask = ops.range(ops.shape(mel_output)[1]) < self.get_seq_len(audio)
        mask = ops.cast(ops.reshape(mask, [1, -1, 1]), mel_output.dtype)
        mel_output = mel_output * mask
        
        if self.pad_to < 0:
            mel_output = ops.pad(
                mel_output, [(0,0), (0, self.max_length - ops.shape(mel_output)[1]), (0,0)]
            )
        elif self.pad_to > 0:
            pad_amt = ops.shape(mel_output)[1] % self.pad_to
            #            if pad_amt != 0:
            mel_output = ops.pad(mel_output, [(0,0), (0, self.pad_to - pad_amt), (0,0)])
        
        return mel_output
    
    def get_config(self):
        config = super(JasperSTFT, self).get_config()
        config['preemph']   = self.preemph
        config['dither']    = self.dither
        config['pad_to']    = self.pad_to
        config['log']       = self.log
        return config
    
class LibrosaSTFT(MelSTFT):
    @property
    def run_eagerly(self):
        return True
    
    @execute_eagerly(signature = TensorSpec(
        shape = (None, None), dtype = 'float32'
    ), numpy = True)
    def make_features(self, audio):
        return librosa.feature.melspectrogram(
            y   = audio,
            sr  = self.sampling_rate,
            n_fft   = self.filter_length,
            hop_length  = self.hop_length,
            n_mels  = self.n_mel_channels
        ).astype(np.float32).T
    
    def mel_spectrogram(self, audio):
        mel_output = self.make_features(
            audio[0], shape = (None, self.n_mel_channels)
        )[None]
        
        return self.normalize(mel_output)
    
_mel_classes = {
    'JasperSTFT'        : JasperSTFT,
    'ConformerSTFT'     : ConformerSTFT,
    'LibrosaSTFT'       : LibrosaSTFT,
    'TacotronSTFT'      : TacotronSTFT,
    'SpeechNetSTFT'     : SpeechNetSTFT,
    'DeepSpeechSTFT'    : DeepSpeechSTFT,
    'WhisperSTFT'       : WhisperSTFT
}