import os
import json
import librosa
import python_speech_features
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn

from utils import dump_json, load_json
from utils.audio.audio_processing import window_sumsquare
from utils.audio.audio_processing import dynamic_range_compression
from utils.audio.audio_processing import dynamic_range_decompression

class MelSTFT(object):
    def __init__(self, 
                 filter_length  = 1024, 
                 hop_length     = 256, 
                 win_length     = 1024,
                 n_mel_channels = 80, 
                 sampling_rate  = 22050, 
                 mel_fmin       = 0.0,
                 mel_fmax       = 8000.0,
                 normalize_mode = None
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

        if self.use_mel_basis:
            mel_basis = librosa_mel_fn(
                sr      = self.sampling_rate, 
                n_fft   = self.filter_length, 
                n_mels  = self.n_mel_channels, 
                fmin    = self.mel_fmin, 
                fmax    = self.mel_fmax
            )
            self.mel_basis = tf.cast(tf.transpose(mel_basis), tf.float32)
    
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
    
    def __call__(self, audio):
        """
            Compute the mel spectrogram of audio. 
            Arguments : 
                - audio : the audio to compute (shape = [length] or [1, length])
            Return :
                - mel   : the mel spectrogram (shape = [1, mel_length, n_mel_channels])
        """
        if len(audio.shape) == 1: audio = tf.expand_dims(audio, axis = 0)
        if tf.shape(audio)[1] < self.filter_length:
            audio = tf.pad(audio, [[0,0], [0, self.filter_length - tf.shape(audio)[1]]])
        
        audio = tf.cast(audio, tf.float32)
        
        mel = self.mel_spectrogram(audio)
        return mel
        
    def get_length(self, audio_length):
        """ Return expected mel_length given the audio_length """
        return int( np.ceil(audio_length / self.hop_length))
        
    def mel_spectrogram(self, audio):
        """
            Computes mel-spectrograms from a batch of waves
            Arguments :
                - audio : tf.Tensor (or ndarray) with shape (batch_size, samples) in range [-1, 1]

            Return :
                - mel_output : tf.Tensor of shape (batch_size, mel_frames, n_mel_channels)
        """
        raise NotImplementedError()
    
    def normalize(self, mel):
        if self.normalize_mode is None:
            return mel
        elif self.normalize_mode == 'per_feature':
            mean = tf.reduce_mean(mel, axis = 1, keepdims = True)
            std = tf.math.reduce_std(mel, axis = 1, keepdims = True)
        elif self.normalize_mode == 'all_feature':
            mean = tf.reduce_mean(mel)
            std = tf.math.reduce_std(mel)
            
        return (mel - mean) / (std + 1e-5)
    
    def get_config(self):
        config = {}
        config['class_name']    = self.__class__.__name__
        config['filter_length']  = self.filter_length
        config['hop_length']     = self.hop_length
        config['win_length']     = self.win_length
        config['n_mel_channels'] = self.n_mel_channels
        config['sampling_rate']  = self.sampling_rate
        config['mel_fmin']       = self.mel_fmin
        config['mel_fmax']       = self.mel_fmax
        config['normalize_mode'] = self.normalize_mode
        
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
            raise ValueError("Mel class name inconnue !\n  Reçu : {}\n  Accepté : {}".format(class_name, list(_mel_classes.keys())))
        
    
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
                 to_magnitude   = True
                ):
        self.filter_length  = filter_length
        self.hop_length     = hop_length
        self.win_length     = win_length
        self.window         = window
        self.to_magnitude   = to_magnitude
        
        scale   = self.filter_length / self.hop_length
        cutoff  = int((self.filter_length / 2 + 1))
        
        fourier_basis   = np.fft.fft(np.eye(self.filter_length))

        fourier_basis = np.vstack([
            np.real(fourier_basis[:cutoff, :]),
            np.imag(fourier_basis[:cutoff, :])
        ])

        forward_basis = tf.cast(tf.expand_dims(fourier_basis, 1), dtype = tf.float32)
        inverse_basis = tf.cast(tf.expand_dims(
            tf.transpose(tf.linalg.pinv(scale * fourier_basis)), 1
        ), dtype = tf.float32)

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = tf.cast(fft_window, tf.float32)

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window
            
        self.forward_basis = tf.transpose(forward_basis, [2, 1, 0])
        self.inverse_basis = tf.transpose(inverse_basis, [2, 1, 0])
    
    @tf.function(input_signature = [tf.TensorSpec(shape = (None, None), dtype = tf.float32)])
    def transform(self, input_data):
        """
            Applyes STFT on input_data
                - input_data tf.Tensor wit shape (batch_size, num_samples)
            Output : (magnitude, phase)
                - magnitude tf.Tensor with shape : 
                    (batch_size, mel_frames, filter_length / 2 + 1)
                - magnitude tf.Tensor with shape : 
                    (batch_size, mel_frames, filter_length / 2 + 1)
        """
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[1]

        # similar to librosa, reflect-pad the input
        input_data = tf.pad(
            input_data,
            [(0,0), (int(self.filter_length / 2), int(self.filter_length / 2))],
            mode='reflect')
        input_data = tf.expand_dims(input_data, -1)

        forward_transform = K.conv1d(
            input_data,
            self.forward_basis,
            strides = self.hop_length,
            padding = 'valid'
        )
        
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :, :cutoff]
        imag_part = forward_transform[:, :, cutoff:]
        
        phase = tf.math.atan2(imag_part, real_part)
        if self.to_magnitude:
            magnitude = tf.math.sqrt(real_part**2 + imag_part**2)
        else:
            magnitude = tf.stack([real_part, imag_part], axis = -1)
            
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
        
        recombine_magnitude_phase = tf.concat([
            magnitude * tf.math.cos(phase), magnitude * tf.math.sin(phase)
        ], axis = -1)

        stride = self.hop_length
        out_length = (mel_frames -1) * self.hop_length + self.inverse_basis.shape[0]
        out_shape = (batch_size, out_length, self.inverse_basis.shape[1])

        inverse_transform = tf.nn.conv1d_transpose(
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

        return tf.squeeze(inverse_transform, 1)

    def __call__(self, audio):
        if len(audio.shape) == 1: audio = tf.expand_dims(audio, axis = 0)
        return self.transform(audio)[0]

class TacotronSTFT(MelSTFT):
    def __init__(self, * args, ** kwargs):
        super(TacotronSTFT, self).__init__(* args, ** kwargs)
        self.stft_fn = STFT(self.filter_length, self.hop_length, self.win_length)

        self.mel_basis = tf.expand_dims(self.mel_basis, 0)

        
    def spectral_normalize(self, magnitudes):
        return dynamic_range_compression(magnitudes)

    def spectral_de_normalize(self, magnitudes):
        return dynamic_range_decompression(magnitudes)

    @tf.function(input_signature = [tf.TensorSpec(shape = (None, None), dtype = tf.float32)])
    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: tf.Tensor (or ndarray) with shape (batch_size, samples) in range [-1, 1]

        RETURNS
        -------
        mel_output: tf.Tensor of shape (batch_size, mel_frames, n_mel_channels)
        """
        magnitudes, phases = self.stft_fn.transform(y)
        mel_output = tf.matmul(magnitudes, self.mel_basis)
        mel_output = self.spectral_normalize(mel_output)
        return self.normalize(mel_output)
    
    
class SpeechNetSTFT(MelSTFT):
    @tf.function(input_signature = [tf.TensorSpec(shape = (None, None), dtype = tf.float32)])
    def mel_spectrogram(self, y):
        stfts = tf.signal.stft(y,
                               fft_length   = self.filter_length,
                               frame_step   = self.hop_length,
                               frame_length = self.win_length,
                               pad_end      = False
                              )
        power_spectrograms = tf.math.real(stfts * tf.math.conj(stfts))
        
        mel_output = tf.tensordot(power_spectrograms, self.mel_basis, 1)
        mel_output = tf.math.log(mel_output + 1e-6)
        return self.normalize(mel_output)
    
class DeepSpeechSTFT(MelSTFT):
    def make_features(self, audio):
        audio = np.array(audio.numpy() * np.iinfo(np.int16).max, dtype = np.int16)
        frames = python_speech_features.sigproc.framesig(
            audio, self.win_length, self.hop_length, np.hanning
        )
        mel = python_speech_features.sigproc.logpowspec(
            frames, self.win_length, norm = True
        )
        mel = mel[:, :self.n_mel_channels]
        return mel
        
    @tf.function(input_signature = [tf.TensorSpec(shape = (None, None), dtype = tf.float32)])
    def mel_spectrogram(self, audio):
        audio = tf.squeeze(audio, 0)
        audio = audio / (tf.reduce_max(tf.abs(audio)) + 1e-5)
        mel_output = tf.py_function(self.make_features, [audio], Tout = tf.float32)
        mel_output.set_shape([None, self.n_mel_channels])
        mel_output = tf.expand_dims(mel_output, 0)
        
        return self.normalize(mel_output)

class JasperSTFT(MelSTFT):
    def __init__(self, dither = 1e-5, preemph = 0.97, log = True, pad_to = 0, ** kwargs):
        super(JasperSTFT, self).__init__(** kwargs)
        self.preemph    = preemph
        self.dither = dither
        self.pad_to = pad_to
        self.log    = log
        
        self.stft_fn = STFT(self.filter_length, self.hop_length, 
                            self.win_length, to_magnitude = False)
        
    def get_seq_len(self, audio):
        return tf.cast(tf.math.ceil(tf.shape(audio)[1] / self.hop_length), tf.int32)
    
    @tf.function(input_signature = [tf.TensorSpec(shape = (None, None), dtype = tf.float32)])
    def mel_spectrogram(self, audio):
        
        if self.dither > 0:
            audio += self.dither * tf.random.normal(tf.shape(audio), dtype = audio.dtype)
        
        if self.preemph is not None:
            audio = tf.concat([tf.expand_dims(audio[:,0], 1), audio[:, 1:] - self.preemph * audio[:, :-1]], axis = 1)
                
        x = self.stft_fn(audio)
        x = tf.reduce_sum(tf.square(x), axis = -1)
        
        mel_output = tf.matmul(x, self.mel_basis)
        
        if self.log:
            mel_output = tf.math.log(mel_output + 1e-20)
        
        mel_output = self.normalize(mel_output)
        
        mask = tf.range(tf.shape(mel_output)[1]) < self.get_seq_len(audio)
        mask = tf.cast(tf.reshape(mask, [1, -1, 1]), mel_output.dtype)
        mel_output = mel_output * mask
        
        if self.pad_to < 0:
            mel_output = tf.pad(mel_output, [(0,0), (0, self.max_length - tf.shape(mel_output)[1]), (0,0)])
        elif self.pad_to > 0:
            pad_amt = tf.shape(mel_output)[1] % self.pad_to
            #            if pad_amt != 0:
            mel_output = tf.pad(mel_output, [(0,0), (0, self.pad_to - pad_amt), (0,0)])
        
        return mel_output
    
    def get_config(self):
        config = super(JasperSTFT, self).get_config()
        config['preemph']   = self.preemph
        config['dither']    = self.dither
        config['pad_to']    = self.pad_to
        config['log']       = self.log
        return config
    
class LibrosaSTFT(MelSTFT):
    def make_features(self, audio):
        return librosa.feature.melspectrogram(
            audio.numpy(),
            self.sampling_rate,
            n_fft = self.filter_length,
            hop_length = self.hop_length,
            n_mels = self.n_mel_channels
        ).astype(np.float32).T
        
    @tf.function(input_signature = [tf.TensorSpec(shape = (None, None), dtype = tf.float32)])
    def mel_spectrogram(self, audio):
        audio = tf.squeeze(audio, 0)

        mel_output = tf.py_function(self.make_features, [audio], Tout = tf.float32)
        mel_output.set_shape([None, self.n_mel_channels])
        mel_output = tf.expand_dims(mel_output, 0)
        
        return self.normalize(mel_output)
    
_mel_classes = {
    'JasperSTFT'        : JasperSTFT,
    'LibrosaSTFT'       : LibrosaSTFT,
    'TacotronSTFT'      : TacotronSTFT,
    'SpeechNetSTFT'     : SpeechNetSTFT,
    'DeepSpeechSTFT'    : DeepSpeechSTFT
}