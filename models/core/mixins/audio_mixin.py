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

from utils.keras import TensorSpec, ops
from utils.audio import MelSTFT, load_audio, load_mel

_supported_audio_format = ('raw', 'audio', 'mel', 'spect', 'spectrogram', 'mel_image')

# Maps the (training) trim hyper-parameter attribute names to the corresponding `load_audio` /
# `trim_silence` / `reduce_noise` keyword names.
_trim_kwargs_keys   = {
    'trim_audio'    : 'trim_silence',
    'reduce_noise'  : 'reduce_noise',
    'trim_threshold'    : 'threshold',
    'min_silence'   : 'min_silence',
    'replace_by'    : 'replace_by',
    'trim_method'   : 'method',
    'trim_mode'     : 'mode'
}

AudioTrainingHParams = {
    'reduce_noise'    : False,
    'trim_audio'  : False,
    'trim_threshold'   : -25,
    'min_silence' : 0.1,
    'replace_by'  : 0.4,
    'trim_method' : 'rms',
    'trim_mode'   : 'remove'
}

class AudioModelMixin:
    def _init_audio(self,
                    rate    = None,
                    audio_format   = 'mel',
                    
                    load_audio_kwargs   = None,
                    
                    mel_fn      = 'TacotronSTFT',
                    mel_config  = None,
                    pad_mel_value   = 0.,
                    
                    # for retro-compatibility
                    audio_rate  = None,
                    mel_fn_type = None,
                    mel_as_image    = None,
                    
                    ** kwargs
                   ):
        """
            Initializes the audio-related variables
            
            Arguments :
                - rate  : the audio sampling rate
                - audio_format  : the audio format handled by the model (audio / mel / mel_image)
                - load_audio_kwargs : kwargs forwarded to `load_audio`
                
                - mel_fn    : either the mel's filename or a mel's type
                - mel_config    : the mel spectrogram's config
                - pad_mel_value : value used to pad mel-spectrogram (in batch)
                
                - mel_fn_type / mel_as_image : for retro-compatibility
        """
        assert audio_format in _supported_audio_format, '{} is not a valid audio format : {}'.format(audio_format, _supported_audio_format)
        
        if audio_rate:  self.audio_rate = rate = audio_rate
        if mel_fn_type is not None:     self.mel_fn_type = mel_fn = mel_fn_type
        if mel_as_image is not None:    audio_format = 'mel' if not mel_as_image else 'mel_image'
        
        if mel_config is None: mel_config = {}
        
        self.rate   = rate
        self.audio_format   = audio_format
        self.pad_mel_value  = pad_mel_value
        
        self.mel_fn = None
        self.mel_config = mel_config
        if self.use_mel:
            # Initialization of mel fn
            if isinstance(mel_fn, MelSTFT):
                self.mel_fn     = mel_fn
            else:
                if self.rate: mel_config['sampling_rate'] = self.rate
                self.mel_fn    = MelSTFT.create(mel_fn, ** mel_config)
        
        self.load_audio_kwargs = load_audio_kwargs if load_audio_kwargs else {}
        # Assert the configuration is valid / complete
        if not self.rate:
            assert self.use_mel, 'You must specify the `audio_rate` parameter !'
            self.rate = self.mel_fn.sampling_rate
        elif self.use_mel:
            assert self.rate == self.mel_fn.sampling_rate, 'The `audio_rate` differs from the `mel_fn.sampling_rate` : {} != {}'.format(self.rate, self.mel_fn.sampling_rate)
    
    @property
    def trim_kwargs(self):
        """
            Effective audio-loading kwargs = the static `load_audio_kwargs` plus every trim
            hyper-parameter that has actually been set (i.e. after a `fit`, or explicitly). A trim
            param left unset is *not* forwarded, so `load_audio` keeps its own default for it.
        """
        kwargs = self.load_audio_kwargs.copy()
        for attr, key in _trim_kwargs_keys.items():
            val = getattr(self, attr, None)
            if val is not None: kwargs[key] = val
        return kwargs

    @property
    def use_mel(self):
        return self.audio_format not in ('audio', 'raw')

    @property
    def mel_as_image(self):
        return 'image' in self.audio_format

    @property
    def mel_file(self):
        return os.path.join(self.save_dir, 'mel_fn.json') if self.use_mel else None
    
    @property
    def audio_dtype(self):
        """ Audio / mel are float-typed : follow the runtime's compute precision (fp32 / fp16). """
        return self.base_dtype

    @property
    def audio_shape(self):
        if not self.use_mel:
            return (None, 1)
        elif not self.mel_as_image:
            return (None, self.n_mel_channels)
        else:
            return (None, self.n_mel_channels, 1)

    @property
    def audio_signature(self):
        return TensorSpec(shape = (None, ) + self.audio_shape, dtype = self.audio_dtype)
    
    @property
    def n_mel_channels(self):
        return self.mel_fn.n_mel_channels if self.use_mel else -1
    
    @property
    def training_hparams(self):
        # Merged automatically into the model's hparams through the MRO (`AudioModelMixin` precedes
        # `BaseModel` in every audio model), so leaf models no longer merge it by hand. The concrete
        # defaults are training-only : they reach `trim_kwargs` only once applied at `fit` time.
        return {** super().training_hparams, ** AudioTrainingHParams}

    def _str_audio(self):
        des = "- Audio rate : {}\n".format(self.rate)
        if self.use_mel:
            des += "- # mel channels : {}\n".format(self.n_mel_channels)
        return des
    
    def _get_sample_index(self, time):
        """
            Given a `time` (in second), returns its position in sample (audio) or mel frames (if `self.use_mel`)
        """
        if time is None: return None
        if time == 0.:   return 0
        n_samples   = int(time * self.rate)
        return n_samples if not self.use_mel else self.mel_fn.get_mel_length(n_samples)

    def _get_sample_time(self, samples):
        if samples is None: return None
        elif samples == 0:  return 0.
        elif self.use_mel: samples = self.mel_fn.get_audio_length(samples)
        return samples / self.rate
    
    def get_audio_input(self, data, ** kwargs):
        """
            Loads the audio with the `utils.audio.load_audio` method
            
            Arguments :
                - data  : any value supported by `load_audio`
                - kwargs    : additional kwargs forwarded to `load_audio`
            Return :
                - audio : 2-D `Tensor` with shape `(audio_len, 1)`
        """
        """ Load audio and returns a 2-D `Tensor` with shape `(audio_len, 1)` """
        return load_audio(data, self.rate, ** {** self.trim_kwargs, ** kwargs})[:, None]
    
    def get_mel_input(self, data, ** kwargs):
        """
            Loads the mel-spectrogram by calling `utils.audio.load_mel`
            
            Arguments :
                - data  : any value supported by `load_mel`
                - kwargs    : additional kwargs forwarded to `load_mel`
            Return :
                if `self.mel_as_image`:
                    - mel   : `Tensor` with shape `(n_frames, self.n_mel_channels, 1)`
                else:
                    - mel   : `Tensor` with shape `(n_frames, self.n_mel_channels)`
        """
        mel = load_mel(data, self.mel_fn, ** {** self.trim_kwargs, ** kwargs})
        return mel if not self.mel_as_image else mel[..., None]
    
    def get_audio(self, data, ** kwargs):
        """
            Either calls `self.get_audio_input` or `self.get_mel_input`
            
            Arguments :
                - data  : any value supported by `utils.audio.{load_audio / load_mel}`
                - kwargs    : additional kwargs forwarded to the right function
            Return :
                If `not self.use_mel`:
                    - audio : 2-D `Tensor` with shape `(audio_len, 1)`
                elif `not self.mel_as_image`:
                    - mel   : 2-D `Tensor` with shape `(n_frames, self.n_mel_channels)`
                else:
                    - mel   : 3-D `Tensor` with shape `(n_frames, self.n_mel_channels, 1)`
        """
        if hasattr(data, 'to_dict'): data = data.to_dict('records')
        if isinstance(data, list):
            return [self.get_audio(data_i, ** kwargs) for data_i in data]
        
        if self.use_mel:
            return self.get_mel_input(data, ** kwargs)
        return self.get_audio_input(data, ** kwargs)

    def get_config_audio(self, * args, ** kwargs):
        if self.use_mel and self._save and not os.path.exists(self.mel_file):
            self.mel_fn.save(self.mel_file)

        return {
            'rate'  : self.rate,
            'audio_format'  : self.audio_format,
            'pad_mel_value' : self.pad_mel_value,
            'load_audio_kwargs' : self.load_audio_kwargs,
            
            'mel_fn'    : self.mel_file
        }