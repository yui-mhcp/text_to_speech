
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
import logging
import pandas as pd
import tensorflow as tf

from hparams import HParams
from models.interfaces.base_model import BaseModel
from utils.audio import MelSTFT, load_audio, load_mel
from utils.audio import random_pad, random_shift, random_noise

logger = logging.getLogger(__name__)

_supported_audio_format = ('raw', 'audio', 'mel', 'spect', 'spectrogram', 'mel_image')

AudioTrainingHParams = HParams(
    trim_audio   = False,
    reduce_noise = False,
    trim_threshold   = 0.1,
    max_silence  = 0.15,
    trim_method  = 'window',
    trim_mode    = 'start_end',
    
    trim_mel     = False,
    trim_factor  = 0.6,
    trim_mel_method  = 'max_start_end'
)

DEFAULT_MEL_FN_CONFIG  = {
    'filter_length'    : 1024,
    'hop_length'       : 256, 
    'win_length'       : 1024,
    'n_mel_channels'   : 80, 
    'sampling_rate'    : 22050, 
    'mel_fmin'         : 0.0,
    'mel_fmax'         : 8000.0,
    'normalize_mode'   : None,
}

class BaseAudioModel(BaseModel):
    def _init_audio(self,
                    audio_rate  = None,
                    audio_format   = 'mel',
                    
                    mel_fn      = 'TacotronSTFT',
                    mel_fn_config      = DEFAULT_MEL_FN_CONFIG,
                    pad_mel_value      = 0.,
                    
                    mel_fn_type     = None,
                    mel_as_image    = None, # for retro-compatibility
                    
                    ** kwargs
                   ):
        """
            Constructor for the audio-related variables
            
            Arguments :
                - audio_rate    : the audio sampling rate
                - audio_format  : the audio format handled by the model (audio / mel / mel_image)
                - mel_fn    : either the mel's filename or a mel's type
                - mel_config    : the mel spectrogram's config
                - pad_mel_value : value used to pad mel-spectrogram (in batch)
                - mel_fn_type / mel_as_image : for retro-compatibility
        """
        assert audio_format in _supported_audio_format
        
        if mel_as_image is not None:
            logger.warning('`mel_as_image` is deprecated, please use `audio_format` to specify the type of input or call `save_config()` to update the config file')
            audio_format = 'mel' if not mel_as_image else 'mel_image'
        if mel_fn_type is not None:
            logger.warning('`mel_fn_type` is deprecated, please use `mel_fn` to specify the Mel class / file or call `save_config()` to update the config file')
            mel_fn = mel_fn_type
            self.mel_fn_type    = mel_fn_type
        
        self.audio_rate = audio_rate
        self.audio_format   = audio_format
        
        self.pad_mel_value  = pad_mel_value
        
        self.mel_fn = None
        self.mel_fn_config  = mel_fn_config
        if self.use_mel_fn:
            # Initialization of mel fn
            if isinstance(mel_fn, MelSTFT):
                self.mel_fn     = mel_fn
            else:
                if self.audio_rate:
                    mel_fn_config['sampling_rate'] = self.audio_rate
                self.mel_fn    = MelSTFT.create(mel_fn, ** mel_fn_config)
        
        # Assert the configuration is valid / complete
        if not self.audio_rate:
            assert self.use_mel_fn, 'You must specify the `audio_rate` parameter !'
            self.audio_rate = self.mel_fn.sampling_rate
        
        if self.use_mel_fn:
            assert self.audio_rate == self.mel_fn.sampling_rate, 'The `audio_rate` differs from the `mel_fn.sampling_rate` : {} != {}'.format(self.audio_rate, self.mel_fn.sampling_rate)
    
    def init_train_config(self, ** kwargs):
        if not hasattr(self, 'trim_kwargs'): self.trim_kwargs = {}
        
        super(BaseAudioModel, self).init_train_config(** kwargs)
        
        if not self.use_mel_fn: self.trim_mel = False
        if hasattr(self, 'trim_mel') and not self.trim_mel: self.trim_mel_method = None
            
    def _update_trim_config(self, key, val):
        self.trim_kwargs[key] = val

    @property
    def use_mel_fn(self):
        return self.audio_format not in ('audio', 'raw')

    @property
    def mel_as_image(self):
        return 'image' in self.audio_format

    @property
    def mel_fn_file(self):
        return os.path.join(self.save_dir, 'mel_fn.json') if self.use_mel_fn else None
    
    @property
    def audio_signature(self):
        if not self.use_mel_fn:
            shape = (None, None, 1)
        elif not self.mel_as_image:
            shape = (None, None, self.n_mel_channels)
        else:
            shape = (None, None, self.n_mel_channels, 1)
        
        return tf.TensorSpec(shape = shape, dtype = tf.float32)
    
    @property
    def n_mel_channels(self):
        return self.mel_fn.n_mel_channels if self.use_mel_fn else -1
    
    @property
    def training_hparams_audio(self):
        return AudioTrainingHParams()
    
    @property
    def training_hparams_mapper(self):
        mapper = super().training_hparams_mapper
        mapper.update({
            'trim_audio'    : lambda val: self._update_trim_config('trim_silence', val),
            'reduce_noise'  : lambda val: self._update_trim_config('reduce_noise', val),
            'trim_method'   : lambda val: self._update_trim_config('method', val),
            'trim_mode'     : lambda val: self._update_trim_config('mode', val),
            'trim_factor'   : lambda val: self._update_trim_config('min_factor', val),
            'trim_threshold'    : lambda val: self._update_trim_config('threshold', val),
            'max_silence'   : lambda val: self._update_trim_config('max_silence', val)
        })
        return mapper

    
    def _str_audio(self):
        des = "- Audio rate : {}\n".format(self.audio_rate)
        if self.use_mel_fn:
            des += "- # mel channels : {}\n".format(self.n_mel_channels)
        return des
    
    def get_audio_input(self, data, ** kwargs):
        """ Load audio and returns a 2-D `tf.Tensor` with shape `(1, audio_len)` """
        kwargs  = {** self.trim_kwargs, ** kwargs}
        audio   = load_audio(data, self.audio_rate, ** kwargs)
        
        return tf.expand_dims(audio, axis = 1)
    
    def get_mel_input(self, data, ** kwargs):
        """
            Load the mel-spectrogram and returns :
                if `self.mel_as_image` : a 3-D `tf.Tensor` with shape `(n_frames, n_channels, 1)`
                else: a 2-D `tf.Tensor` with shape `(n_frames, n_channels)`
        """
        kwargs = {** self.trim_kwargs, ** kwargs}
        mel = load_mel(
            data,
            self.mel_fn,
            trim_mode   = self.trim_mel_method,
            ** kwargs
        )
        
        if len(tf.shape(mel)) == 3: mel = tf.squeeze(mel, 0)
        if self.mel_as_image:
            mel = tf.expand_dims(mel, axis = -1)
        
        return mel
    
    def get_audio(self, data, ** kwargs):
        """ Either calls `get_mel_input` or `get_audio_input` depending on `audio_format` """
        if isinstance(data, list):
            return [self.get_audio(data_i, ** kwargs) for data_i in data]
        elif isinstance(data, pd.DataFrame):
            return [self.get_audio(row, ** kwargs) for idx, row in data.iterrows()]
        
        if self.use_mel_fn:
            input_data = self.get_mel_input(data, ** kwargs)
        else:
            input_data = self.get_audio_input(data, ** kwargs)
        
        return input_data
    
    def _augment_audio(self, audio, max_length = -1):
        """
            Augment `audio` with random noise and random shift / padding if `max_length > len(audio)`
        """
        if max_length > len(audio):
            audio = random_shift(audio, min_length = max_length)
            audio = random_pad(audio, max_length)
        
        audio = tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: random_noise(audio),
            lambda: audio
        )
        
        return audio
    
    def _augment_mel(self, mel, max_length = -1):
        """
            Augment `mel` with random noise and random shift / padding if `max_length > len(audio)`
        """
        if max_length > len(mel):
            max_padding = max_length - tf.shape(mel)[0]
            if max_padding > 0:
                padding_left = tf.random.uniform(
                    (),
                    minval = 0, 
                    maxval = max_padding,
                    dtype  = tf.int32
                )

                if max_padding - padding_left > 0:
                    padding_right = tf.random.uniform(
                        (),
                        minval = 0, 
                        maxval = max_padding - padding_left,
                        dtype = tf.int32
                    )
                else:
                    padding_right = 0

                if self.mel_as_image:
                    padding = [(padding_left, padding_right), (0, 0), (0, 0)]
                else:
                    padding = [(padding_left, padding_right), (0, 0)]

                mel = tf.pad(mel, padding)
        
        
        return tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: mel + tf.random.normal(tf.shape(mel)),
            lambda: mel
        )
    
    def augment_audio(self, inp, ** kwargs):
        """ Either cals `_augment_audio` or `_augment_mel` depending on `audio_format` """
        if self.use_mel_fn:
            return self._augment_mel(inp, ** kwargs)
        else:
            return self._augment_audio(inp, ** kwargs)

    def get_config_audio(self, * args, ** kwargs):
        if self.use_mel_fn and not os.path.exists(self.mel_fn_file):
            self.mel_fn.save_to_file(self.mel_fn_file)

        return {
            'audio_rate'    : self.audio_rate,
            'audio_format'  : self.audio_format,
            'pad_mel_value' : self.pad_mel_value,
            
            'mel_fn'    : self.mel_fn_file
        }
