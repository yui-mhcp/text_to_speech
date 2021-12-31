import os
import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from models.base_model import BaseModel
from custom_architectures import get_architecture
from models.weights_converter import pt_convert_model_weights
from utils.generic_utils import limit_gpu_memory

DEFAULT_MAX_MEL_LENGTH  = 1024

_pytorch_waveglow   = None

def get_nvidia_waveglow():
    global _pytorch_waveglow
    if _pytorch_waveglow is None:
        limit_gpu_memory()
        _pytorch_waveglow = get_architecture('nvidia_waveglow')
    return _pytorch_waveglow

class WaveGlow(BaseModel):
    def __init__(self,
                 audio_rate         = 22050,
                 n_mel_channels     = 80,
                 max_input_length   = DEFAULT_MAX_MEL_LENGTH,
                 **kwargs
                ):
        self.audio_rate         = audio_rate
        self.n_mel_channels     = n_mel_channels
        self.max_input_length   = max_input_length

        super().__init__(**kwargs)
        
        if hasattr(self.vocoder, '_build'): self.vocoder._build()
        
    def init_train_config(self, max_input_length = None, ** kwargs):
        if max_input_length: self.max_input_length   = max_input_length
        super().init_train_config(** kwargs)
        
    def _build_model(self, **kwargs):
        super()._build_model(
            vocoder = {
                'architecture_name' : kwargs.pop('architecture_name', 'waveglow'),
                'n_mel_channels'    : self.n_mel_channels,
                ** kwargs
            }
        )
        
    @property
    def input_signature(self):
        return tf.TensorSpec(
            shape = (None, None, self.n_mel_channels), dtype = tf.float32
        )
    
    @property
    def output_signature(self):
        return tf.TensorSpec(shape = (None, None), dtype = tf.float32)
        
    @property
    def training_hparams(self):
        return super().training_hparams(max_input_length = None)
    
    def __str__(self):
        des = super().__str__()
        des += "Audio rate : {}\n".format(self.audio_rate)
        des += "Mel channels : {}\n".format(self.n_mel_channels)
        return des
    
    def call(self, spect, * args, training = False, ** kwargs):
        return self.infer(spect)
    
    @timer(name = 'inference')
    def infer(self, spect, * args, ** kwargs):
        if isinstance(spect, str): spect = np.load(spect)
        if len(spect.shape) == 2: spect = tf.expand_dims(spect, axis = 0)
            
        return self.vocoder.infer(spect, * args, ** kwargs)
    
    def compile(self, loss = 'mse', metrics = [], **kwargs):
        super().compile(loss = loss, metrics = metrics, ** kwargs)
    
    def get_dataset_config(self, **kwargs):
        kwargs['pad_kwargs']    = {
            'padded_shapes'     : (
                (None, self.n_mel_channels), (None,)
            ),
            'padding_values'    : (-11., 0.)
        }
        kwargs['padded_batch']      = True
        
        return super().get_dataset_config(**kwargs)

    def get_config(self, *args, **kwargs):
        config = super().get_config(*args, **kwargs)
        config['audio_rate']         = self.audio_rate
        config['n_mel_channels']     = self.n_mel_channels
        config['max_input_length']   = self.max_input_length
        
        return config

    @classmethod
    def from_nvidia_pretrained(cls, nom = None, ** kwargs):            
        nvidia_model = get_nvidia_waveglow()
        
        instance = cls(
            nom = nom, max_to_keep = 1, pretrained_name = 'pytorch_nvidia_waveglow', ** kwargs
        )
        
        pt_convert_model_weights(nvidia_model, instance.vocoder)
        
        instance.save()
        
        return instance
    
class PtWaveGlow(object):
    def __init__(self, ** kwargs):
        self.waveglow = get_nvidia_waveglow()
    
    def __call__(self, spect, * args, training = False, ** kwargs):
        return self.infer(spect)
    
    @timer(name = 'inference')
    def infer(self, mels):
        import torch
        
        if isinstance(mels, str): mels = np.load(mels)
        if len(mels.shape) == 2: mels = np.expand_dims(mels, axis = 0)
        
        mels = np.transpose(mels, [0, 2, 1])
        with torch.no_grad():
            mels = torch.FloatTensor(mels).cuda()
            
            audios = self.waveglow.infer(mels).cpu().numpy()
        return audios
    