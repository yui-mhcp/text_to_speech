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

import math
import logging
import numpy as np

from utils import pad_to_multiple
from loggers import timer, time_logger
from models.interfaces import BaseModel
from utils.keras_utils import TensorSpec, ops, graph_compile
from custom_architectures import get_architecture

logger = logging.getLogger(__name__)

DEFAULT_MAX_MEL_LENGTH  = 1024

_pytorch_waveglow   = None

def get_nvidia_waveglow():
    global _pytorch_waveglow
    if _pytorch_waveglow is None:
        from utils.generic_utils import limit_gpu_memory

        limit_gpu_memory()
        _pytorch_waveglow = get_architecture('nvidia_waveglow')
    return _pytorch_waveglow

def _get_steps(length, win_len, hop_len):
    num_steps = int(math.ceil((length - win_len) / hop_len)) + 1
    
    if num_steps == 1: return [0]
    
    max_step = length - win_len
    actual_step_size = max_step / (num_steps - 1)

    return np.round(np.arange(num_steps) * actual_step_size).astype(np.int32)

class WaveGlow(BaseModel):
    def __init__(self,
                 audio_rate         = 22050,
                 n_mel_channels     = 80,
                 pad_mel_value      = -11.,
                 max_input_length   = DEFAULT_MAX_MEL_LENGTH,
                 run_eagerly    = False,
                 ** kwargs
                ):
        self.audio_rate         = audio_rate
        self.n_mel_channels     = n_mel_channels
        self.pad_mel_value      = pad_mel_value
        self.max_input_length   = max_input_length

        super().__init__(** kwargs)
    
    def build(self, vocoder = None, ** kwargs):
        if not vocoder:
            vocoder = {
                'architecture'  : kwargs.pop('architecture', 'waveglow'),
                'n_mel_channels'    : self.n_mel_channels,
                ** kwargs
            }
        return super().build(vocoder = vocoder)
    
    @property
    def run_eagerly(self):
        return True
    
    @property
    def input_signature(self):
        return TensorSpec(shape = (None, None, self.n_mel_channels), dtype = 'float32')
    
    @property
    def output_signature(self):
        return TensorSpec(shape = (None, None), dtype = 'float32')
        
    @property
    def training_hparams(self):
        return super().training_hparams(max_input_length = None)
    
    def __str__(self):
        des = super().__str__()
        des += "- Audio rate : {}\n".format(self.audio_rate)
        des += "- Mel channels : {}\n".format(self.n_mel_channels)
        return des
    
    def prepare_for_xla_inference(self, *, inputs, padding_multiple = 256, ** kwargs):
        if padding_multiple and inputs.shape[1] % padding_multiple != 0:
            inputs = pad_to_multiple(
                inputs, padding_multiple, 1, constant_values = self.pad_mel_value
            )
        
        return {'inputs' : inputs}

    def __call__(self, spect, * args, training = False, ** kwargs):
        return self.infer(spect, ** kwargs)
    
    @timer(name = 'inference WaveGlow')
    def infer(self,
              mel,
              *,
              win_len   = None,
              hop_len   = -64,
              force_pad = None,
              
              batch     = False,
              use_slice = False,
              max_win_len   = None,
              
              ** kwargs
             ):
        if isinstance(mel, str):    mel = np.load(mel)
        if len(mel.shape) == 2:     mel = ops.expand_dims(mel, axis = 0)

        seq_len     = mel.shape[1]
        audio_len   = seq_len * 256
        if win_len is None:
            return self.compiled_infer(mel, ** kwargs)[:, : audio_len]

        if isinstance(win_len, float):
            if not use_slice:
                win_len = int(math.ceil(seq_len / win_len) * win_len)
            else:
                win_len = max(1, seq_len // win_len) * int(win_len)

        if max_win_len is not None: win_len = min(max_win_len, win_len)
        
        kwargs['padding_multiple'] = win_len
        
        if seq_len <= win_len:
            if force_pad is None: force_pad = self.runtime == 'keras'
            if not force_pad: return self.compiled_infer(mel)
            
            win_len = max(win_len, seq_len)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Padding mel with shape {} to {} frames'.format(
                    mel.shape, win_len
                ))
            
            return self.compiled_infer(ops.pad(
                mel, [(0, 0), (0, win_len - seq_len), (0, 0)],
                constant_values = self.pad_mel_value
            ), ** kwargs)[:, : audio_len]
        elif mel.shape[0] > 1:
            logger.info('Batch size is higher than 1 ({}), performing direct inference !'.format(
                mel.shape
            ))
            return self.compiled_infer(mel, ** kwargs)

        win_len, hop_len = win_len, hop_len
        if isinstance(hop_len, float):  hop_len = int(win_len * hop_len)
        if hop_len < 0:                 hop_len = win_len + hop_len

        starts  = _get_steps(seq_len, win_len, hop_len)
        parts   = [mel[:, start : start + win_len] for start in starts]
        overlaps    = ((starts[:-1] + win_len) - starts[1:]) * 256
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Starts : {} - mel shapes : {}'.format(
                starts, [tuple(p.shape) for p in parts]
            ))

        if batch:
            audio_parts = self.compiled_infer(ops.concatenate(parts, axis = 0), ** kwargs)
            audio_parts = ops.convert_to_numpy(audio_parts)
        else:
            audio_parts = [
                self.compiled_infer(p, ** kwargs)[0] for p in parts
            ]
            audio_parts = [ops.convert_to_numpy(a) for a in audio_parts]

        audio = []
        for i, part in enumerate(audio_parts):
            start = 0 if i == 0 else overlaps[i - 1] // 2
            end   = None if i == len(audio_parts) - 1 else -overlaps[i] // 2
            audio.append(part[start : end])

        return ops.concatenate(audio, axis = -1)
    
    def compile(self, loss = 'mse', metrics = [], ** kwargs):
        super().compile(loss = loss, metrics = metrics, ** kwargs)
    
    def get_dataset_config(self, ** kwargs):
        kwargs['pad_kwargs']    = {
            'padding_values'    : (self.pad_mel_value, 0.)
        }
        kwargs['padded_batch']      = True
        
        return super().get_dataset_config(**kwargs)

    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            'audio_rate'    : self.audio_rate,
            'n_mel_channels'    : self.n_mel_channels,
            'pad_mel_value'     : self.pad_mel_value,
            'max_input_length'  : self.max_input_length
        })
        
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
    
    @property
    def audio_rate(self):
        return 22050
    
    def __call__(self, spect, * args, training = False, ** kwargs):
        return self.infer(spect)
    
    @timer(name = 'WaveGlow inference')
    def infer(self, mels):
        import torch
        
        if isinstance(mels, str): mels = np.load(mels)
        if len(mels.shape) == 2: mels = np.expand_dims(mels, axis = 0)
        
        mels = np.transpose(mels, [0, 2, 1])
        with torch.no_grad():
            mels = torch.FloatTensor(mels).cuda()
            
            audios = self.waveglow.infer(mels).cpu().numpy()
        return audios
    