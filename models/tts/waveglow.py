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

import math
import logging
import numpy as np

from loggers import Timer, timer
from utils import pad_to_multiple
from utils.keras import TensorSpec, ops
from ..interfaces.base_audio_model import BaseAudioModel

logger = logging.getLogger(__name__)

class WaveGlow(BaseAudioModel):
    _default_loss   = 'mse'
    
    input_signature = BaseAudioModel.audio_signature
    
    def __init__(self, *, mel_fn = 'TacotronSTFT', pad_mel_value = -11., ** kwargs):
        kwargs['audio_format'] = 'mel'
        
        self._init_audio(mel_fn = mel_fn, pad_mel_value = pad_mel_value, ** kwargs)

        super().__init__(** kwargs)
    
    def build(self, *, model = None, vocoder = None, ** kwargs):
        if vocoder is not None: model = vocoder
        elif model is None:
            model = {
                'architecture'  : kwargs.pop('architecture', 'waveglow'),
                'n_mel_channels'    : self.n_mel_channels,
                ** kwargs
            }
        return super().build(model = model)
    
    @property
    def output_signature(self):
        return TensorSpec(shape = (None, None), dtype = 'float32')
    
    def __str__(self):
        return super().__str__() + self._str_audio()
    
    def prepare_for_xla_inference(self, *, inputs, padding_multiple = 256, ** kwargs):
        if padding_multiple and inputs.shape[1] % padding_multiple != 0:
            inputs = pad_to_multiple(
                inputs, padding_multiple, 1, constant_values = self.pad_mel_value
            )
        
        return {'inputs' : inputs}
    
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
    
    __call__ = infer
    
    def get_dataset_config(self, ** kwargs):
        kwargs['pad_kwargs']    = {
            'padding_values'    : (self.pad_mel_value, 0.)
        }
        
        return super().get_dataset_config(**kwargs)

    def get_config(self):
        return {** super().get_config(), ** self.get_config_audio()}

def _get_steps(length, win_len, hop_len):
    num_steps = int(math.ceil((length - win_len) / hop_len)) + 1
    
    if num_steps == 1: return [0]
    
    max_step = length - win_len
    actual_step_size = max_step / (num_steps - 1)

    return np.round(np.arange(num_steps) * actual_step_size).astype(np.int32)
