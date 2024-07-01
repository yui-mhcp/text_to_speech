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

import numpy as np
import random as py_random

from utils.keras_utils import ops
from utils.audio.audio_io import load_audio

def silence(duration, rate):
    return np.zeros((int(duration * rate),))

def concat(* audios, rate = 22050, silence_time = 0.15):
    if silence_time <= 0.: return np.concatenate(audios)
    result = []
    silence = silence(silence_time, rate)
    for audio in audios:
        result += [audio, silence]
    return np.concatenate(result)

def pad(audio, left_time, right_time, rate = 22050, **kwargs):
    return np.pad(audio, [int(left_time * rate), int(right_time * rate)], ** kwargs)

def merge(* list_audios, rate = 22050, overlap = [-1., 1.], intensite = 1.):
    if isinstance(overlap, (int, float)): overlap = [overlap, overlap]
    def random_overlap():
        return py_random.random() * (overlap[1] - overlap[0]) + overlap[0]
    
    audios = []
    for audio in list_audios:
        if isinstance(audio, str):
            audio = load_audio(audio, target_rate = rate)
        audios.append(audio)
    
    infos = {
        'time'  : [],
        'start' : [],
        'end'   : []
    }
    total_time = 0.
    for audio in audios:
        duree   = len(audio) / rate
        start   = max(0., total_time + random_overlap())
        fin     = start + duree
        total_time = fin
        
        infos['time'].append(duree)
        infos['start'].append(start)
        infos['end'].append(fin)
    
    infos['total_time'] = total_time
        
    merged = np.zeros((int(total_time * rate) + 1,))
    overlap_mask = np.zeros((int(total_time * rate) + 1,))
    for i, audio in enumerate(audios):
        s = int(rate * infos['start'][i])
        facteur = intensite[i] if isinstance(intensite, (list, tuple)) else intensite
        
        merged[s : s + len(audio)] += (audio * intensite)
        overlap_mask[s : s + len(audio)] += 1
    
    merged = np.divide(merged, overlap_mask, where = overlap_mask > 0)
    
    return merged, infos

def random_shift(audio, min_shift = 0, max_shift = 0.5, min_length = None):
    length = len(audio)
    
    if min_length: max_shift = length - min_length - min_shift
    elif max_shift < 1.: max_shift = ops.cast(max_shift * length, 'int32')
    
    if max_shift > 0:
        shift = ops.random.randint(
            (), minval = min_shift,  maxval = max_shift
        )
        
        audio = audio[shift :]
    
    return audio

def random_pad(audio, max_length):
    maxval = max_length - len(audio)
    if maxval > 0:
        padding_left = ops.random.randint(
            (), minval = 0, maxval = maxval
        )
        
        padding_right   = ops.cond(
            maxval - padding_left > 0,
            lambda: ops.random.randint((), minval = 0, maxval = maxval - padding_left),
            lambda: ops.convert_to_tensor(0, 'int32')
        )
        
        if len(ops.shape(audio)) == 2:
            padding = [(padding_left, padding_right), (0, 0)]
        else:
            padding = [(padding_left, padding_right)]
            
        audio = ops.pad(audio, padding)
            
    return audio

def random_noise(audio, intensity = None, max_intensity = 0.2):
    if intensity is None:
        intensity = ops.random.uniform((), minval = 0., maxval = max_intensity)
    
    noize = ops.random.uniform(ops.shape(audio), minval = ops.min(audio), maxval = ops.max(audio))
    return audio * (1. - intensity) + noise * intensity