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

import re
import logging
import itertools
import subprocess
import numpy as np
import librosa.util as librosa_util

from scipy.signal import get_window, resample

from loggers import timer
from ..wrappers import dispatch_wrapper

logger = logging.getLogger(__name__)

_trimming_methods = {}
_ffmpeg_silence_re  = re.compile(r' silence_(start|end): ([0-9]+(?:\.[0-9]*)?)\b')

@timer
def resample_audio(audio, rate, target_rate):
    if rate == target_rate: return audio, rate
    
    audio   = resample(audio, int(len(audio) / rate * target_rate))
    return audio, target_rate

@timer
def convert_audio_dtype(audio, dtype):
    """ Converts `audio` to `dtype` by normalizing by the `dtype` or `audio.dtype` max value """
    if audio.dtype == dtype: return audio
    elif np.issubdtype(audio.dtype, np.floating):
        if np.issubdtype(dtype, np.floating):
            return audio.astype(dtype)
        else:
            return (audio * np.iinfo(dtype).max).astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        return (audio / np.iinfo(audio.dtype).max).astype(dtype)
    else:
        return (audio / np.iinfo(audio.dtype).max * np.iinfo(dtype).max).astype(dtype)

@timer
def normalize_audio(audio, max_val = 32767, dtype = np.int16):
    """
        Normalize audio either to np.int16 (default) or on [-1, 1] range (max_val = 1.) with dtype np.float32
    """
    if max_val <= 1.: dtype = np.float32
    
    audio = audio - np.mean(audio)
    
    max_audio_val = np.max(np.abs(audio))
    if max_audio_val <= 1e-9: return audio.astype(dtype)
    
    return (audio * (max_val / max_audio_val)).astype(dtype)

@timer
def reduce_noise(audio, *, noise_length = 0.2, rate = None, noise = None, use_v1 = True, ** kwargs):
    """
        Use the noisereduce.reduce_noise method in order to reduce audio noise
        It takes as 'noise sample' the 'noise_length' first seconds of the audio (if `noise is None`)
    """
    if noise is None:
        if isinstance(noise_length, float):
            assert rate is not None
            noise_length = int(noise_length * rate)
        
        noise = audio[: noise_length]
    
    if use_v1:
        from .noisereducev1 import reduce_noise
        return reduce_noise(audio, noise)
    else:
        from noisereduce import reduce_noise
        return reduce_noise(audio, sr = rate, y_noise = noise)

@dispatch_wrapper(_trimming_methods, 'method', default = 'window')
@timer
def trim_silence(audio, *, method = 'rms', ** kwargs):
    """
        Removes silence with the given `method` (see below for available techniques)
        
        Arguments : 
            - audio : `np.ndarray`, the audio to process
            - method    : the trim method to use
            - kwargs    : kwargs to passto the trim function
        Return :
            - trimmed_audio : `np.ndarray`, the audio with silence trimmed
    """
    if isinstance(method, bytes): method = method.decode()
    return _trimming_methods[method](audio, ** kwargs)

@trim_silence.dispatch
def trim_silence_rms(audio,
                     *,
                     rate,

                     mode    = 'start_end',
                     threshold   = -25,
                     min_silence = 0.1,
                     
                     block_size = 0.01,

                     replace_by  = 0.5,
                     min_voice_time  = 0.2,
                     
                     ** kwargs
                    ):
    """
        This method detects and removes silences based on Root-Max-Square of the audio amplitude
        This method gives similar results compared to `ffmpeg silencedetect` filter, while being **much faster**, as it is running with full `numpy` operations
        
        Arguments :
            - audio : 1D `np.ndarray`, the audio data
            - rate  : the audio sampling rate
            
            - mode  : the silence removal mode
                      - remove  : removes all silences
                      - start   : removes silence at the start of the audio
                      - end     : removes silence at the end of the audio
                      - start_end   : removes silence at the start & end of the audio
            - threshold : the threshold (in dB) below which it is considered as silence
            - min_silence   : the minimal silence duration
                              equivalent to the `duration` of `ffmpeg silencedetect`
            
            - replace_by    : replace each silence by a shorter silence
                              in order to produce smoother results.
                              Set to 0 to disable
            - min_voice_time    : minimum time between silence to keep
                                  if non-silence is shorter, it is removed.
                                  
            - block_size    : 
    """
    if isinstance(mode, bytes):       mode = mode.decode()
    if isinstance(replace_by, float): replace_by = int(replace_by * rate)
    if isinstance(block_size, float): block_size = int(block_size * rate)
    block_time = block_size / rate
    
    amplitude_threshold = 10 ** (threshold / 20.0)
    
    blocks = audio # / np.max(np.abs(audio))
    if not np.issubdtype(blocks.dtype, np.floating): blocks = blocks / np.iinfo(blocks.dtype).max
    
    pad = len(blocks) % block_size
    if pad != 0: blocks = np.pad(blocks, [(0, block_size - pad)])
    
    blocks  = blocks.reshape(-1, block_size)
    rms     = np.sqrt(np.max(blocks ** 2, axis = 1))
    
    silent_blocks = rms < amplitude_threshold
    
    idx = 0
    starts, ends = [], []
    for is_silent, group in itertools.groupby(silent_blocks):
        n = len(list(group))
        if is_silent and n * block_time >= min_silence:
            starts.append(idx * block_time)
            ends.append(min(len(audio) / rate, (idx + n) * block_time))
        idx += n
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Silence detected at {}'.format(', '.join([
            '[{:.3f} : {:.3f}]'.format(s, e) for s, e in zip(starts, ends)
        ])))
    
    if min_voice_time:
        for i in reversed(range(len(starts) - 1)):
            if starts[i + 1] - ends[i] < min_voice_time:
                starts.pop(i + 1)
                ends.pop(i)

    if mode == 'remove':
        replace_half    = replace_by // 2
        
        _is_silence = np.zeros((len(audio), ), dtype = bool)
        for start, end in zip(starts, ends):
            start, end = int(start * rate), int(end * rate)
            if start == 0:
                _is_silence[: max(0, end - replace_by)] = True
            elif abs(end - len(audio)) <= 1:
                _is_silence[start + replace_by :] = True
            else:
                start, end = start + replace_half, end - replace_half
                if start < end: _is_silence[start : end] = True
                
        audio = audio[~_is_silence]
    else:
        if 'end' in mode and abs(ends[-1] * rate - len(audio)) <= 1:
            audio = audio[: int(starts[-1] * rate) + replace_by]
        if 'start' in mode and starts[0] == 0:
            audio = audio[max(0, int(ends[0] * rate) - replace_by) :]
    
    return audio

@trim_silence.dispatch
def trim_silence_ffmpeg(audio,
                        *,
                        rate,
                        format  = None,
                        
                        mode    = 'start_end',
                        threshold   = -25,
                        min_silence = 0.1,
                        
                        replace_by  = 0.5,
                        min_voice_time  = 0.2,
                        
                        
                        ** kwargs
                       ):
    import ffmpeg
    
    if isinstance(mode, bytes):       mode = mode.decode()
    if isinstance(replace_by, float): replace_by = int(replace_by * rate)
    if format is None: format = 'f32le' if audio.dtype == 'float32' else 's16le'
    
    result = subprocess.Popen(
        ffmpeg \
            .input('pipe:0', format = format, ac = 1, ar = rate)
            .filter('silencedetect', n = '{}dB'.format(threshold), d = float(min_silence)) \
            .output('pipe:0', format = 'null') \
            .compile() + ['-nostats'],
        stdin  = subprocess.PIPE,
        stderr = subprocess.PIPE
    ).communicate(audio.tobytes())[1].decode('utf-8')
    
    silences = _ffmpeg_silence_re.findall(result)
    if not silences: return audio
    
    starts  = [float(v) for _, v in silences[::2]]
    ends    = [float(v) for _, v in silences[1::2]]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Silence detected at {}'.format(', '.join([
            '[{:.3f} : {:.3f}]'.format(s, e) for s, e in zip(starts, ends)
        ])))

    if min_voice_time:
        for i in reversed(range(len(starts) - 1)):
            if starts[i + 1] - ends[i] < min_voice_time:
                starts.pop(i + 1)
                ends.pop(i)

    if mode == 'remove':
        replace_half    = replace_by // 2
        
        _is_silence = np.zeros((len(audio), ), dtype = bool)
        for start, end in zip(starts, ends):
            start, end = int(start * rate), int(end * rate)
            if start == 0:
                _is_silence[: max(0, end - replace_by)] = True
            elif abs(end - len(audio)) <= 1:
                _is_silence[start + replace_by :] = True
            else:
                start, end = start + replace_half, end - replace_half
                if start < end: _is_silence[start : end] = True
                
        audio = audio[~_is_silence]
    else:
        if 'end' in mode and abs(ends[-1] * rate - len(audio)) <= 1:
            audio = audio[: int(starts[-1] * rate) + replace_by]
        if 'start' in mode and starts[0] == 0:
            audio = audio[max(0, int(ends[0] * rate) - replace_by) :]
    
    return audio

@trim_silence.dispatch
def trim_silence_window(audio,
                        *,
                        
                        rate,

                        power   = 2,
                        mode    = 'start_end',
                        
                        threshold   = 0.1,
                        adaptive_threshold  = True,
                        
                        window_type = 'triangular',
                        window_length   = 0.2,
                        add_start   = 0,
                        add_end     = 1.5,
                        
                        max_trim_factor = 5,
                        
                        debug       = False,
                        plot_kwargs = {},
                        
                        ** kwargs
                       ):
    """
        Trims silences at start / end (or both) with windowed-based thresholding
        
        Arguments :
            - audio : the audio to trim
            - rate  : the audio rate (only used if `window_length` is a float)
            
            - power : the power to use to smooth the audio
            - mode  : either 'start', 'end' or 'start_end' (where to trim)
            
            - threshold : the threshold to use
            - adaptive_threshold    : if True, the threshold is the median between `threshold`, `thredho / 25` and `np.mean(conv[...]) * 5`. It allows to dynamically reduce the threshold if the start / end of the audio is already small
            
            - window_type   : the shape for the convolution window
            - window_length : the length for the window (if float, `window_length = int(window_length * rate)`)
            
            - max_trim_factor   : returns the original audio if the trimmed length is smaller than the length of the original audio divided by this value
            
            - debug / plot_kwargs   : whether to plot the trimming
            - kwargs    : unused
        Returns : the trimmed audio
    """
    if isinstance(mode, bytes): mode = mode.decode()
    if isinstance(window_type, bytes): window_type = window_type.decode()
    assert mode in ('start', 'end', 'start_end'), 'Invalid mode : {}'.format(mode)
    
    if isinstance(window_length, float): window_length = int(window_length * rate)
    
    if window_type == 'mean':
        window = np.ones((window_length,)) / window_length
    elif window_type == 'linear':
        window = np.arange(window_length) / window_length
    elif window_type == 'triangular':
        window = np.concatenate([
            np.linspace(0, 1, window_length // 2),
            np.linspace(1, 0, window_length // 2)
        ]) / (window_length // 2)

    powered = np.power(audio, power)
    conv    = np.convolve(powered, window, mode = 'valid')

    trimmed = audio
    idx_start, idx_end = [], []
    th_start, th_end = threshold, threshold
    if 'end' in mode:
        if adaptive_threshold:
            th_end  = min(threshold, max(np.mean(conv[-window_length:]) * 5, threshold / 50))
        
        idx_end = np.where(conv > th_end)[0]
        if len(idx_end) > 0:
            trimmed = trimmed[:idx_end[-1] + int(window_length * add_end)]
    
    if 'start' in mode:
        if adaptive_threshold:
            th_start = min(threshold, max(np.mean(conv[: window_length]) * 5, threshold / 50))
        idx_start = np.where(conv > th_start)[0]
        if len(idx_start) > 0:
            trimmed = trimmed[max(0, idx_start[0] - int(window_length * add_start)) :]
    
    if debug:
        from utils.plot_utils import plot_multiple
        plot_kwargs.setdefault('color', 'red')
        plot_kwargs.setdefault('ncols', 1)
        plot_kwargs.setdefault('x_size', 10)
        plot_kwargs.setdefault('vlines_kwargs', {'colors' : 'w'})
        plot_multiple(
            powered_audio = powered, convolved = conv,
            ylim = (0, threshold), hlines = (th_start, th_end), vlines = (
                idx_start[0] if len(idx_start) else 0, idx_end[-1] if len(idx_start) else len(audio)
            ), use_subplots = True, ** plot_kwargs
        )
    
    return trimmed if len(trimmed) > len(audio) // max_trim_factor else audio

@trim_silence.dispatch('remove')
def remove_silence(audio, *, rate, threshold = 0.025, min_silence = 0.15,  ** kwargs):
    """
        Create a mean-window, compute the convolution of it and remove all zones that are lower than the threshold. 
    """
    window_length = int(min_silence * rate)

    mask = np.ones((window_length,)) / (window_length * threshold)

    conv = np.convolve(np.square(audio), mask, mode = 'same')
    
    return audio[conv > min(threshold, np.mean(conv) / 2)]
    
@trim_silence.dispatch('threshold')
def trim_silence_simple(audio, *, threshold = 0.1, mode = 'start_end', ** kwargs):
    assert mode in ('start', 'end', 'start_end')
    
    indexes = np.where(np.abs(audio - np.mean(audio)) > threshold)[0]
    if len(indexes) > 0:
        if 'end' in mode: audio = audio[ : indexes[-1]]
        if 'start' in mode: audio = audio[indexes[0] : ]
            
    return audio

