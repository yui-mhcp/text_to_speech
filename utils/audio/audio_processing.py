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

import enum
import numpy as np
import librosa.util as librosa_util

from scipy.signal import get_window

from loggers import timer
from utils.keras_utils import ops
from utils import get_enum_item, convert_to_str, dispatch_wrapper

_trimming_methods = {}

class WindowType(enum.IntEnum):
    MEAN    = 0
    LINEAR  = 1
    TRIANGULAR  = 2

@dispatch_wrapper(_trimming_methods, 'method', default = 'window')
@timer
def trim_silence(audio, method = 'window', ** kwargs):
    """
        Removes silence with the given `method` (see below for available techniques)
        
        Arguments : 
            - audio : `np.ndarray`, the audio to process
            - method    : the trim method to use
            - kwargs    : kwargs to passto the trim function
        Return :
            - trimmed_audio : `np.ndarray`, the audio with silence trimmed
    """
    method = convert_to_str(method)
    return _trimming_methods.get(method, trim_silence_simple)(audio, ** kwargs)

@trim_silence.dispatch
def trim_silence_window(audio,
                        rate    = 22050,
                        power   = 2,
                        mode    = 'start_end',
                        
                        threshold   = 0.1,
                        adaptive_threshold  = True,
                        
                        window_type = WindowType.TRIANGULAR,
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
    mode = convert_to_str(mode)
    assert mode in ('start', 'end', 'start_end'), 'Invalid mode : {}'.format(mode)
    window_type = get_enum_item(window_type, WindowType)
    
    if isinstance(window_length, float): window_length = int(window_length * rate)
    
    if window_type == WindowType.MEAN:
        window = np.ones((window_length,)) / window_length
    elif window_type == WindowType.LINEAR:
        window = np.arange(window_length) / window_length
    elif window_type == WindowType.TRIANGULAR:
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
def remove_silence(audio, rate = 22050, threshold = 0.025, max_silence = 0.15,  ** kwargs):
    """
        Create a mean-window, compute the convolution of it and remove all zones that are lower than the threshold. 
    """
    window_length = int(max_silence * rate)

    mask = np.ones((window_length,)) / (window_length * threshold)

    conv = np.convolve(np.square(audio), mask, mode = 'same')
    
    return audio[conv > min(threshold, np.mean(conv) / 2)]
    
@trim_silence.dispatch('threshold')
def trim_silence_simple(audio, threshold = 0.1, mode = 'start_end', ** kwargs):
    assert mode in ('start', 'end', 'start_end')
    
    indexes = np.where(np.abs(audio - np.mean(audio)) > threshold)[0]
    if len(indexes) > 0:
        if 'end' in mode: audio = audio[ : indexes[-1]]
        if 'start' in mode: audio = audio[indexes[0] : ]
            
    return audio

@trim_silence.dispatch
def trim_silence_mel(mel, mode = 'start_end', min_factor = 0.5, ** kwargs):
    min_amp, max_amp = ops.min(mel), ops.max(mel)
    
    min_val = min_amp + (max_amp - min_amp) * min_factor
    
    if mode == 'mean':
        return mel[ops.mean(mel, axis = -1) >= min_val]
    
    elif mode == 'max':
        return mel[ops.max(mel, axis = -1) >= min_val]
    
    else:
        if 'max' in mode:
            frames_amp = ops.max(mel, axis = -1)
        else:
            frames_amp = ops.mean(mel, axis = -1)
                    
        start, stop = 0, len(mel) - 1
        if 'end' in mode:
            while stop >= 0 and frames_amp[stop] < min_val: stop = stop - 1
        if 'start' in mode:
            while start < stop and frames_amp[start] < min_val: start = start + 1
        
        if stop == start: start, stop = 0, ops.shape(mel)[0]
                
        return mel[start : stop]
    
@timer
def reduce_noise(audio, noise_length = 0.2, rate = None, noise = None, use_v1 = True, ** kwargs):
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
        import utils.audio.noisereducev1 as nr
        return nr.reduce_noise(audio, noise)
    else:
        import noisereduce as nr
        return nr.reduce_noise(audio, sr = rate, y_noise = noise)
    

@timer
def convert_audio_dtype(audio, dtype):
    """ Converts `audio` to `dtype` by normalizing by the `dtype` or `audio.dtype` max value """
    if audio.dtype == dtype: return audio
    if np.issubdtype(audio.dtype, np.floating) and np.issubdtype(dtype, np.floating):
        return audio.astype(dtype)
    
    if np.issubdtype(audio.dtype, np.integer): audio = audio / np.iinfo(audio.dtype).max
    if np.issubdtype(dtype, np.integer): audio = audio * np.iinfo(dtype).max
    
    return audio.astype(dtype)

@timer
def normalize_audio(audio, max_val = 32767, dtype = np.int16, normalize_by_mean = False):
    """
        Normalize audio either to np.int16 (default) or on [-1, 1] range (max_val = 1.) with dtype np.float32
    """
    if max_val <= 1.: dtype = np.float32
    
    audio = audio - np.mean(audio)
    
    if normalize_by_mean:
        max_audio_val = np.mean(np.sort(np.abs(audio))[-len(audio) // 100:])
    else:
        max_audio_val = np.max(np.abs(audio))
    if max_audio_val <= 1e-9: return audio.astype(dtype)
    
    normalized = ((audio / max_audio_val) * max_val).astype(dtype)
    return np.clip(normalized, -max_val, max_val)

def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters = 30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)

    signal = stft_fn.inverse(magnitudes, angles)[:, 0]

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C = 1, clip_val = 1e-5):
    """
        Arguments :
            - x : `Tensor`, the audio to compress
            - C : compression factor
        Return :
            - compressed    : `Tensor` with same shape as `x`
    """
    return ops.log(ops.maximum(x, clip_val) * C)


def dynamic_range_decompression(x, C = 1):
    """
        Arguments :
            - x : `Tensor`, the audio to decompress
            - C : compression factor
        Return :
            - decompressed  : `Tensor` with same shape as `x`
    """
    return ops.exp(x) / C

