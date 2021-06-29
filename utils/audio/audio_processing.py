import numpy as np
import tensorflow as tf
import noisereduce as nr
import librosa.util as librosa_util

from scipy.signal import get_window
from scipy.ndimage.morphology import binary_dilation

max_int16 = (2 ** 15) - 1

def trim_silence(audio, method = 'window', ** kwargs):
    """
        Reduce silence at the beginning or inside theaudio depending on the method 
        Arguments : 
            - audio : the audio to process
            - method    : the trim method to use
            - kwargs    : kwargs to passto the trim function
        Return :
            - trimmed_audio
        
        Method available :
        - window    : trim silence start / end by using a convolutional window in order to smooth the audios and reduce impactof 'tic' noise at thebeginning / end. 
        - remove    : use the same concept as the window version but allowto remove silence longer than a maximum 'max_silence' time
        - vad       : use the voice activity detection but not working well
        - default   : will use the 'simple' method which trims silence at start / end just by thresholding
    """
    if method == 'window':
        return trim_silence_window(audio, ** kwargs)
    elif method == 'mel':
        return trim_silence_mel(audio, ** kwargs)
    elif method == 'remove':
        return remove_silence(audio, ** kwargs)
    elif method == 'vad':
        return trim_silence_vad(audio, ** kwargs)
    else:
        return trim_silence_simple(audio, ** kwargs)

def remove_silence(audio, rate = 22050, threshold = 0.025, max_silence = 0.15, 
                   ** kwargs):
    """
        Create a mean-window, compute the convolution of it and remove all zones that are lower than the threshold. 
    """
    window_length = int(max_silence * rate)

    mask = np.ones((window_length,)) / (window_length * threshold)

    conv = np.convolve(np.square(audio), mask, mode = 'same')
    
    return audio[conv > min(threshold, np.mean(conv) / 2)]
    
def trim_silence_window(audio, rate = 22050, mode = 'start_end', threshold = 0.1,
                        window_type = 'triangular', window_length = 0.2, 
                        debug = False, plot_kwargs = {}, ** kwargs):
    assert window_type in ('mean', 'linear', 'triangular')
    assert mode in ('start', 'end', 'start_end')
    if isinstance(window_length, float): window_length = int(window_length * rate)
    
    if window_type == 'mean':
        mask = np.ones((window_length,)) / window_length
    elif window_type == 'linear':
        mask = np.arange(window_length) / window_length
    elif window_type == 'triangular':
        mask = np.concatenate([
            np.linspace(0, 1, window_length // 2),
            np.linspace(1, 0, window_length // 2)
        ]) / (window_length // 2)

    conv = np.convolve(np.square(audio), mask, mode = 'valid')

    trimmed = audio
    if 'end' in mode:
        th_end = min(threshold, max(np.mean(conv[-window_length:]) * 5, threshold / 25))
        idx_end = np.where(conv > th_end)[0]
        if len(idx_end) > 0:
            trimmed = trimmed[:idx_end[-1] + int(window_length * 1.5)]
    
    if 'start' in mode:
        th_start = min(threshold, max(np.mean(conv[: window_length]) * 5, threshold / 25))
        idx_start = np.where(conv > th_start)[0]
        if len(idx_start) > 0:
            trimmed = trimmed[idx_start[0] :]
    
    if debug:
        from utils.plot_utils import plot_multiple
        plot_multiple(
            squared_audio = np.square(audio),
            convolved = conv, ylim = (0, 0.1),
            vlines = (idx_start[0], idx_end[-1]),
            use_subplots = True, color = 'red', ** plot_kwargs
        )
    
    return trimmed if len(trimmed) > len(audio) // 4 else audio

def trim_silence_simple(audio, threshold = 0.1, mode = 'start_end', ** kwargs):
    assert mode in ('start', 'end', 'start_end')
    
    indexes = np.where(np.abs(audio - np.mean(audio)) > threshold)[0]
    if len(indexes) > 0:
        if 'end' in mode: audio = audio[ : indexes[-1]]
        if 'start' in mode: audio = audio[indexes[0] : ]
            
    return audio

def trim_silence_mel(mel, mode = 'start_end', min_factor = 0.5, ** kwargs):
    max_amp = tf.reduce_max(mel)
    min_amp = tf.reduce_min(mel)
    
    min_val = min_amp + (max_amp - min_amp) * min_factor
    
    if mode == 'mean':
        mean_frames_amp = tf.reduce_mean(mel, axis = -1)
        return tf.boolean_mask(mel, mean_frames_amp >= min_val)
    
    elif mode == 'max':
        max_frames_amp = tf.reduce_max(mel, axis = -1)
        return tf.boolean_mask(mel, max_frames_amp >= min_val)
    
    else:
        if 'max' in mode:
            frames_amp = tf.reduce_max(mel, axis = -1)
        else:
            frames_amp = tf.reduce_mean(mel, axis = -1)
                    
        start, stop = 0, tf.shape(mel)[0] - 1
        if 'start' in mode:
            while frames_amp[start] < min_val: start = start + 1
        if 'end' in mode:
            while frames_amp[stop] < min_val: stop = stop - 1
                
        return mel[start : stop]
    
def reduce_noise(audio, noise_length = 0.2, rate = 22050, noise = None, ** kwargs):
    """
        Use the noisereduce.reduce_noise method in order to reduce audio noise
        It takes as 'noise sample' the 'noise_length' first seconds of the audio
    """
    if isinstance(noise_length, float): noise_length = int(noise_length * rate)
    
    if noise is None: noise = audio[: noise_length]
    return nr.reduce_noise(audio_clip = audio, noise_clip = noise)
    
def tf_normalize_audio(audio, max_val = 1., dtype = tf.float32):
    return tf.cast((audio / tf.reduce_max(tf.abs(audio))) * max_val, dtype)

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


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)

    signal = tf.squeeze(stft_fn.inverse(magnitudes, angles), 1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return tf.math.log(tf.clip_by_value(
        x, 
        clip_value_min = clip_val, 
        clip_value_max = tf.reduce_max(x)
    ) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return tf.exp(x) / C
