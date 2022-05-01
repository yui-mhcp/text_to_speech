
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
import time
import shutil
import librosa
import logging
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from pydub import AudioSegment
from scipy.signal import resample
from scipy.io.wavfile import write, read

from utils.audio import audio_processing
from utils.thread_utils import ThreadedQueue

MAX_DISPLAY_TIME = 600

_audio_player   = None

""" Generic functions to load audio and mel """

def load_audio(data, rate, ** kwargs):
    """
        Load audio from different type of data :
            - str : filename of the audio file
            - np.ndarray / tf.Tensor    : raw audio
            - dict / pd.Series : 
                'audio' : raw audio
                'wavs_<rate>'   : filename for audio of correct rate
                'filename'      : filename for audio (resample if needed)
        Return : audio 
    """
    if isinstance(data, str):
        _, audio = read_audio(data, target_rate = rate, ** kwargs)
    elif isinstance(data, (np.ndarray, tf.Tensor)):
        audio = data
    elif isinstance(data, (dict, pd.Series)):
        if 'audio' in data:
            audio = data['audio']
        else:
            audio_key = 'wavs_{}'.format(rate)
            if audio_key not in data: 
                audio_key = 'filename' if 'filename' in data else 'audio_filename'
            audio = tf_read_audio(data[audio_key], target_rate = rate, ** kwargs)
    else:
        raise ValueError("Unknown audio type : {}\n{}".format(type(data), data))
    
    return audio

def load_mel(data, stft_fn, trim_mode = None, ** kwargs):
    """
        Load mel from different type of data :
            - dict / pd.Series  : 
                'mel'   : raw mel
                stft.dir_name   : filename of mel
            - other : call load_audio(data) and apply stft_fn on audio
        Return : mel spectrogram (as 2D tf.Tensor)
    """
    if isinstance(data, (dict, pd.Series)) and 'mel' in data:
        mel = data['mel']
    elif isinstance(data, (dict, pd.Series)) and stft_fn.dir_name in data:
        def load_mel(filename):
            return np.load(filename.numpy().decode('utf-8'))

        mel = tf.py_function(load_mel, [data[stft_fn.dir_name]], Tout = tf.float32)
        mel.set_shape([None, stft_fn.n_mel_channels])
        return mel
    else:
        audio   = load_audio(data, stft_fn.rate, ** kwargs)
        mel     = stft_fn(audio)
    
    if len(mel.shape) == 3: mel = tf.squeeze(mel, 0)
    
    if trim_mode is not None:
        kwargs.update({'method' : trim_mode, 'rate' : stft_fn.rate})
        mel = audio_processing.trim_silence(mel, ** kwargs)
    
    return mel

""" 
    Méthodes pour jouer / lire de l'audio 
    Les méthodes read_<format>(filename) renvoie un tuple : 
    - (rate, data)
"""

def resample_file(filename, new_rate, filename_out = None):
    if filename_out is None:
        base_name, ext = filename.split('.')[0], filename.split('.')[-1]
        filename_out = '{}_{}{}'.format(base_name, new_rate, ext)
    if os.path.exists(filename_out): return filename_out
    try:
        rate, audio = read_audio(filename, target_rate = new_rate)
    except ValueError as e:
        logging.error("Error while loading file {} !\n{}".format(filename, e))
        return None
    write_audio(audio, filename_out, rate, verbose = False)
    return filename_out

def play_audio(filename, block = True):
    if block:
        return subprocess.run(['ffplay', '-nodisp', '-autoexit', filename])
    else:
        global _audio_player
        if _audio_player is None:
            _audio_player = ThreadedQueue(play_audio, max_workers = 0)
            _audio_player.start()
        
        _audio_player.append(filename = filename, block = True)

def display_audio(filename, rate = None, play = False,
                  debut = None, fin = None, temps = None, ** kwargs):
    from IPython.display import Audio, display
    if isinstance(filename, str):
        rate, audio = read_audio(filename, target_rate = rate, ** kwargs)
    else:
        assert rate is not None
        audio = filename
    
    if temps:
        if debut: fin = debut + temps
        elif fin: debut = max(0, fin - temps)
        else: fin = temps

    if debut is not None: debut = int(rate * debut)
    if fin is not None: fin = int(rate * fin)

    if debut or fin: audio = audio[debut : fin]
    
    display(Audio(audio[:int(MAX_DISPLAY_TIME * rate)], rate = rate, autoplay = play))
    return rate, audio

def tf_read_audio(filename, target_rate = None, ** kwargs):
    def decode_and_read(filename):
        filename = filename.numpy().decode('utf-8')
        return read_audio(filename, target_rate = target_rate, normalize = True, ** kwargs)
    _, audio = tf.py_function(decode_and_read, [filename], 
                              Tout = [tf.int32, tf.float32])
    audio.set_shape([None])
    return audio

def read_audio(filename, target_rate = None, normalize = True, 
               offset = 0, reduce_noise = False, trim_silence = False,
               debut = 0, fin = 0, temps = None, ** kwargs):
    ext = filename.split('.')[-1]
    if ext not in _supported_audio_formats:
        raise ValueError("Unhandled extension ! \n  Accepted : {} \n  Got : {}".format(
            list(_supported_audio_formats.keys()), ext
        ))
    
    if 'read' not in _supported_audio_formats[ext]:
        raise ValueError("Read mot not handled for extension {}".format(ext))
    
    rate, audio = _supported_audio_formats[ext]['read'](filename)
    
    if len(audio) == 0:
        logging.warning("Audio {} is empty !".format(filename))
        return np.ones((rate,), dtype = np.float32)
    
    if target_rate is not None and target_rate != rate:
        ratio = target_rate / rate
        audio = resample(audio, int(len(audio) * ratio))
        rate = target_rate
    
    if offset > 0:
        if isinstance(offset, float): offset = int(offset * rate)
        audio = audio[offset : - offset]
    
    if normalize:
        audio = audio_processing.normalize_audio(audio, max_val = 1.)
    
    if reduce_noise:
        audio = audio_processing.reduce_noise(audio, rate = rate, ** kwargs)
        audio = audio_processing.normalize_audio(audio, max_val = 1.)
    
    if trim_silence:
        audio = audio_processing.trim_silence(audio, rate = rate, ** kwargs)
        
    if temps:
        if debut: fin = debut + temps
        if fin: debut = max(0, fin - temps)
    
    if fin: audio = audio[ : int(fin * rate)]
    if debut: audio = audio[int(debut * rate) : ]
    
    return rate, audio

def read_wav(filename):
    return read(filename)

def read_pydub(filename):
    audio = AudioSegment.from_mp3(filename)
    audio_np = np.array(audio.get_array_of_samples())
    if audio.channels > 1: audio_np = audio_np[::audio.channels]
    return audio.frame_rate, audio_np

def read_librosa(filename):
    audio, rate = librosa.load(filename, sr = None)
    return rate, audio

def read_video_audio(filename):
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        logging.error("You must install moviepy : pip install moviepy")
        return None

    with VideoFileClip(filename) as video:
        audio = video.audio

        fps     = audio.fps
        array   = audio.to_soundarray()
    
    if len(array.shape) > 1: array = array[:,0]
    
    return fps, array

""" 
    Méthodes pour sauver de l'audio 
    Signature classique : 
    write_<format>(audio, filename, rate)
"""

def write_audio(audio, filename, rate=22050, normalize = True, 
                factor = 32767, verbose = False):
    ext = filename.split('.')[-1]
    if ext not in _supported_audio_formats:
        raise ValueError("Extension non gérée ! \nAccepte : {} \nReçu :{}".format(list(_supported_audio_formats.keys()), ext))
        
    if 'write' not in _supported_audio_formats[ext]:
        raise ValueError("Writing mode is not available for format : {}".format(ext))
        
    logging.log(logging.INFO if verbose else logging.DEBUG, "Saving audio to {}".format(filename))
    
    normalized = audio
    if normalize:
        normalized = audio_processing.normalize_audio(
            audio, max_val = factor, normalize_by_mean = False
        )
    
    _supported_audio_formats[ext]['write'](normalized, filename, rate)
    
def write_wav(audio_np, filename, rate):
    write(filename, rate, audio_np)
    
def write_pydub(audio_np, filename, rate):
    audio = AudioSegment(
        audio_np.tobytes(), frame_rate = rate, sample_width = audio_np.dtype.itemsize, channels = 1
    )
    file = audio.export(filename, format = filename.split('.')[-1])
    file.close()
    
""" Processing functions to read / write audios depending of the extension """

_video_ext  = ('mp4', 'mov', 'ovg', 'avi')

_supported_audio_formats = {
    ** {ext : {'read' : read_video_audio} for ext in _video_ext},
    'wav'   : {'read' : read_wav, 'write' : write_wav},
    'mp3'   : {'read' : read_pydub, 'write' : write_pydub},
    'm4a'   : {'read' : read_pydub, 'write' : write_pydub},
    'flac'  : {'read' : read_librosa},
    'opus'  : {'read' : read_librosa}
}
