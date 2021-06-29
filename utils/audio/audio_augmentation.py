import random
import librosa
import numpy as np
import tensorflow as tf

from utils.audio.audio_io import load_audio

def silence(duration, rate = 22050):
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
        return random.random() * (overlap[1] - overlap[0]) + overlap[0]
    
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
    length = tf.shape(audio)[0]
    
    if min_length: max_shift = length - min_length - min_shift
    elif max_shift < 1.: max_shift = tf.cast(max_shift * length)
    
    if max_shift > 0:
        shift = tf.random.uniform(
            (), minval = min_shift, 
            maxval = max_shift,
            dtype = tf.int32
        )
        
        audio = audio[shift :]
    
    return audio

def random_pad(audio, max_length):
    maxval = max_length - tf.shape(audio)[0]
    if maxval > 0:
        padding_left = tf.random.uniform(
            (), minval = 0, 
            maxval = maxval,
            dtype = tf.int32
        )
            
        if maxval - padding_left > 0:
            padding_right = tf.random.uniform(
                (), minval = 0, 
                maxval = maxval - padding_left,
                dtype = tf.int32
            )
        else:
            padding_right = 0
        
        if len(tf.shape(audio)) == 2:
            padding = [(padding_left, padding_right), (0, 0)]
        else:
            padding = [(padding_left, padding_right)]
            
            audio = tf.pad(audio, padding)
            
    return audio

def random_noise(audio, intensite = -1, max_intensite = 0.2):
    if intensite < 0:
        intensite = tf.random.uniform(
            (), minval = 0,
            maxval = max_intensite,
            dtype = tf.float32
        )
    
    if intensite == 0: return audio
    
    noise = tf.random.uniform(
        tf.shape(audio), 
        minval = tf.reduce_min(audio),
        maxval = tf.reduce_max(audio),
        dtype = audio.dtype
    )

    return audio * (1. - intensite) + noise * intensite