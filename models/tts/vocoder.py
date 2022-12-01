
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
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils import time_to_string, pad_batch
from utils.audio import display_audio, play_audio, load_audio, write_audio
from models.interfaces import BaseModel
from models.tts.waveglow import PtWaveGlow

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

_stream_msg = {
    'fr'    : 'Entrez le texte à lire :',
    'en'    : 'Enter text to read :'
}
_end_msg = {
    'fr'    : 'Au revoir, à la prochaine !',
    'en'    : 'Goodbye, see you soon !'
}

_pipelines  = {}

class Vocoder(object):
    def __init__(self):
        self.__synthesizer  = None
        self.__vocoder      = None
    
    def set_synthesizer(self, model):
        if isinstance(model, BaseModel):
            self.__synthesizer = model
        elif isinstance(model, str):
            from models import get_pretrained

            self.__synthesizer = get_pretrained(model)
    
    def set_vocoder(self, model = None, model_class = None):
        if  model is None:
            self.__vocoder = PTWaveGlow()
        elif isinstance(model, str):
            from models import get_pretrained

            self.__vocoder = get_pretrained(model)
        elif isinstance(model, BaseModel):
            self.__vocoder = model
        else:
            raise ValueError("Unknown vocoder type : {}\n  {}".format(type(model), model))
    
    @property
    def synthesizer(self):
        return self.__synthesizer
    
    @property
    def vocoder(self):
        return self.__vocoder
    
    @property
    def lang(self):
        return self.synthesizer.lang
    
    @property
    def rate(self):
        return self.synthesizer.audio_rate
    
    def predict(self, sentences, ** kwargs):
        """ Use `help(vocoder_inference)` for complete information """
        t0 = time.time()
        
        result = vocoder_inference(
            sentences, synthesizer = self.synthesizer, vocoder = self.vocoder, ** kwargs
        )
        total_time = max(time.time() - t0, 1e-6)
        
        generated_time = sum([i.get('time', -1) for _, i in result])
        logger.info("{} generated in {} ({} generated / sec)".format(
            time_to_string(generated_time),
            time_to_string(total_time),
            time_to_string(generated_time / total_time)
        ))

        return result
    
    def stream(self, directory = None, play = True, ** kwargs):
        """ Run a streaming TTS procedure where you can enter text and the model reads it ! """
        predict_kwargs = {
            'directory' : directory, 'display' : True, 'play' : play, ** kwargs
        }
        while True:
            text = input(_stream_msg.get(self.lang, _stream_msg['en']))
            if len(text) < 2: break
            
            self.predict(text, ** predict_kwargs)
        
        if self.lang in _end_msg:
            self.predict(_end_msg[self.lang], ** predict_kwargs)

def maybe_load_mel(mel):
    if isinstance(mel, list): return [maybe_load_mel(m) for m in mel]
    if isinstance(mel, dict): mel = mel.get('mel', None)
    return np.load(mel) if isinstance(mel, str) else mel

@timer
def vocoder_infer(infos, vocoder, rate, pad_value, ** kwargs):
    """
        Gets a (list of) mel-spectrogram (or dict with `mel` key) and returns the (list of) corresponding audio (produced by `vocoder`)
    """
    result = infos if isinstance(infos, list) else [infos]

    mels    = maybe_load_mel(result)
    should_skip = [False if m is not None else True for m in mels]
    mels    = [m for m in mels if m is not None]

    if len(mels) > 0:
        batch   = pad_batch(mels, pad_value = pad_value) if len(mels) > 1 else np.array(mels)
        audios  = vocoder.infer(batch)

    outputs, idx = [], 0
    for i, (res, skip) in enumerate(zip(result, should_skip)):
        if skip:
            outputs.append(res if isinstance(res, dict) else None)
            continue
        
        mel, audio = mels[idx], audios[idx]
        if len(mels) > 1: audio = audio[: len(mel) * 1024]
        if isinstance(res, dict):
            res.update({'audio' : audio, 'time' : len(audio) / rate})
            outputs.append(res)
        else:
            outputs.append(audio)
        idx += 1
    
    return outputs if isinstance(infos, list) else outputs[0]

@timer
def vocoder_inference(sentences,
                      vocoder,
                      synthesizer,
                      pad_value    = -11.,
                      vocoder_batch_size    = 1,
                      
                      save_parts    = None,
                      
                      persistent    = True,
                      
                      ext       = 'mp3',
                      overwrite = False,
                      directory = None,
                      filename  = 'audio_{}.{}',
                      
                      silence_time = 0.15,
                      
                      display   = False,
                      play      = False,
                      
                      blocking  = True,
                      ** kwargs
                     ):
    """
        Compute waveglow inference on a (list of) Synthesizer predictions
        
        Arguments :
            - sentences : (list of) text to read
            - vocoder   : vocoder model to use to transform synthesized mel spectrograms to audio
            - synthesizer   : synthesizer to use to transform text to mel spectrogram
            - persistent    : whether to keep the created `Pipeline` or not
            
            - vocoder_batch_size    : batch_size to use for prediction (vocoder)
            
            - ext   : extension for audio file (default mp3)
            - overwrite : whether to regenerate audio if it exists (or reuse it)
            - directory : where to save data (if not provided, do not save)
            - filename  : filename format for audio file for single text
            
            - silence_time  : the time of silence between each text part (of a single sentence)
            
            - blocking  : whether to wait until the end or not (if not, it can return empty / not updated `infos` !)
            - display / play    : whether to display / play the resulting audio
            - kwargs    : propagated to `synthesizer.get_streaming_pipeline` and `pipeline.append()`
        
        Returns : a list of tuples [(text, infos), ...]
            Where `infos` is a dict containing the information returned by the pipeline
            If the inference is performed (either overwrite or data was not already generated), all keys will be there (i.e. {mels, attn_weights, gates, audios, audio}) but if data is restored, some keys can be missing (see `help(Tacotron2.get_pipeline)` for more information)
        
        Currently, this function creates "global" pipelines and re-use it if already created. It means that the 1st call to `vocoder_inference` will create the pipeline with all its configuration (`batch_size`, `directory`, ...) but subsequent calls will re-use the same pipeline : all these configurations will simply be ignored. 
        You can skip this behavior by setting `persistent = False` but be careful with this especially if you set `blocking = False` ! If you re-call this function directly after, it will probably re-create a new `Pipeline` which will be executed in parallel of the previous one, meaning that the same models might be call in multiple threads at the same time. 
        If you set `blocking = True`, the models' calls will be finished before ending this function call. It will therefore not cause any issue. 
        
        If you call this functions with multiple times with different synthesizers, all the pipelines (1 per synthesizer) will co-exist. If `blocking = True`, it will not be an issue as the co-existing (unused) pipelines will not be active when calling this function (sleeping threads). 
        However, if `blocking = False` (or in a multi-threaded scenario), multiple pipelines may be active at the same time, meaning that multiple models can be predicting at the same time. 
        In theory it should not be an issue but I do not know how `tensorflow` handles that. 
    """
    def maybe_save_silence():
        silence_filename = silence
        if audio_dir is not None:
            silence_filename = os.path.join(audio_dir, filename.format('silence', ext))
            if not os.path.exists(silence_filename):
                write_audio(audio = silence, filename = silence_filename, rate = rate)
        
        return silence_filename
    
    def concat_and_save(audios, audio_filename):
        concat = []
        for a in audios:
            concat.extend([load_audio(a, rate = rate), silence])
        concat = np.concatenate(concat)
        
        if audio_dir is not None:
            if audio_filename is None:
                num_pred        = len(os.listdir(audio_dir))
                audio_filename  = os.path.join(audio_dir, filename.format(num_pred, ext))

            write_audio(audio = concat, filename = audio_filename, rate = rate)
        return audio_filename if audio_dir else concat
    
    def concatenate_audios(result, overwritten_data = {}, ** kwargs):
        """ Concatenates the audios and (possibly) saves it """
        audios = result.get('audios', [])
        if len(audios) == 0:
            result.update({
                'audio' : maybe_save_silence(), 'time' : silence_time
            })
        elif len(audios) == 1:
            result.update({
                'audio' : audios[0] if isinstance(audios[0], str) else concat_and_save(
                    audios, overwritten_data.get('audio', None)
                ),
                'time'  : result['times'][0]
            })
        else:
            result.update({
                'audio' : concat_and_save(audios, overwritten_data.get('audio', None)),
                'time'  : sum(result['times'])
            })
        
        return result
    
    kwargs.update({'overwrite' : overwrite})
    for k in ['save_plot', 'save_mel', 'save_audio']: kwargs.setdefault(k, save_parts)
    
    audio_dir = None if directory is None else os.path.join(directory, 'audios')
    if directory is None: display = True
    else: os.makedirs(audio_dir, exist_ok = True)
    
    rate    = synthesizer.audio_rate

    global _pipelines
    if _pipelines.get(synthesizer.nom, None) is None:
        pipeline = synthesizer.get_pipeline(
            post_processing = {
                'consumer'  : lambda out, ** kwargs: vocoder_infer(
                    out, vocoder, rate = rate, pad_value = pad_value, ** kwargs
                ),
                'name'  : 'vocoder_inference',
                'description'   : vocoder_infer.__doc__,
                'batch_size'    : vocoder_batch_size,
                'allow_multithread' : False
            },
            post_group  = {
                'name' : 'save_audio', 'consumer' : concatenate_audios
            },
            expected_keys   = 'audio',
            directory   = directory,
            save    = directory is not None,
            ** kwargs
        )
        if display:
            pipeline.add_listener(
                lambda result, ** kwargs: display_audio(result['audio'], rate = rate),
                name = 'display_audio'
            )
        if play:
            pipeline.add_listener(
                lambda result, ** kwargs: play_audio(result['audio'], rate = rate, block = False),
                name = 'play_audio'
            )
        
        _pipelines[synthesizer.nom] = pipeline
    
    if not isinstance(sentences, (list, tuple, pd.DataFrame)): sentences = [sentences]
    
    # Define silence waveform
    silence = np.zeros((int(rate * silence_time),))

    pipeline    = _pipelines[synthesizer.nom]
    if not persistent: _pipelines.pop(synthesizer.nom)
    
    if not blocking:
        pipeline.extend(sentences, ** kwargs)
        if not persistent: pipeline.stop_when_empty()
        
        if overwrite:
            return [(sent, {}) for sent in sentences]
        
        return [(sent, {}) for sent in sentences]
        #with pipeline.mutex_db:
        #    return [(sent, pipeline._get_from_database(sent, {})) for sent in sentences]
    
    return [
        (sent, result) for sent, result in zip(
            sentences, pipeline.extend_and_wait(sentences, stop = not persistent, ** kwargs)
        )
    ]

