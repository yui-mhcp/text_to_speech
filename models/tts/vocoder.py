
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
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from utils.thread_utils import Consumer
from utils import time_to_string, load_json, dump_json, pad_batch
from utils.audio import display_audio, play_audio, load_audio, write_audio
from models.interfaces import BaseModel
from models.tts.waveglow import PtWaveGlow

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
_vocoder_consumer   = None

class Vocoder(object):
    def __init__(self):
        self.__synthesizer  = None
        self.__vocoder      = None
    
    def set_synthesizer(self, nom):
        if isinstance(nom, BaseModel):
            self.__synthesizer = nom
        elif isinstance(nom, str):
            from models import get_pretrained

            self.__synthesizer = get_pretrained(nom)
    
    def set_vocoder(self, nom = None, model_class = None):
        if  nom is None:
            self.__vocoder = PTWaveGlow()
        elif isinstance(nom, str):
            from models import get_pretrained

            self.__vocoder = get_pretrained(nom)
        elif isinstance(nom, BaseModel):
            self.__vocoder = nom
        else:
            raise ValueError("Unknown vocoder type : {}\n  {}".format(type(nom), nom))
    
    @property
    def synthesizer(self):
        return self.__synthesizer
    
    @property
    def vocoder(self):
        return self.__vocoder
    
    @property
    def synthesizer_class(self):
        return type(self.__synthesizer) if self.__synthesizer is not None else None
    
    @property
    def vocoder_class(self):
        return type(self.__vocoder) if self.__vocoder is not None else None
    
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
            sentences,
            vocoder     = self.vocoder,
            synthesizer = self.synthesizer,
            rate    = self.rate,
            ** kwargs
        )
        total_time = time.time() - t0
        if total_time <= 1e-9: total_time = 1e-6
        
        generated_time = sum([i.get('duree', -1) for _, i in result])
        logging.info("{} generated in {} ({} generated / sec)".format(
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

@timer
def vocoder_inference(sentences,
                      vocoder,
                      synthesizer,
                      pad_value    = -11.,
                      rate     = 22050,
                      vocoder_batch_size    = 1,
                      
                      persistent    = True,
                      
                      ext       = 'mp3',
                      overwrite = False,
                      directory     = None,
                      filename      = 'audio_{}.{}',
                      concat        = True,
                      concat_filename  = 'concat_audios_{}.{}',
                       
                      silence_time = 0.15,
                       
                      blocking  = True,
                      display   = False,
                      play      = False,
                      ** kwargs
                     ):
    """
        Compute waveglow inference on a (list of) Synthesizer predictions
        
        Arguments :
            - sentences : (list of) text to read
            - vocoder   : vocoder model to use to transform synthesized mel spectrograms to audio
            - synthesizer   : synthesizer to use to transform text to mel spectrogram
            - rate      : audio rate (typically 22050 for NVIDIA's pretrained model)
            - persistent    : whether to keep the created `Pipeline` or not
            
            - batch_size    : batch_size to use for prediction (synthesizer)
            - vocoder_batch_size    : batch_size to use for prediction (vocoder)
            
            - ext   : extension for audio file (default mp3)
            - concat    : whether to concat all predictions in a big audio
            - overwrite : whether to regenerate audio if it exists (or reuse it)
            - directory : where to save data (if not provided, do not save)
            - filename  : filename format for audio file for single text
            - concat_filename   : filename format of the full concatenation
            
            - silence_time  : the time of silence between each text part (of a single sentence)
            
            - blocking  : whether to wait until the end or not (if not, it can return empty / not updated `infos` !)
            - display / play    : whether to display / play the resulting audio
            - kwargs    : propagated to `synthesizer.get_streaming_pipeline` and `pipeline.append()`
        
        Return : a list of tuples [(text, infos), ...]
            Where 'infos' is the 'infos' from tacotron_predictions + 'audio' and 'duree' entries
        
        Note : if `concat`, it also creates a pair for the full result in the result
        
        /!\ This function has not been tested for `vocoder_batch_size > 1` and might produce  unexpected behavior for empty sentences and can infer duplicated sentences.
        
        Currently, this function creates "global" pipelines and re-use it if already created. It means that the 1st call to `vocoder_inference` will create the pipeline with all its configuration (`batch_size`, `directory`, ...) but subsequent calls will re-use the same pipeline : all these configurations will simply be ignored. 
        You can skip this behavior by setting `persistent = False` but be careful with this especially if you put `blocking = False` ! If you re-call this function directly after, it will probably re-create a new `Pipeline` which will be executed in parallel of the previous one, meaning that the same models might be call in multiple threads at the same time. 
        If you set `blocking = True`, the models' calls will be finished before ending this function call. It will therefore not cause any issue. 
    """
    def maybe_load_mel(mel):
        if isinstance(mel, list): return [maybe_load_mel(m) for m in mel]
        return np.load(mel) if isinstance(mel, str) else mel

    @timer
    def vocoder_infer(pred):
        data = pred if isinstance(pred, list) else [pred]

        to_update = [
            infos for sent, infos in data
            if 'mel' in infos and infos.get('timestamp', 0) > sent_mapping.get(sent, {}).get('timestamp', 0)
        ]
        mels = maybe_load_mel([infos['mel'] for infos in to_update])
        
        audios = []
        if len(mels) > 0:
            batch = np.array(mels) if len(mels) == 1 else pad_batch(mels, pad_value = pad_value)

            if batch.shape[1] > 0:
                audios = vocoder.infer(batch)
            else:
                audios = [silence] * len(batch)
        
        for infos_i, audio_i in zip(to_update, audios):
            if len(audios) > 1: audio_i = audio_i[: len(mel_i) * 1024]
            infos_i.update({'audio' : audio_i, 'duree' : len(audio_i) / rate})
        
        return pred
    
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
    
    def concatenate_audios(output):
        if output is None: return None
        text, infos, updated = output

        if updated or 'audio' not in infos:
            audios = infos.get('audios', [])
            if len(audios) == 0:
                infos['audio'] = maybe_save_silence()
                infos['duree'] = silence_time
            elif len(audios) == 1:
                infos['audio'] = infos['audios'][0] if isinstance(infos['audios'][0], str) else concat_and_save(audios, infos.get('audio', None))
                infos['duree'] = infos['durees'][0]
            else:
                infos['audio'] = concat_and_save(audios, infos.get('audio', None))
                infos['duree'] = sum(infos['durees'])
            updated = True
        
        return (text, infos, updated)
    
    global _pipelines, _vocoder_consumer
    if _pipelines.get(synthesizer.nom, None) is None:
        if _vocoder_consumer is None:
            _vocoder_consumer   = Consumer(
                vocoder_infer,
                batch_size  = vocoder_batch_size,
                max_workers = min(kwargs.get('pipeline_workers', 0), 1)
            )

        display_and_play_consumers = []
        if display: display_and_play_consumers.append(
            lambda out: display_audio(out[1]['audio'], rate = rate)
        )
        if play: display_and_play_consumers.append(
            lambda out: play_audio(out[1]['audio'], rate = rate, block = False)
        )
        _pipelines[synthesizer.nom] = synthesizer.get_streaming_pipeline(
            post_processing = _vocoder_consumer,
            post_group  = {
                'consumer'  : concatenate_audios,
                'consumers' : display_and_play_consumers,
                'name'      : 'save_audio'
            },
            required_keys   = 'audio',
            directory   = directory,
            save    = directory is not None,
            ** kwargs
        )
    
    if directory is None: display = True
    if not isinstance(sentences, (list, tuple)): sentences = [sentences]
    # define files / dirs to save results
    audio_dir = None if directory is None else os.path.join(directory, 'audios')
    if audio_dir: os.makedirs(audio_dir, exist_ok = True)
    
    # Define silence waveform
    silence = np.zeros((int(rate * silence_time),))

    sentence_splitter, grouper, pipeline, text_mapping, sent_mapping = _pipelines[synthesizer.nom]
    
    timestamps  = {
        text : text_mapping.get(text, {}).get('timestamps', None) for text in set(sentences)
    }
    timestamps  = {k : v for k, v in timestamps.items() if v is not None}
    
    for sent in sentences:
        sentence_splitter(sent, overwrite = overwrite, blocking = blocking, ** kwargs)
    
    if overwrite and blocking:
        def is_updated(text):
            last_timestamps = timestamps[text]
            new_timestamps  = text_mapping.get(text, {}).get('timestamps', last_timestamps)

            is_new = all(last_t < t for last_t, t in zip(last_timestamps, new_timestamps))

            return is_new
        
        for text in timestamps.keys():
            text_mapping.wait_for(text, lambda: is_updated(text))
    
    if not persistent:
        if not blocking:
            logging.warning('Be careful when calling this function with `blocking = False and persistent = False` !')
        _vocoder_consumer   = None
        _pipelines.pop(synthesizer.nom)
        
    if not blocking:
        return [(text, text_mapping.get(text, {})) for text in sentences]
    return [(text, text_mapping[text]) for text in sentences]

