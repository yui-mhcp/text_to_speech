
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
from utils import time_to_string, load_json, dump_json, pad_batch
from utils.audio import display_audio, load_audio, write_audio
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
    
    @timer(name = 'full inference')
    def predict(self,
                sentences,
                directory       = None,
                silence_time    = 0.15,
                save_mel        = True,
                
                batch_size  = 16,
                vocoder_batch_size  = 1,
                
                overwrite   = False,
                display = False,
                play    = False,
                debug   = False, 
                ** kwargs
               ):
        """
            Perform Text-To-Speech synthesis on `sentences` and return result as a list of information
            
            Arguments : 
                - sentences     : sentences to read (str or list of str)
                - directory     : where to save result (if None, don't save result and return raw audio / mels)
                - silence_time  : silence between each part of sentence (after splitting)
                
                - batch_size    : synthesizer batch_size
                - vocoder_batch_size    : vocoder batch_size
                
                - overwrite : whether to use already generated audio (if any) or regenerate it
                
                - display   : whether to display the result for each sentence
                - play      : whether to auto-play the result (put to `False` if multiple sentences)
                - debug     : whether to show the time to generate
                
                - kwargs    : kwargs passed to both `synthesizer.predict()` and `vocoder_inference()`
            
            Return :
                list of tuple (text, infos) where infos is a dict
                    - splitted      : the splitted original text
                    - mel_files     : mel spectrogram files for each splitted part
                    - audio_files   : raw audio (if directory is None) or filename of the full audio
        """
        t0 = time.time()
        
        result = vocoder_inference(
            sentences,
            vocoder     = self.vocoder,
            synthesizer = self.synthesizer,
            rate    = self.rate,
            batch_size  = batch_size,
            vocoder_batch_size  = vocoder_batch_size,
            overwrite   = overwrite,
            directory   = directory,
            silence_time    = silence_time,
            display     = display or directory is None,
            play    = play,
            save_mel    = directory is not None and save_mel,
            ** kwargs
        )
        total_time = time.time() - t0
        
        generated_time = sum([i['duree'] for _, i in result])
        logging.info("{} generated in {} ({} generated / sec)".format(
            time_to_string(generated_time),
            time_to_string(total_time),
            time_to_string(generated_time / total_time)
        ))

        return result
    
    def stream(self, directory = None, play = True, tqdm = lambda x: x, ** kwargs):
        """
            Run a streaming TTS procedure where you can enter text and the model will read it !
        """
        predict_kwargs = {
            'directory' : directory, 'display' : True, 'play' : play,
            'tqdm' : tqdm, ** kwargs
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
                      batch_size   = 8,
                      vocoder_batch_size    = 1,
                      
                      ext       = 'mp3',
                      concat        = True,
                      overwrite     = False,
                      directory     = None,
                      save_mel      = True,
                      filename      = 'audio_{}.{}',
                      concat_filename  = 'concat_audios_{}.{}',
                       
                      silence_time = 0.15,
                       
                      display  = False,
                      play     = False,
                      tqdm    = lambda x: x,
                      ** kwargs
                     ):
    """
        Compute waveglow inference on a list of Synthesizer predictions
        
        Arguments :
            - sentences : list of text to read
            - vocoder   : vocoder model to use to transform synthesized mel spectrograms to audio
            - synthesizer   : synthesizer to use to transform text to mel spectrogram
            - rate      : audio rate (typically 22050 for NVIDIA's pretrained model)
            - batch_size    : batch_size to use for prediction (synthesizer)
            - vocoder_batch_size    : batch_size to use for prediction (vocoder)
            
            - ext   : extension for audio file (default mp3)
            - concat    : whether to concat all predictions in a big audio
            - overwrite : whether to regenerate audio if it exists (or reuse it)
            - directory : where to save data (if not provided, do not save)
            - filename  : filename format for audio file for single text
            - concat_filename   : filename format of the full concatenation
            
            - silence_time  : the time of silence between each text part (of a single sentence)
            
            - display / player  : whether to display / play the resulting audio
            - tqdm  : progress bar
            - kwargs    : unused kwargs
            
        Return : a list of tuples [(text, infos), ...]
            Where 'infos' is the 'infos' from tacotron_predictions + 'audio' and 'duree' entries
        
        Note : if `concat`, it also creates a pair for the full result in the result
    """
    def maybe_load_mel(mel):
        return np.load(mel) if isinstance(mel, str) else mel
    
    def maybe_save_silence():
        silence_filename = silence
        if audio_dir is not None:
            silence_filename = os.path.join(audio_dir, filename.format('silence', ext))
            if not os.path.exists(silence_filename):
                write_audio(silence, silence_filename, rate = rate)
        
        return {'audio' : silence_filename, 'duree' : silence_time}
    
    time_logger.start_timer('initialization')
    
    if not isinstance(sentences, (list, tuple)): sentences = [sentences]
    
    if display: tqdm = lambda x: x
    if len(sentences) > 1: play = False
    # define files / dirs to save results
    audio_dir, map_file = None, None
    if directory is not None:
        audio_dir   = os.path.join(directory, 'audios')
        map_file    = os.path.join(directory, 'map.json')
        
        os.makedirs(audio_dir, exist_ok = True)
    # Load already generated information (if any)
    infos_pred = {}
    if directory is not None and os.path.exists(map_file):
        infos_pred = load_json(map_file)
    
    to_synthesize = sentences if overwrite else [
        s for s in sentences if not infos_pred.get(s, {}).get('audio', None)
    ]

    time_logger.stop_timer('initialization')

    synthesized = synthesizer.predict(
        to_synthesize,
        batch_size  = batch_size,
        directory   = directory,
        overwrite   = overwrite,
        save        = directory is not None and save_mel,
        ** kwargs
    )
    
    time_logger.start_timer('processing')

    for (txt, infos) in synthesized:
        infos_pred.setdefault(txt, {})
        infos_pred[txt].update(infos)
    
    # Define silence waveform
    silence = np.zeros((int(rate * silence_time),))
    
    # Define variables for batch inference
    uniques = {}
    texts, mels, is_last = [], [], []
    for text in sentences:
        # Store processed text to not re-process them
        if text in uniques: continue
        elif 'mels' not in infos_pred.get(text, {}):
            infos_pred[text] = maybe_save_silence()
            continue
        
        uniques[text] = True
        
        # Skip this text if audio already exists
        infos = infos_pred[text]
        if 'audio' in infos and not overwrite: continue
        # Add information to process
        texts   += [text] * len(infos['mels'])
        mels    += infos['mels']
        is_last += [0] * (len(infos['mels']) - 1) + [1]
    
    time_logger.stop_timer('processing')

    audio_parts = {}
    for start in tqdm(range(0, len(mels), vocoder_batch_size)):
        time_logger.start_timer('batching')
        # Load all mels as np.ndarray
        batch = [maybe_load_mel(m) for m in mels[start : start + vocoder_batch_size]]
        # Pad batch for inference
        batch = pad_batch(batch, pad_value = pad_value)
        
        time_logger.stop_timer('batching')

        # Perform vocoder inference
        if batch.shape[1] > 0:
            audios = vocoder.infer(batch)
        else:
            audios = [silence] * len(batch)

        # Process each audio individually
        for i in range(len(audios)):
            idx = start + i
            audio, text, mel, last_part = audios[i], texts[idx], mels[idx], is_last[idx]
            # Truncate audio if needed
            if vocoder_batch_size > 1: audio = audio[:len(mel) * 1024]
            # Add audio + silence to the corresponding text
            audio_parts.setdefault(text, [])
            audio_parts[text] += [audio, silence]
            
            if last_part:
                # Produce full audio as concatenation of different parts
                audio = np.concatenate(audio_parts[text])
                audio_parts[text] = audio
                
                audio_filename = audio
                # Save the generated audio (if required)
                if audio_dir is not None:
                    time_logger.start_timer('saving')

                    num_pred = len(os.listdir(audio_dir))
                    # This trick allows to really overwrite the audio if it already exists
                    audio_filename = infos_pred[text].get(
                        'audio', 
                        os.path.join(audio_dir, filename.format(num_pred, ext))
                    )
                    if not save_mel:
                        infos_pred[text].pop('mels', None)
                    # Save audio to filename
                    write_audio(audio, audio_filename, rate)
                    
                    time_logger.stop_timer('saving')
                
                # Add information about audio on output information
                infos_pred[text].update({
                    'audio' : audio_filename,
                    'duree' : len(audio) / rate
                })

    if display:
        time_logger.start_timer('display')
        
        for text in sentences:
            print("Text : {}\n".format(text))
            audio = audio_parts.get(text, infos_pred[text]['audio'])
            display_audio(audio, rate = rate, play = play)
        
        time_logger.stop_timer('display')
    
    # Concat all sentences as a big audio (if required)
    if len(sentences) > 1 and concat and (audio_dir is not None or display):
        # Compute full text / audio by concatenating each sentence 
        concat_text   = '\n'.join(sentences)
        concat_audio  = np.concatenate([
            audio_parts.get(text, load_audio(infos_pred[text]['audio'], rate))
            for text in sentences
        ])
        # Save full audio (if required)
        audio_filename = concat_audio
        if audio_dir is not None:
            time_logger.start_timer('saving')

            num_pred = len([f for f in os.listdir(audio_dir) if f.startswith('concat_')])
            audio_filename = infos_pred.get(concat_text, {}).get(
                'audio',
                os.path.join(audio_dir, concat_filename.format(num_pred, ext))
            )
            write_audio(concat_audio, audio_filename, rate)
            
            time_logger.stop_timer('saving')
        
        infos_pred[concat_text] = {
            'audio' : audio_filename,
            'duree' : len(concat_audio) / rate
        }
        
        if display:
            time_logger.start_timer('display')

            print("Concatenated text : {}\n".format(concat_text))
            display_audio(concat_audio, rate = rate)
            
            time_logger.stop_timer('display')
    
    if map_file is not None:
        time_logger.start_timer('saving json')
        dump_json(map_file, infos_pred, indent = 4)
        time_logger.stop_timer('saving json')
    
    return [(text, infos_pred.get(text, maybe_save_silence())) for text in sentences]

