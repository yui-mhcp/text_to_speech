import os
import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from utils import time_to_string, load_json, dump_json, pad_batch
from utils.audio import display_audio, load_audio, write_audio
from models.tts.tacotron2 import Tacotron2
from models.tts.waveglow import PtWaveGlow

_stream_msg = {
    'fr'    : 'Entrez le texte à lire :',
    'en'    : 'Enter text to read :'
}
_end_msg = {
    'fr'    : 'Au revoir, à la prochaine !',
    'en'    : 'Goodbye, see you soon !'
}

_default_synthesizer    = Tacotron2
_default_vocoder        = PtWaveGlow

class Vocoder(object):
    def __init__(self):
        self.__synthesizer  = None
        self.__vocoder      = None
    
    def set_synthesizer(self, nom, model_class = None):
        if model_class is None:
            model_class = self.synthesizer_class
        self.__synthesizer = model_class(nom = nom)
    
    def set_vocoder(self, nom = None, model_class = None):
        if model_class is None:
            model_class = self.vocoder_class
        self.__vocoder = model_class(nom = nom)
    
    @property
    def synthesizer(self):
        return self.__synthesizer
    
    @property
    def vocoder(self):
        return self.__vocoder
    
    @property
    def synthesizer_class(self):
        return type(self.__synthesizer) if self.__synthesizer is not None else _default_synthesizer
    
    @property
    def vocoder_class(self):
        return type(self.__vocoder) if self.__vocoder is not None else _default_vocoder
    
    @property
    def lang(self):
        return self.synthesizer.lang
    
    @property
    def rate(self):
        return self.synthesizer.audio_rate
    
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
        start = time.time()
        
        mels_pred = self.synthesizer.predict(
            sentences,
            batch_size  = batch_size,
            directory   = directory,
            overwrite   = overwrite,
            save        = directory is not None and save_mel,
            debug       = debug,
            ** kwargs
        )
        
        result = vocoder_inference(
            self.vocoder,
            mels_pred,
            rate    = self.rate,
            batch_size  = vocoder_batch_size,
            overwrite   = overwrite,
            directory   = directory,
            silence_time    = silence_time,
            display     = display or directory is None,
            play    = play,
            ** kwargs
        )
        total_time = time.time() - start
        
        if debug:
            generated_time = sum([i['duree'] for _, i in result])
            print("{} generated in {} ({} generated / sec)".format(
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
    
def vocoder_inference(vocoder,
                      mels_predictions,
                      pad_value    = -11.,
                      rate     = 22050,
                      batch_size   = 1,
                       
                      ext      = 'mp3',
                      concat       = True,
                      overwrite    = False,
                      directory    = None,
                      filename      = 'audio_{}.{}',
                      concat_filename  = 'concat_audios_{}.{}',
                       
                      silence_time = 0.15,
                       
                      display  = False,
                      play     = False,
                      tqdm    = tqdm,
                      ** kwargs
                     ):
    """
        Compute waveglow inference on a list of Synthesizer predictions
        
        Arguments :
            - vocoder  : vocoder model to use to transform synthesized mel spectrograms to audio
            - mels_predictions  : list of tuple [(text, infos), ...]
                Where infos is a dict with at least key 'mels' or 'mel_filenames'
            - rate      : audio rate (typically 22050 for NVIDIA's pretrained model)
            - batch_size    : batch_size to use for prediction (default 1)
            
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
    
    if display: tqdm = lambda x: x
    if len(mels_predictions) > 1: play = False
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
    
    # Define variables for batch inference
    uniques = {}
    texts, mels, is_last = [], [], []
    for text, infos in mels_predictions:
        # Store processed text to not re-process them
        if text in uniques: continue
        uniques[text] = True
        
        # Skip this text if audio already exists
        infos_pred.setdefault(text, {})
        if 'audio' in infos_pred[text] and not overwrite: continue
        # Add information to process
        texts   += [text] * len(infos['mels'])
        mels    += infos['mels']
        is_last += [0] * (len(infos['mels']) - 1) + [1]
    
    # Define silence waveform
    silence = np.zeros((int(rate * silence_time),))
    
    audio_parts = {}
    for start in tqdm(range(0, len(mels), batch_size)):
        # Load all mels as np.ndarray
        batch = [maybe_load_mel(m) for m in mels[start : start + batch_size]]
        # Pad batch for inference
        batch = pad_batch(batch, pad_value = pad_value)
        # Perform vocoder inference
        audios = vocoder.infer(batch)
        
        # Process each audio individually
        for i in range(len(audios)):
            idx = start + i
            audio, text, mel, last_part = audios[i], texts[idx], mels[idx], is_last[idx]
            # Truncate audio if needed
            if batch_size > 1: audio = audio[:len(mel) * 1024]
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
                    num_pred = len(os.listdir(audio_dir))
                    # This trick allows to really overwrite the audio if it already exists
                    audio_filename = infos_pred[text].get(
                        'audio', 
                        os.path.join(audio_dir, filename.format(num_pred, ext))
                    )
                    # Save audio to filename
                    write_audio(audio, audio_filename, rate)
                
                # Add information about audio on output information
                infos_pred[text].update({
                    'audio' : audio_filename,
                    'duree' : len(audio) / rate
                })
                # Display audio (with text) if required
                if display:
                    print("Text : {}\n".format(text))
                    display_audio(audio, rate = rate, play = play)
    
    # Concat all sentences as a big audio (if required)
    if len(mels_predictions) > 1 and concat and (audio_dir is not None or display):
        # Compute full text / audio by concatenating each sentence 
        concat_text   = '\n'.join([text for text, _ in mels_predictions])
        concat_audio  = np.concatenate([
            audio_parts.get(text, load_audio(infos_pred[text]['audio'], rate))
            for text, _ in mels_predictions
        ])
        # Save full audio (if required)
        audio_filename = concat_audio
        if audio_dir is not None:
            num_pred = len([f for f in os.listdir(audio_dir) if f.startswith('concat_')])
            audio_filename = infos_pred.get(concat_text, {}).get(
                'audio',
                os.path.join(audio_dir, concat_filename.format(num_pred, ext))
            )
            write_audio(concat_audio, audio_filename, rate)
        
        infos_pred[concat_text] = {
            'audio' : audio_filename,
            'duree' : len(concat_audio) / rate
        }
        
        if display:
            print("Concatenated text : {}\n".format(concat_text))
            display_audio(concat_audio, rate = rate)
    
    if map_file is not None:
        dump_json(map_file, infos_pred, indent = 4)
    
    return [(text, infos_pred[text]) for text, _ in mels_predictions]

