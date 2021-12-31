import os
import time
import logging

from utils import load_json
from models.tts import get_audio_file, get_model_name, tts, WaveGlow, Tacotron2
from utils.audio import play_audio

_default_directory = os.path.join('loggers', 'tts_logs')

class TTSHandler(logging.Handler):
    def __init__(self, lang = 'en', model = None, directory = _default_directory,
                 generate_if_not_exist = False, level = 0, ** kwargs):
        super().__init__(level = level)
        self.lang = lang
        self.model = model if model is not None else get_model_name(lang)
        self.directory = os.path.join(directory, self.model)
        self.generate_if_not_exist = generate_if_not_exist
        
        self._generating = False
    
    def play(self, message):
        filename = get_audio_file(message, directory = self.directory)
        
        _is_restoring = Tacotron2._is_restoring or WaveGlow._is_restoring
        if filename is None and self.generate_if_not_exist and not _is_restoring:
            self._generating = True
            
            filename = tts(
                message, model = self.model, directory = self.directory,
                save_mel = False, display = False
            )[0][1]['audio']
            time.sleep(0.1)
            
            self._generating = False
        
        if filename is not None:
            play_audio(filename, block = False)
    
    def emit(self, record):
        if self._generating: return
        self.play(self.format(record))
