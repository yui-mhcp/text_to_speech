
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

from utils import load_json
from models.tts import get_audio_file, get_model_name, tts, WaveGlow, Tacotron2
from utils.audio import play_audio

_default_directory = os.path.join('loggers', 'tts_logs')

class TTSHandler(logging.Handler):
    def __init__(self, lang = 'en', model = None, directory = _default_directory,
                 generate_if_not_exist = False, level = 0, ** kwargs):
        super().__init__(level = level)
        self.lang   = lang
        self.model  = model if model is not None else get_model_name(lang)
        self.directory  = os.path.join(directory, self.model)
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
