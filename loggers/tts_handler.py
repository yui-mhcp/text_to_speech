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

import os
import logging

from models.tts import get_audio_file, get_model_name, tts

_default_directory = os.path.join('loggers', 'tts_logs')

class TTSHandler(logging.Handler):
    def __init__(self,
                 lang   = 'en',
                 model  = None,
                 level  = 0,
                 directory  = _default_directory,
                 generate_if_not_exist  = False,
                 ** kwargs
                ):
        super().__init__(level = level)
        self.lang   = lang
        self.model  = model if model is not None else get_model_name(lang)
        self.directory  = os.path.join(directory, self.model)
        self.generate_if_not_exist = generate_if_not_exist
        
        self._generating = False
    
    def play(self, message):
        filename = get_audio_file(
            message, directory = self.directory, lang = self.lang, model = self.model
        )
        
        _is_restoring = Tacotron2._is_restoring or WaveGlow._is_restoring
        if filename is None and self.generate_if_not_exist and not _is_restoring:
            self._generating = True
            
            filename = tts(
                message,
                model       = self.model,
                directory   = self.directory,
                save_plot   = False,
                save_mel    = False,
                display     = False,
                play    = True
            )[0][1]['audio']
            
            self._generating = False
    
    def emit(self, record):
        if self._generating: return
        self.play(self.format(record))
