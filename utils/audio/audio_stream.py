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

import time
import logging
import threading

from utils.threading import run_in_thread

logger = logging.getLogger(__name__)

class AudioStream:
    def __init__(self, rate = 22050, fps = 10, format = 'float32', channels = 1, ** _):
        self.fps    = fps
        self.rate   = rate
        self.format = format
        self.channels   = channels
        self.chunk_size = rate // fps
        
        self.mutex  = threading.RLock()
        self._audio = None
        self._stream    = None
        self._finalizer = None

    def start(self):
        import pyaudio

        with self.mutex:
            if self.is_active(): return self
    
            self._audio = pyaudio.PyAudio()
            self._stream    = self._audio.open(
                rate    = self.rate,
                format  = getattr(pyaudio, 'pa' + self.format.capitalize()),
                output  = self.output,
                input   = self.input,
                channels    = self.channels,
                stream_callback = self.stream_callback,
                frames_per_buffer   = self.chunk_size
            )
            logger.debug('{} is started'.format(self.__class__.__name__))
            
            self._finalizer = self.start_finalizer()
        
        return self
    
    def join(self, ** kwargs):
        if self._finalizer is not None: self._finalizer.join(** kwargs)
        
    def pause(self):
        self.terminate()
    
    def is_active(self):
        with self.mutex:
            return self._stream is not None and self._stream.is_active()
    
    def terminate(self, force = True):
        with self.mutex:
            if self._stream is None: return
            elif not force and self._stream.is_active(): return
            
            self._stream.close()
            self._audio.terminate()
            
            self._stream = None
            self._audio  = None
        
        logger.debug('{} is stopped'.format(self.__class__.__name__))
    
    @run_in_thread
    def start_finalizer(self):
        while self.is_active():
            time.sleep(1. / self.fps)
        
        self.terminate(force = False)

