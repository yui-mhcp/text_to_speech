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

import queue
import threading
import numpy as np

from .audio_stream import AudioStream

class AudioPlayer(AudioStream):
    def __init__(self, rate, buffer, format = 'float32', ** kwargs):
        super().__init__(rate, format = format, ** kwargs)
        self.buffer = buffer
        
        self.wait_time = .25 / self.fps
    
    input   = property(lambda self: False)
    output  = property(lambda self: True)

    def stream_callback(self, data, frame_count, time_info, flags):
        import pyaudio
        
        try:
            data = self.buffer.get(timeout = self.wait_time)
            if isinstance(data, threading.Event):
                data.set()
                data = self.buffer.get(timeout = self.wait_time)
        except queue.Empty:
            return np.zeros((self.chunk_size, ), 'float32'), pyaudio.paComplete

        if data is None:             data = np.zeros((self.chunk_size, ), 'float32')
        elif isinstance(data, np.ndarray) and len(data) < self.chunk_size:
            data = np.pad(data, self.chunk_size - len(data))
        return data, pyaudio.paContinue
    
    def play(self):
        self.start()
    
    def stop(self):
        self.buffer.clear()
    
    def pause(self):
        self.terminate()
    
