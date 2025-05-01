# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
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
    def __init__(self, rate = None, buffer = None, format = 'float32', ** kwargs):
        super().__init__(rate, format = format, ** kwargs)
        if buffer is None: buffer = queue.Queue()
        self.buffer = buffer
        
        self.wait_time = .25 / self.fps
    
    input   = property(lambda self: False)
    output  = property(lambda self: True)

    def stream_callback(self, data, frame_count, time_info, flags):
        import pyaudio
        
        with self.mutex:
            try:
                data = self.buffer.get(timeout = self.wait_time)
                if isinstance(data, threading.Event):
                    data.set()
                    data = self.buffer.get(timeout = self.wait_time)
            except queue.Empty:
                self._finished.set()
                return np.zeros((self.chunk_size, ), 'float32'), pyaudio.paComplete

        if data is None:             data = np.zeros((self.chunk_size, ), 'float32')
        elif isinstance(data, np.ndarray) and len(data) < self.chunk_size:
            data = np.pad(data, self.chunk_size - len(data))
        return data, pyaudio.paContinue
    
    def play(self):
        self.start()
    
    def stop(self):
        while True:
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                return
    
    def pause(self):
        self.terminate()
    
    def append(self, audio, rate, *, add_event = False, add_silence = True):
        self.rate = rate
        for s in range(0, len(audio), self.chunk_size):
            self.buffer.put(audio[s : s + self.chunk_size])
        
        event = None
        if add_event:
            event = threading.Event()
            self.buffer.put(event)
        
        if add_silence:
            self.buffer.put(None)
        
        return event

