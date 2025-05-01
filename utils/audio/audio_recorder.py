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

import numpy as np

from .audio_stream import AudioStream

class AudioRecorder(AudioStream):
    def __init__(self, callback, format, rate = None, channels = None, max_time = 10., ** kwargs):
        super().__init__(rate, format = format, channels = channels, ** kwargs)
        self.callback   = callback
        self.max_time   = max_time
        
        self.audio_chunks   = []

    
    input   = property(lambda self: True)
    output  = property(lambda self: False)
    
    @property
    def audio(self):
        return np.concatenate(self.audio_chunks, axis = -1)
    
    def stream_callback(self, data, frame_count, time_info, flags):
        import pyaudio
        
        chunk = np.frombuffer(data, self.format).reshape((-1, self.channels)).T
        self.audio_chunks.append(chunk)
        if self.callback is not None: self.callback(chunk)
        
        if self.max_time and len(self.audio_chunks) / self.fps >= self.max_time:
            status = pyaudio.paComplete
            self._finished.set()
        else:
            status = pyaudio.paContinue

        return data, status
    
    def record(self):
        self.start()
    
    def stop(self):
        self.terminate()

