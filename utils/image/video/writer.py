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

import logging
import numpy as np

from abc import ABC, abstractmethod

from ..image_io import convert_to_uint8
from .ffmpeg_reader import _get_video_size

logger = logging.getLogger(__name__)

class VideoWriter(ABC):
    def __init__(self, path, framerate, audio = None, ** kwargs):
        self.path   = path
        self.audio  = audio
        self.config = kwargs
        self.framerate  = framerate
        self.height, self.width = _get_video_size(None, kwargs)
        
    def __enter__(self):
        return self
    
    def __exit__(self, * args):
        self.release()
    
    @abstractmethod
    def write(self, frame):
        """ Write a frame """
    
    @abstractmethod
    def release(self):
        """ Release the stream """

class FFMPEGWriter(VideoWriter):
    def __init__(self, * args, input_format = 'rgb24', compression = None, loglevel = 'error', ** kwargs):
        import ffmpeg
        
        super().__init__(* args, ** kwargs)
        
        self.input_format = input_format
        if 'pix_fmt' not in self.config: self.config['pix_fmt'] = input_format
        
        if compression == False:    self.config['crf'] = '0'
        elif compression == 'low':  self.config.update({'crf' : 18, 'preset' : 'fast'})
        elif compression == 'high': self.config.update({'crf' : 28, 'preset' : 'veryslow'})

        s = self.config.pop('s', '{}x{}'.format(self.width, self.height))
        
        #self.video = self.video.filter('tonemap', tonemap='hable', desat=0, peak=0.95)
        #video = video.filter('eq', contrast='1.0', brightness='0.03', gamma='0.95', gamma_r='0.95', gamma_g='0.95', gamma_b='1.05')
        
        if isinstance(self.audio, str): self.audio = ffmpeg.input(self.audio).audio
        audio = () if self.audio is None else (self.audio, )

        if audio and 'acodec' not in self.config: self.config['acodec'] = 'copy'
        
        logger.info('Initializing writer :\n- Input config  : {}\n- Output config : {}'.format(
            {'format' : 'rawvideo' , 's' : s, 'framerate' : self.framerate, 'pix_fmt' : input_format},
            self.config
        ))

        self.writer = ffmpeg.input(
            'pipe:', format = 'rawvideo' , s = s, framerate = self.framerate, pix_fmt = input_format
        ).output(
            * audio, self.path, loglevel = loglevel, ** self.config
        ).overwrite_output().run_async(pipe_stdin = True, pipe_stdout = True)

    def __str__(self):
        return 'FFMPEGWriter {} {}>'.format(
            self.path, ' '.join(['{}={}'.format(k, v) for k, v in self.config.items()])
        )
        
    def write(self, frame):
        if isinstance(frame, np.ndarray):
            if np.issubdtype(frame.dtype, np.floating):
                out_dtype = 'uint8' if self.input_format == 'rgb24' else 'uint16'
                frame = (frame * np.iinfo(out_dtype).max).astype(out_dtype)
            
            frame = frame.tobytes()
        
        self.writer.stdin.write(frame)
    
    def release(self):
        self.writer.stdin.close()
        self.writer.stdout.close()
        self.writer.wait()

class OpenCVWriter(VideoWriter):
    def __init__(self, * args, ** kwargs):
        import cv2
        
        super().__init__(* args, ** kwargs)
        
        self.writer = cv2.VideoWriter(
            self.path, cv2.VideoWriter_fourcc(*'MPEG'), self.framerate, (self.width, self.height)
        )
        
    def write(self, frame):
        self.writer.write(convert_to_uint8(frame)[:, :, ::-1])
    
    def release(self):
        self.writer.release()
        if self.audio is not None:
            set_video_audio(self.path, self.audio)

def set_video_audio(video_filename, audio_filename, blocking = True, ** kwargs):
    basename, _, ext = video_filename.rpartition('.')
    
    cmd = ffmpeg.output(
        ffmpeg.input(video_filename).video,
        ffmpeg.input(audio_filename).audio,
        basename + '.' + ext,
        loglevel = 'error',
        ** kwargs
    ).overwrite_output()
    if blocking:
        cmd.run()
    else:
        cmd.run_async()
    
