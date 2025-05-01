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

from functools import cache

logger = logging.getLogger(__name__)

_hdr_default_config = {
    'color_range'       : 'tv',
    'color_space'       : 'bt2020nc',
    'color_transfer'    : 'smpte2084',
    'color_primaries'   : 'bt2020',
}
_hdr_mapping = {
    'color_space' : 'colorspace', 'color_transfer' : 'color_trc'
}

class FFMPEGReader:
    def __init__(self,
                 path,
                 *,
                 
                 to_numpy   = True,
                 batch_size = 0,
                 
                 hdr    = 'auto',
                 use_10bits = False,
                 
                 quiet  = False,
                 ** kwargs
                ):
        import ffmpeg
        
        assert isinstance(path, str), 'The path should be a file/device path, got {}'.format(path)
        
        height, width = _get_video_size(path, kwargs)

        self.path   = path
        self.config = kwargs
        self.width  = width
        self.height = height
        self.use_10bits = use_10bits
        
        self.to_numpy   = to_numpy
        self.batch_size = batch_size
        
        if not self.is_file:
            if 'format' not in self.config:       self.config['format'] = 'v4l2'
            if 'input_format' not in self.config: self.config['input_format'] = self.probe['pix_fmt']
            if 's' not in self.config:
                self.config['s'] = '{}x{}'.format(width, height)

        
        if hdr is True:
            hdr = {
                _hdr_mapping.get(k, k) : self.probe.get(k, v)
                for k, v in _hdr_default_config.items()
            }
        elif hdr == 'auto':
            hdr = {
                _hdr_mapping.get(k, k) : self.probe.get(k, None) for k in _hdr_default_config
            }
            hdr = {k : v for k, v in hdr.items() if v}
        
        self.hdr_config = hdr or {}
        
        
        self.bytes_per_frame    = int(self.width * self.height * self.bytes_per_pixel)
        if self.use_10bits: self.bytes_per_frame *= 2
        if self.batch_size: self.bytes_per_frame *= self.batch_size
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Initializing ffmpeg reader with\n- Input config  : {}\n- Output config : {}'.format(self.config, {'pix_fmt' : self.pixel_format}))
        

        self.output_stream = (
            ffmpeg.input(self.path, ** self.config)
            .output('pipe:', format = 'rawvideo', pix_fmt = self.pixel_format)
            .run_async(pipe_stdin = True, pipe_stdout = True, quiet = quiet)
        )
    
    @property
    def is_file(self):
        return self.probe['codec_name'] != 'rawvideo'

    @property
    def framerate(self):
        if 'framerate' in self.config: return self.config['framerate']
        num, den = self.probe['avg_frame_rate'].split('/')
        return int(num) / int(den)

    @property
    def pixel_format(self):
        if self.to_numpy:
            return 'rgb24' if not self.use_10bits else 'rgb48le'
        else:
            fmt = self.probe['pix_fmt']
            if fmt.endswith('10le'):
                return fmt if self.use_10bits else fmt.replace('10le', '')
            else:
                return fmt if not self.use_10bits else fmt + '10le'
    
    @property
    def probe(self):
        return get_video_infos(self.path)

    @property
    def bytes_per_pixel(self):
        if self.pixel_format.startswith('rgb'):
            return 3
        elif self.pixel_format.startswith('yuv420p'):
            return 1.5
        else:
            raise NotImplementedError()

    @property
    def full_config(self):
        return {'pix_fmt' : self.pixel_format, ** self.config, ** self.hdr_config}
    
    def __len__(self):
        return -1 if not self.is_file else int(self.probe['rames'])
    
    def __str__(self):
        return 'FFMPEG {}={} {}'.format(
            'file' if self.is_file else 'device',
            self.path,
            ' '.join(['{}={}'.format(k, v) for k, v in self.full_config.items()])
        )

    def __enter__(self):
        return self

    def __exit__(self, * args):
        self.release()

    def __iter__(self):
        ret = True
        while ret:
            ret, frame = self.read()
            if ret: yield frame

    def read(self):
        data = self.output_stream.stdout.read(self.bytes_per_frame)
        if len(data) == 0:
            return False, None
        elif self.to_numpy:
            dtype = 'uint8' if not self.use_10bits else 'uint16'
            shape = (self.height, self.width, 3)
            if self.batch_size: shape = (-1, ) + shape
            
            return True, np.frombuffer(data, dtype).reshape(shape)
        else:
            return True, data

    def release(self):
        try:
            self.output_stream.stdin.close()
            self.output_stream.stdout.close()
            self.output_stream.wait()
        except Exception as e:
            logger.error('An error occured while closing {} : {}'.format(self.path, e))

    def get_writer_config(self, codec = 'libx264', compression = None, ** kwargs):
        if self.hdr_config and 'x264-params' not in kwargs and 'x264-params':
            kwargs['x264-params'] = 'hdr-opt=1:repeat-headers=1:colorprim={}:transfer={}:colormatrix={}:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):max-cll=1000,400'.format(
                self.hdr_config['color_primaries'],
                self.hdr_config['color_trc'],
                self.hdr_config['colorspace'],
            )
        
        return {
            ** {k : v for k, v in self.config.items() if 'format' not in k and k != 't'},
            's' : '{}x{}'.format(self.width, self.height),
            'framerate' : self.framerate,
            ** self.hdr_config,
            ** kwargs,
            'input_format' : self.pixel_format,
            'vcodec' : codec
        }

@cache
def probe(path):
    import ffmpeg
    
    return ffmpeg.probe(path)

def get_video_infos(path):
    return next(s for s in probe(path)['streams'] if s['codec_type'] == 'video')

def _get_video_size(path, kwargs):
    if 'image_h' in kwargs and 'image_w' in kwargs: return (kwargs['image_h'], kwargs['image_w'])
    elif 'height' in kwargs and 'width' in kwargs:  return (kwargs['height'], kwargs['width'])
    elif 'image_shape' in kwargs:   return kwargs['image_shape'][:2]
    elif 's' in kwargs:
        w, h = kwargs['s'].split('x')
        return (int(h), int(w))
    elif path:
        infos = get_video_infos(path)
        return (int(infos['height']), int(infos['width']))
    else:
        raise ValueError('You must specify the frame shape')
    
        
