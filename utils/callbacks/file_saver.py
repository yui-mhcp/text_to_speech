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

import os
import re
import glob
import numpy as np

from threading import Lock

from loggers import Timer
from .callback import Callback
from ..threading import Stream, FakeLock
from ..file_utils import dump_data, dump_json
from ..generic_utils import to_json
from ..keras import ops

_index_file_format_re = re.compile(r'\{i?(:\d{2}d)?\}')

class FileSaver(Callback):
    def __init__(self,
                 key,
                 file_format,
                 
                 *,
                 
                 data_key   = None,
                 additional_keys    = None,
                 
                 index  = -1,
                 index_key  = None,
                 
                 save_fn    = dump_data,
                 save_in_parallel   = False,
                 
                 name   = None,
                 
                 ** kwargs
                ):
        super().__init__(name = name or 'saving {}'.format(key), ** kwargs)
        
        self.key    = key
        self.data_key   = data_key or key
        self.file_format    = file_format
        self.additional_keys    = additional_keys or []
        
        self.index  = index
        self.index_key  = index_key
        self.use_index  = _index_file_format_re.search(file_format) is not None

        self.save_fn    = save_fn
        self.save_in_parallel   = int(save_in_parallel)
    
    def _get_index(self, output):
        if not self.use_index: return -1
        
        if self.index_key in output:
            return output[self.index_key]
        
        with self.mutex:
            if self.index == -1:
                self.index = len(glob.glob(_index_file_format_re.sub('*', self.file_format)))

            idx = self.index
            self.index += 1
        return idx
    
    def _format_filename(self, infos, output):
        idx     = self._get_index(output)
        kwargs  = {}
        if '{basename}' in self.file_format and 'basename' not in output:
            kwargs['basename'] = '.'.join(
                os.path.basename(infos['filename']).split('.')[:-1]
            )
        
        return self.file_format.format(idx, i = idx, ** output, ** kwargs)

    def __repr__(self):
        des = '<{}'.format(self.__class__.__name__)
        if self.key:        des += ' key={}'.format(self.key)
        if self.data_key:   des += ' data_key={}'.format(self.data_key)
        return des + '>'
    
    def build(self):
        super().build()
        
        directory = os.path.dirname(self.file_format)
        if not os.path.exists(directory): os.makedirs(directory)
        
        self.saver  = Stream(self.save, max_workers = self.save_in_parallel)
        self.mutex  = Lock() if self.save_in_parallel > 1 and self.use_index else FakeLock()
        
    def apply(self, infos, output, ** _):
        if isinstance(output.get(self.key, None), str):
            if self.key not in infos: infos[self.key] = output[self.key]
            return
        elif infos.get(self.key, None) is None:
            infos[self.key] = self._format_filename(infos, output)
        
        self.saver(infos[self.key], output[self.data_key], ** {
            k : output[k] for k in self.additional_keys
        })

    def join(self):
        if self.built: self.saver.join()
    
    def save(self, filename, data, ** kwargs):
        with Timer(self.name): self.save_fn(filename, data, ** kwargs)

class AudioSaver(FileSaver):
    def __init__(self, key = 'audio', file_format = 'audio-{}.mp3', ** kwargs):
        if 'save_fn' not in kwargs:
            from utils.audio import write_audio
            kwargs['save_fn'] = write_audio
        
        kwargs['additional_keys'] = ['rate']
        super().__init__(key, file_format, ** kwargs)

class ImageSaver(FileSaver):
    def __init__(self, key = 'filename', file_format = 'image-{}.jpg', data_key = 'image', ** kwargs):
        if 'save_fn' not in kwargs:
            from utils.image import save_image
            kwargs['save_fn'] = save_image
        
        super().__init__(key, file_format, data_key = data_key, ** kwargs)

class SpectrogramSaver(FileSaver):
    def __init__(self, key = 'mel', file_format = 'mel-{}.npy', ** kwargs):
        super().__init__(key, file_format, ** kwargs)

    def save(self, filename, data):
        if isinstance(data, list):
            data = [ops.convert_to_numpy(d) for d in data]
            data = np.concatenate(data, axis = 0)
        return super().save(filename, data)
    
class JSONSaver(FileSaver):
    def __init__(self,
                 data,
                 filename,
                 primary_key,
                 
                 *,
                 
                 force_keys = (),
                 
                 name   = 'saving json',
                 
                 ** kwargs
                ):
        super().__init__(None, filename, name = name, ** kwargs)
        
        self.data   = data
        self.force_keys = force_keys
        self.primary_key    = primary_key
        
        self.save_in_parallel = min(1, self.save_in_parallel)
    
    def __repr__(self):
        return '<{} file={}>'.format(self.__class__.__name__, self.file_format)

    def apply(self, infos, output):
        if self.primary_key not in infos: return None
        
        key = infos[self.primary_key]
        if not isinstance(key, str): return

        infos = to_json(infos)
        with self.mutex:
            self.data[key]  = infos
            self.updated    = True
        
        self.saver()
        return key
    
    def save(self):
        with Timer(self.name):
            data = self.data
            if self.save_in_parallel:
                with self.mutex:
                    if not self.updated: return
                    self.updated = False
                    data = data.copy()
            dump_json(self.file_format, data, indent = 4)
    
