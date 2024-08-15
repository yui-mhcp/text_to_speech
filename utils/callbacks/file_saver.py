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
import re
import glob
import logging

from threading import Lock

from .callback import Callback
from loggers import time_logger
from utils.threading import Consumer
from utils.generic_utils import to_json
from utils.file_utils import dump_data, dump_json
from utils.keras_utils import ops

logger = logging.getLogger(__name__)

_index_file_format_re = re.compile(r'\{i?(:\d{2}d)?\}')

class FileSaver(Callback):
    def __init__(self,
                 data_key,
                 file_format,
                 
                 index  = -1,
                 index_key  = None,
                 
                 save_fn    = dump_data,
                 use_multithreading = False,
                 
                 name   = 'saving',
                 
                 ** kwargs
                ):
        super().__init__(name = name, ** kwargs)
        
        self.data_key   = data_key
        self.file_format    = file_format

        self.index  = index
        self.index_key  = index_key
        self.use_index  = _index_file_format_re.search(file_format) is not None
        
        self.save_fn    = save_fn
        self.use_multithreading = use_multithreading
    
    def __repr__(self):
        des = '<{}'.format(self.__class__.__name__)
        if self.key:        des += ' key={}'.format(self.key)
        if self.data_key:   des += ' data_key={}'.format(self.data_key)
        return des + '>'
    
    def build(self):
        super().build()
        
        directory = os.path.dirname(self.file_format)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if self.use_multithreading:
            self.save = Consumer(self._save).start()
        else:
            self.save = self._save
        
    def apply(self, infos, output, ** _):
        if isinstance(output.get(self.key, None), str):
            return output[self.key]
        elif isinstance(infos.get(self.key, None), str):
            filename = infos[self.key]
        else:
            filename = self.format_filename(infos, output)
        
        self.save(filename, output[self.data_key])
        
        return filename

    def join(self):
        if self.use_multithreading and self.built: self.save.join()
    
    def format_filename(self, infos, output):
        idx     = self.get_index(output)
        kwargs  = {}
        if '{basename}' in self.file_format and 'basename' not in output:
            kwargs['basename'] = '.'.join(
                os.path.basename(infos['filename']).split('.')[:-1]
            )
        
        return self.file_format.format(idx, i = idx, ** output, ** kwargs)
    
    def get_index(self, output):
        if not self.use_index: return -1
        
        if self.index_key in output:
            return output[self.index_key]
        elif self.index == -1:
            self.index = len(glob.glob(_index_file_format_re.sub('*', self.file_format)))
        
        idx = self.index
        self.index += 1
        return idx
    
    def _save(self, filename, data):
        with time_logger.timer(self.name):
            self.save_fn(filename, data)

class AudioSaver(FileSaver):
    def __init__(self, data_key = 'audio', file_format = 'audio-{}.mp3', ** kwargs):
        if 'save_fn' not in kwargs:
            from utils.audio import save_audio
            kwargs['save_fn'] = save_audio
        
        super().__init__(data_key = data_key, file_format = file_format, ** kwargs)

class ImageSaver(FileSaver):
    def __init__(self, data_key = 'image', file_format = 'image-{}.jpg', ** kwargs):
        if 'save_fn' not in kwargs:
            from utils.image import save_image
            kwargs['save_fn'] = save_image
        
        super().__init__(data_key = data_key, file_format = file_format, ** kwargs)

class JSonSaver(FileSaver):
    def __init__(self,
                 filename,
                 data,
                 primary_key,
                 
                 use_multithreading = False,

                 name   = 'saving json',
                 
                 ** _
                ):
        super().__init__(
            name    = name,
            data_key    = None,
            file_format = filename,
            use_multithreading = use_multithreading
        )
        
        self.data   = data
        self.filename   = filename
        self.primary_key    = primary_key
    
        if self.use_multithreading:
            self.mutex = Lock()
        
    def __repr__(self):
        return '<{} file={}>'.format(self.__class__.__name__, self.filename)

    def update_data(self, infos, output):
        key = infos[self.primary_key] if self.primary_key in infos else output[self.primary_key]
        if isinstance(key, str):
            _updated    = []
            for k, v in output.items():
                if ops.is_array(v): continue
                v = to_json(v)
                if k in infos and infos[k] == v:
                    continue

                infos[k] = v
                _updated.append(k)
            
            if key not in self.data:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('- Add new entry {} to data'.format(key))
            
                self.data[key] = to_json(infos)
            elif not _updated:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('- Entry {} is already in data'.format(key))
                return False
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug(' - Keys {} have been updated for entry {}'.format(_updated, key))
            
            if self.use_multithreading:
                with self.mutex: self.updated = True
            return True
        
        return False
    
    def apply(self, infos, output):
        self.save()
    
    def _save(self):
        with time_logger.timer(self.name):
            data = self.data
            if self.use_multithreading:
                with self.mutex:
                    if not self.updated: return
                    self.updated = False
                    data = self.data.copy()
            else:
                data = self.data
            dump_json(self.filename, data, indent = 4)
    
