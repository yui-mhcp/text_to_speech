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
import logging

from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)

class Runtime(metaclass = ABCMeta):
    _engines = {}
    
    def __init__(self, path, *, engine = None, reload = False, ** kwargs):
        if engine is None:
            if path not in self._engines or reload:
                self._engines[path] = self.load_engine(path, ** kwargs)
            engine = self._engines[path]
        
        self.path   = path
        self.engine = engine
    
    def __repr__(self):
        return '<{} path={}>'.format(self.__class__.__name__, self.path)
    
    @abstractmethod
    def __call__(self, * args, ** kwargs):
        """ Performs custom runtime inference """
    
    @staticmethod
    @abstractmethod
    def load_engine(path, ** kwargs):
        """ Loads the custom runtime engine """

    
    @classmethod
    def build_from(cls, function, path, overwrite = False, ** kwargs):
        if os.path.exists(path) and not overwrite:
            return cls(path, ** kwargs)

        if isinstance(function, str):
            if function.endswith('.onnx'):
                return cls.from_onnx(function, path, ** kwargs)
            elif function.endswith('.pth'):
                return cls.from_torch(function, path, ** kwargs)
            elif os.path.isdir(path):
                return cls.from_tensorflow(function, path, ** kwargs)
            else:
                raise NotImplementedError('Invalid path : {}'.format(path))
        
        import keras
        
        if keras.backend.backend() == 'tensorflow':
            return cls.from_tensorflow(function, path, ** kwargs)
        elif keras.backend.backend() == 'torch':
            return cls.from_torch(function, path, ** kwargs)
        else:
            raise NotImplementedError()

    @classmethod
    def from_tensorflow(cls, function, path, ** kwargs):
        """ Creates the engine from a `tf.function`, and saves it to `path` """
        raise NotImplementedError('{} cannot be initialized from `tf.function`'.format(cls.__name__))
    
    @classmethod
    def from_torch(cls, function, path, ** kwargs):
        """ Creates the engine from a `torch.compile`, and saves it to `path` """
        raise NotImplementedError('{} cannot be initialized from `torch.compile`'.format(cls.__name__))

    @classmethod
    def from_onnx(cls, onnx_path, path, ** kwargs):
        """ Creates the engine from a `.onnx` engine, and saves it to `path` """
        raise NotImplementedError('{} cannot be initialized from `ONNX`'.format(cls.__name__))
