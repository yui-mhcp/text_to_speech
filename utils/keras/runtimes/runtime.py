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

    # Whether the runtime supports training (weights update). Inference-only runtimes
    # (TRT, ONNX, ...) keep it `False`; only `KerasRuntime` overrides it to `True`.
    # Named `supports_training` (not `trainable`) to avoid any confusion with
    # `keras.Model.trainable`, which means "are the weights frozen".
    supports_training   = False

    # Generation capabilities, so that models test *features* rather than runtime names
    # (`self.runtime == 'trt_llm'`). Only generation runtimes (`TensorRTLLMRuntime`)
    # override them to `True` :
    #   - `supports_streaming`  : `__call__(streaming = True)` returns a stream-like object
    #     (iterable + `abort()` / `is_aborted()`) yielding intermediate outputs
    #   - `supports_guided_decoding`    : the `allowed_tokens` kwarg restricts generation
    #     to the given token ids
    #   - `supports_stop_condition`     : the `stop_condition` kwarg (callable / regex on the
    #     decoded text) stops the generation once matched
    supports_streaming  = False
    supports_guided_decoding    = False
    supports_stop_condition     = False

    @property
    def base_dtype(self):
        """ Native compute *float* precision of the runtime (`float32` / `float16` / `bfloat16`).

            Used to build the float input / output signatures (`BaseModel.base_dtype`, modality
            mixins' `<modality>_dtype`) so they reflect the real engine precision and avoid useless
            host casts (e.g. a `float16` TensorRT engine gets `float16` mel inputs).

            Overridden per runtime ; the default is kept as `float32` (and, on purpose, avoids
            importing keras to keep inference-only runtimes keras-free). """
        return 'float32'

    def __init__(self, path, *, engine = None, reload = False, ** kwargs):
        if engine is None:
            if path not in self._engines or reload:
                self._engines[path] = self.load_engine(path, ** kwargs)
            engine = self._engines[path]
        
        self.path   = path
        self.engine = engine
    
    def __repr__(self):
        return '<{} path={}>'.format(self.__class__.__name__, self.path)

    @property
    def compiled_call(self):
        """ Graph/XLA-compiled forward pass. Inference-only runtimes are already "compiled" and
            simply return themselves ; `KerasRuntime` overrides this to compile the keras model. """
        return self

    @property
    def compiled_infer(self):
        """ Graph/XLA-compiled generation (`infer`) pass ; see `compiled_call`. """
        return self
    
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