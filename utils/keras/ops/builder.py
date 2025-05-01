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

import inspect
import importlib
import numpy as np

from functools import cache

from .. import timer

from .execution_contexts import *

@timer(debug = True)
def fast_is_not_tensor(x):
    return not hasattr(x, 'device') or isinstance(x, (np.ndarray, np.floating, np.integer))

class Ops:
    def __init__(self, name, disable_np = False, submodule = None, nested_arg = None, ** kwargs):
        self._keras_fn  = kwargs.pop('keras_fn', None)
        self._numpy_fn  = kwargs.pop('numpy_fn', name) if not disable_np else None
        self._tensorflow_fn = kwargs.pop('tensorflow_fn', None)
        
        self.name   = name
        self.kwargs = kwargs
        self.submodule  = submodule
        self.nested_arg = nested_arg
        self.nested_argname = None
        
        self.built  = False
        
        if isinstance(self._numpy_fn, str):
            self._numpy_fn = self.import_function(np, submodule, self._numpy_fn)
        
        if self._numpy_fn is None:
            call = self._call_keras
            if nested_arg:
                self.nested_argname = list(
                    inspect.signature(self._numpy_fn).parameters.keys()
                )[self.nested_arg]
        else:
            call = self._call_numpy_or_keras
        
        self.call   = timer(fn = call, name = self.name, debug = True)
        self.__name__   = name
    
    def build(self):
        if self.built: return
        
        wrapped = None
        if is_tensorflow_backend():
            if not self._tensorflow_fn:
                self.call_keras_fn = self.keras_fn
            else:
                self.call_keras_fn = self.tensorflow_fn
                wrapped = self.tensorflow_fn
            
        elif self._tensorflow_fn is False:
            assert callable(self._keras_fn)
            self.call_keras_fn = self.keras_fn
        else:
            self.call_keras_fn = self._call_tf_or_keras
        
        if wrapped is None: wrapped = self.keras_fn
        
        self.__doc__    = wrapped.__doc__
        self.__wrapped__    = wrapped
        self.built  = True
    
    @property
    def numpy_fn(self):
        return self._numpy_fn

    @property
    def keras_fn(self):
        if not callable(self._keras_fn):
            if self.kwargs and get_backend() + '_fn' in self.kwargs:
                module, name = get_backend_module(), self.kwargs[get_backend() + '_fn']
                if callable(name):
                    self._keras_fn = name
                    return self._keras_fn
            else:
                module, name = 'keras.ops' if self.submodule != 'random' else 'keras', self.name
            
            self._keras_fn = self.import_function(module, self.submodule, name)
            if self._keras_fn is None and not self.submodule:
                for module in ('keras.random', 'keras.ops.image'):
                    self._keras_fn = self.import_function(module, None, name)
                    if self._keras_fn is not None:
                        self.submodule = module.split('.')[-1]
                        break
            
            if self._keras_fn is None:
                raise NotImplementedError('`{}.{}` is not available for backend {}'.format(
                    module, '' if not self.submodule else '.' + self.submodule, name, get_backend()
                ))
            
        return self._keras_fn
    
    @property
    def tensorflow_fn(self):
        if not callable(self._tensorflow_fn):
            if self._tensorflow_fn is None:
                self._tensorflow_fn = self.get_backend_function(self.keras_fn, 'tensorflow')
            elif isinstance(self._tensorflow_fn, str):
                self._tensorflow_fn = self.import_function(
                    'tensorflow', self.submodule, self._tensorflow_fn
                )
            else:
                raise ValueError('Invalid `tensorflow_fn` : {}'.format(self._tensorflow_fn))
        
        return self._tensorflow_fn
    
    def __repr__(self):
        return '{}{}'.format(self.name, inspect.signature(self))
    
    def __str__(self):
        return '<Ops name={} numpy={} built={}>'.format(
            self.name, self._numpy_fn is not None, self.built
        )
    
    def __call__(self, * args, ** kwargs):
        return self.call(* args, ** kwargs)
    
    def _call_numpy_or_keras(self, * args, ** kwargs):
        if self._is_numpy(* args, ** kwargs):   return self.numpy_fn(* args, ** kwargs)
        else:                                   return self._call_keras(* args, ** kwargs)
    
    def _call_keras(self, * args, ** kwargs):
        if not self.built: self.build()
        return self.call_keras_fn(* args, ** kwargs)

    def _call_tf_or_keras(self, * args, ** kwargs):
        if is_tensorflow_graph():   return self.tensorflow_fn(* args, ** kwargs)
        else:                       return self.keras_fn(* args, ** kwargs)

    def _is_numpy(self, * args, ** kwargs):
        if self.nested_arg is None:
            return not (
                any(not fast_is_not_tensor(arg) for arg in args)
                or any(not fast_is_not_tensor(v) for v in kwargs.values())
            )
        elif self.nested_arg < len(args):
            return not any(not fast_is_not_tensor(arg) for arg in args[self.nested_arg])
        else:
            return not any(not fast_is_not_tensor(arg) for arg in kwargs[self.nested_argname])
    
    @staticmethod
    def import_function(module, submodule, name):
        parent_module, _, name = name.rpartition('.')
        if not parent_module:
            if isinstance(module, str): module = importlib.import_module(module)
        else:
            if not isinstance(module, str): module = module.__name__
            module = importlib.import_module(module + '.' + parent_module)
        
        if submodule and hasattr(module, submodule): module = getattr(module, submodule)
        return getattr(module, name, None)
    
    @staticmethod
    @cache
    def get_backend_function(keras_fn, backend):
        module, name = keras_fn.__module__.split('.')[-1], keras_fn.__name__
        
        return timer(fn = getattr(
            importlib.import_module('keras.src.backend.{}.{}'.format(backend, module)), name
        ), name = '{}_{}'.format(backend, name), debug = True)
