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

import time
import logging
import inspect

from functools import cached_property, partial, wraps

from .. import ops, timer
from .runtime import Runtime

logger  = logging.getLogger(__name__)

class SavedModelRuntime(Runtime):
    def __init__(self,
                 path,
                 
                 *,
                 
                 prepare    = None,
                 prepare_for_xla    = None,
                 prepare_for_graph  = None,
                 
                 prefer_xla = False,
                 
                 ** kwargs
                ):
        super().__init__(path, ** kwargs)
        
        self.prepare    = prepare
        self.prepare_for_xla    = prepare_for_xla
        self.prepare_for_graph  = prepare_for_graph
        
        self.prefer_xla = prefer_xla
        
        self.signatures = {
            k : v.structured_input_signature[1]
            for k, v in self.engine.signatures.items()
            if hasattr(self.engine, k)
        }
        
        if hasattr(self.engine, 'serve'):
            self._default = self.engine.serve
        else:
            self._default = getattr(self.engine, list(self.signatures.keys())[0])
        
        self.xla_compiled   = {}
    
    @property
    def argnames(self):
        return self.engine.signatures[list(self.signatures.keys())[0]]._arg_keywords
    
    @property
    def endpoints(self):
        return {
            name : getattr(self.engine, name) for name in self.signatures
        }
    
    def __repr__(self):
        return '<SavedModel runtime path={} endpoints={}>'.format(
            self.path, tuple(self.endpoints.keys())
        )
    
    @timer(name = 'SavedModel runtime inference')
    def __call__(self, * args, use_xla = None, recompile = False, ** kwargs):
        if use_xla is None: use_xla = self.prefer_xla
        kwargs.update({name : arg for name, arg in zip(self.argnames, args)})
        
        inputs = kwargs
        if self.prepare is not None:
            inputs = self.prepare(** inputs)
        if self.prepare_for_graph is not None:
            inputs = self.prepare_for_graph(** inputs)
        if use_xla and self.prepare_for_xla is not None:
            inputs = self.prepare_for_xla(** inputs)
        
        inputs = {k : v for k, v in inputs.items() if k in self.argnames}
        
        name, signature = self.find_endpoint(inputs)
        
        inputs = {
            k : ops.convert_to_tf_tensor(v, signature[k].dtype) for k, v in inputs.items()
        }

        endpoint = getattr(self.engine, name)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('The endpoint {} is executed with {}'.format(
                name, {k : tuple(v.shape) for k, v in inputs.items()}
            ))

        if not use_xla or ops.is_tensorflow_graph():
            return endpoint(** inputs)

        if recompile or name not in self.xla_compiled:
            import tensorflow as tf
            
            self.xla_compiled[name] = timer(
                fn = tf.function(endpoint, jit_compile = True), name = 'xla_{}'.format(name)
            )

        return self.xla_compiled[name](** inputs)

    def find_endpoint(self, inputs):
        candidates = list(self.signatures.items())
        if len(candidates) == 1: return candidates[0]
        
        for name, sign in candidates:
            if _is_compatible_with(sign, inputs):
                return name, sign
        
        raise RuntimeError('No endpoint is compatible with the provided inputs\n  Got : {}\n  Accepted : {}'.format(inputs, {k : v for k, v in candidates}))
    
    @staticmethod
    def load_engine(path, ** _):
        import tensorflow as tf
        
        return tf.saved_model.load(path)

    @classmethod
    def from_tensorflow(cls, function, path, *, signatures = None, endpoints = None, ** kwargs):
        import keras
        
        if not signatures and not endpoints:
            if hasattr(inspect, 'get_annotations'):
                signatures = inspect.get_annotations(function)
            else:
                signatures = getattr(function, '__annotations__', {})

        assert signatures or endpoints
        
        if not endpoints: endpoints = {'serve' : signatures}
        
        archive = keras.export.ExportArchive()
        for endpoint, signature in endpoints.items():
            signature = _get_tf_spec(signature)
            if not isinstance(signature, (list, tuple, dict)):
                signature = [signature]
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Computing function for endpoint {} with signature : {}'.format(
                    endpoint, signature
                ))

            compiled_fn = function
            if isinstance(signature, dict) and any(not hasattr(v, 'shape') for v in signature.values()):
                static      = {k : v for k, v in signature.items() if not hasattr(v, 'shape')}
                signature   = {k : v for k, v in signature.items() if k not in static}
                compiled_fn = partial(compiled_fn, ** static)

            if not signature:
                raise ValueError('Signature for endpoint {} is empty'.format(endpoint))

            if isinstance(signature, dict):
                keys, values = list(zip(* signature.items()))
                compiled_fn = _redirection_wrapper(compiled_fn, keys)
                signature = list(values)

            archive.add_endpoint(name = endpoint, fn = compiled_fn, input_signature = signature)

        return cls(archive.write_out(path))

def _is_compatible_with(s1, s2):
    if isinstance(s1, dict):
        return all(
            _is_compatible_with(s1_v, s2[k]) for k, s1_v in s1.items()
        )
    elif isinstance(s1, (list, tuple)):
        return all(_is_compatible_with(s1_i, s2_i) for s1_i, s2_i in zip(s1, s2))
    
    elif s1.shape.rank is None: return True
    return len(s1.shape) == len(s2.shape) and all(
        s1_d is None or s1_d == s2_d for s1_d, s2_d in zip(s1.shape, s2.shape)
    )

def _redirection_wrapper(fn, keys):
    @wraps(fn)
    def wrapped(* args, ** kwargs):
        kwargs.update({name : arg for name, arg in zip(keys, args)})
        return fn(** kwargs)
    
    wrapped.__signature__ = inspect.Signature([
        inspect.Parameter(name = name, kind = inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in keys
    ])

    return wrapped

def _get_tf_spec(value):
    if isinstance(value, dict):
        return {k : _get_tf_spec(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return value.__class__(_get_tf_spec(v) for v in value)
    elif not hasattr(value, 'shape'):
        return value
    
    import tensorflow as tf
    
    dtype = getattr(value.dtype, 'name', value.dtype) or 'float32'
    return tf.TensorSpec(shape = value.shape, dtype = dtype)
