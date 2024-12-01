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

import time
import logging

from functools import cached_property

from loggers import timer
from .runtime import Runtime
from utils.keras_utils import ops
from utils import time_to_string, get_args, get_kwargs, get_annotations, args_to_str

logger  = logging.getLogger(__name__)

class SavedModelRuntime(Runtime):
    def __init__(self,
                 * args,
                 
                 prepare    = None,
                 prepare_for_xla    = None,
                 prepare_for_graph  = None,
                 
                 prefer_xla = False,
                 
                 ** kwargs
                ):
        super().__init__(* args, ** kwargs)
        
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
        
        self.default_kwargs = tuple(get_kwargs(self._default).items())
        self.static_kwargs  = set(self.default_kwargs.keys())
        
        self.xla_compiled   = {}
    
    @property
    def endpoints(self):
        return {
            name : getattr(self.engine, name) for name in self.signatures
        }

    @cached_property
    def args(self):
        return get_args(self._default)
    
    @cached_property
    def kwargs(self):
        return get_kwargs(self._default)
    
    @cached_property
    def arg_names(self):
        return set(list(self.args) + list(self.kwargs.keys()))
    
    @cached_property
    def input_names(self):
        return [arg for arg in self.args if arg not in self.kwargs]
    
    @cached_property
    def prepare_fn_kwargs(self):
        p_args = [] if self.prepare is None else list(get_kwargs(self.prepare).keys())
        g_args = [] if self.prepare_for_graph is None else list(get_kwargs(self.prepare_for_graph).keys())
        x_args = [] if self.prepare_for_xla is None else list(get_kwargs(self.prepare_for_xla).keys())
        
        return set(p_args + g_args + x_args)
    
    @cached_property
    def endpoint_kwargs(self):
        infos = {
            name : (get_kwargs(getattr(self.engine, name)), signature)
            for name, signature in self.signatures.items()
        }
        
        endpoints = {}
        for name, (kwargs, signature) in infos.items():
            endpoints.setdefault(tuple(kwargs.items()), []).append((name, signature))

        for kw, signatures in endpoints.items():
            endpoints[kw] = sorted(
                signatures, reverse = True, key = lambda info: sum(
                    -1 if sign.shape.rank is None else sum(s is not None for s in sign.shape)
                    for sign in info[1].values()
                )
            )
        
        return endpoints
    
    def __repr__(self):
        return '<SavedModel runtime path={} endpoints={}>'.format(
            self.path, tuple(self.endpoints.keys())
        )
    
    @timer(name = 'SavedModel runtime inference')
    def __call__(self, * args, use_xla = None, recompile = False, ** kwargs):
        if use_xla is None: use_xla = self.prefer_xla
        
        prep_kwargs, inputs = self.prepare_inputs(args, kwargs)
        
        if self.prepare is not None:
            inputs = self.prepare(** {** prep_kwargs, ** inputs})
        if self.prepare_for_graph is not None:
            inputs = self.prepare_for_graph(** {** prep_kwargs, ** inputs})
        if use_xla and self.prepare_for_xla is not None:
            inputs = self.prepare_for_xla(** {** prep_kwargs, ** inputs})
        
        static_kwargs = {k : v for k, v in inputs.items() if k in self.static_kwargs}
        inputs        = {arg : inputs[arg] for arg in self.input_names}
        
        name, signature = self.find_endpoint(inputs, static_kwargs)
        
        inputs = {
            k : ops.convert_to_tf_tensor(v, signature[k].dtype) for k, v in inputs.items()
        }

        endpoint = getattr(self.engine, name)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('The endpoint {} is executed with {}'.format(
                name, args_to_str(kwargs = inputs)
            ))

        if not use_xla or not ops.executing_eagerly():
            return endpoint(** inputs)

        if recompile or name not in self.xla_compiled:
            import tensorflow as tf
            
            self._compiled[name] = timer(
                tf.function(endpoint, jit_compile = True), name = 'xla_{}'.format(name)
            )

        return self.xla_compiled[name](** inputs)

    def prepare_inputs(self, args, kwargs):
        args_dict   = {name : arg for name, arg in zip(self.args, args)}
        prep_kwargs = {k : v for k, v in kwargs.items() if k in self.prepare_fn_kwargs}
        
        unsupported = set(kwargs.keys()) - self.arg_names
        if unsupported:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Removing kwargs {}'.format(unsupported))
            for k in unsupported: kwargs.pop(k)
        
        kwargs.update(args_dict)
        return prep_kwargs, kwargs

    def find_endpoint(self, inputs, static_kwargs):
        candidates = self.endpoint_kwargs
        if len(candidates) == 1:
            candidates = list(candidates.values())[0]
        elif static_kwargs:
            for _kwargs, _candidates in candidates.items():
                if all(static_kwargs.get(k, v) == v for k, v in _kwargs):
                    candidates = _candidates
                    break
            
            if isinstance(candidates, dict):
                raise RuntimeError('No endpoint corresponds to the set of static kwargs\n  Got : {}\n  Accepted : {}'.format(static_kwargs, tuple(candidates.keys())))
        else:
            candidates = candidates[self.default_kwargs]
        
        if len(candidates) == 1:
            return candidates[0]
        
        for name, sign in candidates:
            if is_compatible_with(sign, tensors):
                return name, sign
        
        raise RuntimeError('No endpoint is compatible with the provided inputs\n  Got : {}\n  Accepted : {}'.format(_inp_to_str(inputs), {k : v for k, v in candidates}))

    def compile(self, use_xla = False):
        import tensorflow as tf
        
        for name, sign in self.signatures.items():
            if all(s.shape.is_fully_defined() for s in sign.values()):
                t0 = time.time()
                
                inputs = {k : tf.zeros(v.shape, dtype = v.dtype) for k, v in sign.items()}
                self(** inputs, use_xla = use_xla)
                
                logger.info('Endpoint {} compiled in {:.3f} sec'.format(
                    name, time_to_string(time.time() - t0)
                ))
    
    @staticmethod
    def load_engine(directory, ** _):
        import tensorflow as tf
        
        return tf.saved_model.load(directory)

def is_compatible_with(s1, s2):
    if isinstance(s1, dict):
        return all(
            is_compatible_with(s1_v, s2[k]) for k, s1_v in s1.items()
        )
    elif isinstance(s1, (list, tuple)):
        return all(is_compatible_with(s1_i, s2_i) for s1_i, s2_i in zip(s1, s2))
    
    elif s1.shape.rank is None: return True
    return len(s1.shape) == len(s2.shape) and all(
        s1_d is None or s1_d == s2_d for s1_d, s2_d in zip(s1.shape, s2.shape)
    )
