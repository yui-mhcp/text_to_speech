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
from .saved_model_runtime import _redirection_wrapper, _get_tf_spec

logger  = logging.getLogger(__name__)

class ONNXRuntime(Runtime):
    def __init__(self, path, *, prepare = None, ** kwargs):
        super().__init__(path, ** kwargs)
        
        self.prepare    = prepare
        
        self.argnames   = [inp.name for inp in self.engine.get_inputs()]
        self.dtypes     = {
            inp.name : inp.type[:-1].split('(')[1]
            for inp in self.engine.get_inputs()
        }
    
    @timer(name = 'ONNX runtime inference')
    def __call__(self, * args, recompile = False, ** kwargs):
        kwargs.update({name : arg for name, arg in zip(self.argnames, args)})
        
        inputs = kwargs
        if self.prepare is not None:
            inputs = self.prepare(** inputs)

        inputs = {
            k : ops.convert_to_numpy(v, self.dtypes[k]) for k, v in inputs.items() if k in self.dtypes
        }
        outputs = self.engine.run(None, inputs)
        return outputs[0] if len(outputs) == 1 else outputs
    
    @staticmethod
    def load_engine(path, ** _):
        import onnxruntime as ort
        
        return ort.InferenceSession(path, providers = ['CUDAExecutionProvider'])

    @classmethod
    def from_tensorflow(cls, function, path, *, signature = None, opset = 18, ** kwargs):
        import tf2onnx
        import tensorflow as tf
        
        if not signature:
            if hasattr(inspect, 'get_annotations'):
                signature = inspect.get_annotations(function)
            else:
                signature = getattr(function, '__annotations__', {})
        
        names, spec = list(zip(* signature.items()))
        tf2onnx.convert.from_function(
            tf.function(_redirection_wrapper(function, names)),
            input_signature = _get_tf_spec(spec),
            opset = opset,
            output_path = path
        )
        return cls(path)

    @classmethod
    def from_torch(self, function, path, *, signature = None, opset = 18, ** kwargs):
        import torch
        
        if not signature:
            if hasattr(inspect, 'get_annotations'):
                signature = inspect.get_annotations(function)
            else:
                signature = getattr(function, '__annotations__', {})

        tensors = {
            k : _tensor_from_spec(v) for k, v in signature.items()
        }
        names, inputs = list(zip(* tensors.items()))
        
        return torch.onnx.export(
            torch.jit.script(_redirection_wrapper(function, names)),
            inputs,
            path,
            input_names = names,
            output_names    = ["output"],
            opset_version   = opset_version,
            do_constant_folding = True
        )


def _tensor_from_spec(spec):
    import torch
    
    assert spec.shape is not None, 'The shape must be defined !'
    return torch.empty(spec.shape, dtype = getattr(torch, spec.dtype or 'float32'))
