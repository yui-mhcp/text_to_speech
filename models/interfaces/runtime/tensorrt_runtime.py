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
import logging
import numpy as np

from .runtime import Runtime
from loggers import timer, time_logger
from utils.keras_utils import ops

logger = logging.getLogger(__name__)

class TensorRTRuntime(Runtime):
    def __init__(self, * args, ** kwargs):
        import torch
        import tensorrt as trt
        
        super().__init__(* args, ** kwargs)
        
        DTYPE_MAPPING   = {
            trt.DataType.BF16   : 'bfloat16',
            
            trt.DataType.BOOL   : 'bool',
            
            trt.DataType.FLOAT  : 'float32',
            trt.DataType.HALF   : 'float16',
            
            trt.DataType.INT8   : 'int8',
            trt.DataType.INT32  : 'int32',
            trt.DataType.INT64  : 'int64',
            trt.DataType.UINT8  : 'uint8'
        }
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.prepare_fn = None
        self.num_prepare_args   = None
        if os.path.exists(os.path.splitext(self.path)[0] + '-prepare.pth'):
            self.prepare_fn = torch.jit.load(
                os.path.splitext(self.path)[0] + '-prepare.pth', map_location = self.device
            )
            self.num_prepare_args   = len(self.prepare_fn.forward.schema.arguments) - 1
        
        self.bindings   = list(self.engine)
        self._inputs    = tuple(
            n for n in self.bindings if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self._outputs   = tuple(
            n for n in self.bindings if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        )
        self._dtypes    = tuple(
            DTYPE_MAPPING[self.engine.get_tensor_dtype(n)] for n in self.bindings
        )
        self._shapes    = tuple(
            self.engine.get_tensor_shape(n) for n in self.bindings
        )
        self._var_shapes    = set([
            n for n in self._inputs if any(s == -1 for s in self.engine.get_tensor_shape(n))
        ])
    
    @property
    def inputs(self):
        return self._inputs
    
    @property
    def outputs(self):
        return self._outputs

    @property
    def dtypes(self):
        return self._dtypes
    
    @property
    def shapes(self):
        return self._shapes
    
    def __repr__(self):
        return '<TensorRTRuntime path={} inputs={} outputs={}>'.format(
            os.path.basename(self.path), self.inputs, self.outputs
        )
    
    @timer(name = 'TensorRT runtime inference')
    def __call__(self, * args, ** kwargs):
        import torch
        
        for name, arg in zip(self.bindings, args): kwargs[name] = arg
        
        if self.prepare_fn is not None:
            with time_logger.timer('pre-processing'):
                with torch.no_grad():
                    inputs = [
                        self.prepare_tensor(kwargs[k], self.shapes[i], self.dtypes[i], self.device)
                        for i, k in enumerate(self.inputs[: self.num_prepare_args])
                    ]

                    processed = self.prepare_fn(* inputs)
                    if isinstance(processed, (list, tuple)):
                        for name, t in zip(self.inputs, processed): kwargs[name] = t
                    elif isinstance(processed, dict):
                        for i, (k, v) in enumerate(processed.items()):
                            if i < len(inputs) or k not in kwargs: kwargs[k] = v
                    else:
                        kwargs[self.inputs[0]] = processed
        
        
        with self.engine.create_execution_context() as context:
            tensors = [None] * len(self.bindings)
            
            for i, name in enumerate(self.bindings):
                if name in kwargs:
                    tensor = kwargs[name]
                elif i >= len(self.inputs): # output tensor
                    tensor = torch.empty(
                        * context.get_tensor_shape(name),
                        dtype   = getattr(torch, self.dtypes[i]),
                        device  = self.device
                    )
                else:
                    raise RuntimeError('Missing input #{} : {}'.format(i, name))
                
                tensors[i] = self.prepare_tensor(
                    tensor, shape = self.shapes[i], dtype = self.dtypes[i], device = self.device
                )
                if name in self._var_shapes and i < len(self.inputs):
                    context.set_input_shape(name, tensors[i].shape)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Calling TensorRT engine with :\n  Inputs : {}\n  Outputs : {}'.format(
                    {name : tuple(tensors[i].shape) for i, name in enumerate(self.inputs)},
                    {name : tuple(tensors[i].shape) for i, name in enumerate(self.outputs, start = len(self.inputs))},
                ))

            if not context.execute_v2(bindings = [t.data_ptr() for t in tensors]):
                raise RuntimeError('An exception occured while running the TensorRT context')
            
            return tensors[len(self.inputs) :] if len(self.outputs) > 1 else tensors[-1]
    
    @staticmethod
    def prepare_tensor(tensor, shape, dtype, device):
        tensor = ops.convert_to_torch_tensor(tensor, dtype = dtype)
        
        if len(tensor.shape) < len(shape): tensor = tensor.unsqueeze(0)
        
        if TensorRTRuntime.should_permute(tensor, shape):
            last = len(tensor.shape) - 1
            tensor = tensor.permute(0, last, * range(1, last))
        
        return tensor.to(device = device)

    @staticmethod
    def should_permute(tensor, shape):
        return len(shape) > 2 and tensor.size(1) != -1 and tensor.size(1) != shape[1]
    
    @staticmethod
    def load_engine(filename, ** _):
        import tensorrt as trt
        
        with open(filename, "rb") as f, trt.Runtime(trt.Logger()) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
