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
import sys
import logging
import numpy as np

from .. import ops, timer
from .runtime import Runtime
from .onnx_runtime import ONNXRuntime

logger = logging.getLogger(__name__)

class TensorRTRuntime(Runtime):
    def __init__(self, path, *, prepare = None, ** kwargs):
        import torch
        import tensorrt as trt
        
        super().__init__(path, ** kwargs)
        
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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_prepare_args   = None
        
        if not prepare and os.path.exists(os.path.splitext(self.path)[0] + '-prepare.pth'):
            prepare = os.path.splitext(self.path)[0] + '-prepare.pth'
        
        if isinstance(prepare, str):
            prepare = torch.jit.load(prepare, map_location = self.device)

        self.prepare    = prepare
        self.num_prepare_args   = None if not prepare else len(prepare.forward.schema.arguments) - 1
        
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
    def argnames(self):
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
            os.path.basename(self.path), self.argnames, self.outputs
        )
    
    @timer(name = 'TensorRT runtime inference')
    def __call__(self, * args, ** kwargs):
        import torch
        
        kwargs.update({name : arg for name, arg in zip(self.bindings, args)})
        
        if self.prepare is not None:
            with torch.no_grad():
                inputs = [
                    self.prepare_tensor(kwargs[k], self.shapes[i], self.dtypes[i], self.device)
                    for i, k in enumerate(self.argnames[: self.num_prepare_args])
                ]

                processed = self.prepare(* inputs)
                if isinstance(processed, (list, tuple)):
                    kwargs.update({name : t for name, t in zip(self.argnames, processed)})
                elif isinstance(processed, dict):
                    kwargs.update({
                        k : v for i, (k, v) in enumerate(processed.items())
                        if i < len(inputs) or k not in kwargs
                    })
                else:
                    kwargs[self.argnames[0]] = processed
        
        
        with self.engine.create_execution_context() as context:
            tensors = [None] * len(self.bindings)
            
            for i, name in enumerate(self.bindings):
                if name in kwargs:
                    tensors[i] = self.prepare_tensor(
                        kwargs[name], self.shapes[i], dtype = self.dtypes[i], device = self.device
                    )
                    if name in self._var_shapes and i < len(self.argnames):
                        context.set_input_shape(name, tensors[i].shape)

                elif i >= len(self.argnames): # output tensor
                    tensors[i] = torch.empty(
                        tuple(context.get_tensor_shape(name)),
                        dtype   = getattr(torch, self.dtypes[i]),
                        device  = self.device
                    )
                else:
                    raise RuntimeError('Missing input #{} : {}'.format(i, name))
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Calling TensorRT engine with :\n  Inputs : {}\n  Outputs : {}'.format(
                    {name : tuple(tensors[i].shape) for i, name in enumerate(self.argnames)},
                    {name : tuple(tensors[i].shape) for i, name in enumerate(self.outputs, start = len(self.argnames))},
                ))

            if not context.execute_v2(bindings = [t.data_ptr() for t in tensors]):
                raise RuntimeError('An exception occured while running the TensorRT context')
            
            return tensors[len(self.argnames) :] if len(self.outputs) > 1 else tensors[-1]
    
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
    
    @classmethod
    def from_tensorflow(cls, function, path, ** kwargs):
        onnx_runtime = ONNXRuntime.from_tensorflow(
            function, path.replace('.engine', '.onnx'), ** kwargs
        )
        return cls.from_onnx(onnx_runtime.path, path, ** kwargs)

    @classmethod
    def from_torch(cls, function, path, ** kwargs):
        onnx_runtime = ONNXRuntime.from_torch(
            function, path.replace('.engine', '.onnx'), ** kwargs
        )
        return cls.from_onnx(onnx_runtime.path, path, ** kwargs)

    @classmethod
    def from_onnx(cls, function, path, *, workspace = 16, debug = False, ** kwargs):
        assert isinstance(function, str) and function.endswith('.onnx')
        
        import tensorrt as trt
        
        trt_logger = trt.Logger(trt.Logger.INFO)
        if debug: trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(trt_logger)
        config  = builder.create_builder_config()

        #explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(0)
        with trt.OnnxParser(network, trt_logger) as parser:
            with open(function, "rb") as f:
                if not parser.parse(f.read()):
                    logger.error("Failed to load ONNX file: {}".format(function))
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    sys.exit(1)
        
        inputs  = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        des = "Network :"
        for inp in inputs:
            des += "\nInput '{}' with shape {} and dtype {}".format(inp.name, inp.shape, inp.dtype)
        for out in outputs:
            des += "\nOutput '{}' with shape {} and dtype {}".format(out.name, out.shape, out.dtype)
        logger.info(des)
        
        if any(inp.dtype == trt.DataType.HALF for inp in inputs):
            config.set_flag(trt.BuilderFlag.FP16)
        
        if any(any(s == -1 for s in inp.shape) for inp in inputs):
            profile = builder.create_optimization_profile()
            for inp in inputs:
                if any(s == -1 for s in inp.shape):
                    profile.set_shape(inp.name, ** kwargs[inp.name])
            
            config.add_optimization_profile(profile)
        
        engine = builder.build_serialized_network(network, config)
        if engine is not None:
            with open(path, 'wb') as file:
                file.write(engine)
        
        return cls(path)
