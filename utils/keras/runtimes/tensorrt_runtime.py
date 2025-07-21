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

from math import prod

from .. import Timer, ops, timer
from .runtime import Runtime
from .onnx_runtime import ONNXRuntime

logger = logging.getLogger(__name__)

class TensorRTRuntime(Runtime):
    def __init__(self, path, *, prepare = None, ** kwargs):
        import tensorrt as trt
        import pycuda.driver as cuda

        cuda.init()
        self.cuda_ctx   = cuda.Device(0).make_context()

        super().__init__(path, ** kwargs)
        
        self.context    = self.engine.create_execution_context()
        self.stream     = cuda.Stream()

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
        
        self.num_prepare_args   = None
        
        if not prepare and os.path.exists(os.path.splitext(self.path)[0] + '-prepare.pth'):
            prepare = os.path.splitext(self.path)[0] + '-prepare.pth'
        
        if isinstance(prepare, str):
            import torch
            
            prepare = torch.jit.load(prepare, map_location = torch.device('cuda'))

        self.prepare    = prepare
        self.num_prepare_args   = None if not prepare else len(prepare.forward.schema.arguments) - 1
        if prepare is not None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
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
        
        self._inp_shapes    = {}
        self._out_shapes    = {}
        self._cpu_buffers   = {}
        self._gpu_buffers   = {}
    
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
        """Inférence avec gestion du contexte CUDA"""
        import pycuda.driver as cuda
        # Activer le contexte CUDA
        kwargs.update({name : arg for name, arg in zip(self.bindings, args)})
        
        if self.prepare is not None:
            import torch
            
            with Timer('pre-processing'):
                with torch.no_grad():
                    inputs = [
                        self.prepare_tensor(kwargs[k], self.shapes[i], self.dtypes[i], self.device, torch = True)
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
        

        self.cuda_ctx.push()

        inp_shapes = {k : getattr(kwargs[k], 'shape', ()) for k in self.argnames}
        if inp_shapes != self._inp_shapes:
            with Timer('Memory allocation'):
                self.cuda_ctx.push()

                for buf in self._gpu_buffers.values(): buf.free()

                self._inp_shapes    = inp_shapes
                self._out_shapes    = {}
                self._cpu_buffers   = {}
                self._gpu_buffers   = {}

                try:
                    for i, name in enumerate(self.bindings):
                        if name in kwargs:
                            kwargs[name] = inp = self.prepare_tensor(
                                kwargs[name], self.shapes[i], dtype = self.dtypes[i]
                            )
                            shape = inp.shape

                            if name in self._var_shapes and i < len(self.argnames):
                                self.context.set_input_shape(name, inp.shape)

                        elif i >= len(self.argnames): # output tensor
                            shape = tuple(self.context.get_tensor_shape(name))
                            self._out_shapes[name] = shape
                        else:
                            raise RuntimeError('Missing input #{} : {}'.format(i, name))

                        self._cpu_buffers[name] = cuda.pagelocked_empty(prod(shape), self.dtypes[i])
                        self._gpu_buffers[name] = cuda.mem_alloc(self._cpu_buffers[name].nbytes)

                    for name in self.bindings:
                        self.context.set_tensor_address(name, int(self._gpu_buffers[name]))
                finally:
                    self.cuda_ctx.pop()
        else:
            for i, name in enumerate(self._inputs):
                kwargs[name] = self.prepare_tensor(
                    kwargs[name], self.shapes[i], dtype = self.dtypes[i]
                )

        if False and logger.isEnabledFor(logging.DEBUG):
            logger.debug('Calling TensorRT engine with :\n  Inputs : {}\n  Outputs : {}'.format(
                {name : tuple(tensors[i].shape) for i, name in enumerate(self.argnames)},
                {name : tuple(tensors[i].shape) for i, name in enumerate(self.outputs, start = len(self.argnames))},
            ))

        try:
            with Timer('execution'):
                for name in self._inputs:
                    np.copyto(self._cpu_buffers[name], kwargs[name].ravel())
                    cuda.memcpy_htod_async(
                        self._gpu_buffers[name],  self._cpu_buffers[name],  self.stream
                    )

                if not self.context.execute_async_v3(stream_handle = self.stream.handle):
                    raise RuntimeError("TensorRT inference failed.")

                for name in self._outputs:
                    cuda.memcpy_dtoh_async(
                        self._cpu_buffers[name],  self._gpu_buffers[name],  self.stream
                    )

                self.stream.synchronize()

                outputs = [
                    self._cpu_buffers[name].reshape(self._out_shapes[name]) for name in self._outputs
                ]
        finally:
            self.cuda_ctx.pop()
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    @staticmethod
    def prepare_tensor(tensor, shape, dtype, device = None, torch = False):
        if torch:
            tensor = ops.convert_to_torch_tensor(tensor, dtype = dtype)
        else:
            tensor = ops.convert_to_numpy(tensor, dtype = dtype)
        
        if len(tensor.shape) < len(shape): tensor = tensor[None]
        
        if TensorRTRuntime.should_permute(tensor, shape):
            last = len(tensor.shape) - 1
            if torch:
                tensor = tensor.permute(0, last, * range(1, last))
            else:
                tensor = tensor.transpose([0, last] + list(range(1, last)))
        
        if torch:
            tensor = tensor.to(device = device)
        elif not tensor.flags['C_CONTIGUOUS'] or not tensor.flags['WRITEABLE']:
            tensor = np.ascontiguousarray(tensor)

        return tensor

    @staticmethod
    def should_permute(tensor, shape):
        return len(shape) > 2 and shape[1] != -1 and tensor.shape[1] != shape[1]
    
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
