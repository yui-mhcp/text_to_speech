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
import glob
import logging
import numpy as np

from .. import ops, timer
from .runtime import Runtime

logger = logging.getLogger(__name__)

class TensorRTLLMBertRuntime(Runtime):
    def __init__(self, path, ** kwargs):
        import torch
        import tensorrt as trt
        
        self.stream = torch.cuda.Stream().cuda_stream
        
        super().__init__(path, ** kwargs)

        self._dtype_mapping   = {
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
        
        self.bindings   = list(self.engine.engine)
        self._inputs    = tuple(
            n for n in self.bindings if self.engine.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self._outputs   = tuple(
            n for n in self.bindings if self.engine.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        )
        self._dtypes    = tuple(
            self._dtype_mapping[self.engine.engine.get_tensor_dtype(n)] for n in self.bindings
        )
        self._shapes    = tuple(
            self.engine.engine.get_tensor_shape(n) for n in self.bindings
        )
        self._var_shapes    = set([
            n for n in self._inputs if any(s == -1 for s in self.engine.engine.get_tensor_shape(n))
        ])
        
        self.sos_token  = -1
        self.eos_token  = -1
        self.pad_token  = -1
    
    def set_tokens(self, sos_token = None, eos_token = None, pad_token = None):
        if sos_token not in (-1, None): self.sos_token = sos_token
        if eos_token not in (-1, None): self.eos_token = eos_token
        if pad_token not in (-1, None): self.pad_token = pad_token
    
    @property
    def remove_input_padding(self):
        return len(self.shapes[0]) == 1
    
    @property
    def embedding_dim(self):
        return self.shapes[-1][-1]
    
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
        return '<TensorRTLLMBertRuntime path={} inputs={} outputs={}>'.format(
            os.path.basename(self.path), self.argnames, self.outputs
        )
    
    @timer(name = 'TensorRTLLM runtime inference')
    def __call__(self, inputs, ** _):
        import torch
        import tensorrt as trt
        
        from tensorrt_llm.runtime import TensorInfo
        
        _is_batch = _is_batched(inputs)
        if not _is_batch: inputs = [inputs]
        
        if self.remove_input_padding:
            tokens, lengths, pos_ids = [], [], []
            for inp in inputs:
                tokens.extend(inp)
                lengths.append(len(inp))
                pos_ids.extend(range(2, len(inp) + 2))
            
            tokens = torch.tensor(np.array(tokens), dtype = torch.int32, device = self.device)
        else:
            if len(inputs) > 1:
                tokens = [
                    torch.tensor(np.array(tok), dtype = torch.int32, device = 'cpu')
                    for tok in inputs
                ]
                tokens = torch.nn.utils.rnn.pad_sequence(
                    tokens, batch_first = True, padding_value = self.pad_token
                ).to(device = self.device)
            else:
                tokens = torch.tensor(np.array(inputs), dtype = torch.int32, device = self.device)
            lengths = [len(inp) for inp in inputs]
            pos_ids = np.array([np.arange(2, tokens.shape[1] + 2)] * len(inputs))
            
        
        inputs  = {
            'input_ids' : tokens,
            'input_lengths' : torch.tensor(np.array(lengths), dtype = torch.int32, device = self.device),
            'token_type_ids'    : torch.zeros_like(tokens)
        }
        if self.remove_input_padding:
            inputs.update({
                'position_ids'  : torch.tensor(pos_ids, dtype = torch.int32, device = self.device),
                'max_input_length'  : torch.tensor(
                    [max(lengths)], dtype = torch.int32, device = self.device
                )
            })

        output_infos = self.engine.infer_shapes([
            TensorInfo(k, trt.DataType.INT32, v.shape) for k, v in inputs.items()
        ])
        outputs = {
            i.name : torch.zeros(
                * i.shape, dtype = getattr(torch, self._dtype_mapping[i.dtype]), device = self.device
            )
            for i in output_infos
        }

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Calling TensorRTLLM engine with :\n  Inputs : {}\n  Outputs : {}'.format(
                {name : tuple(v.shape) for name, v in inputs.items()},
                {name : tuple(v.shape) for name, v in outputs.items()}
            ))

        if not self.engine.run(inputs, outputs, self.stream):
            raise RuntimeError('An exception occured while running the TensorRT context')

        if _is_batch and self.remove_input_padding:
            outputs = {k : list(torch.split(v, lengths)) for k, v in outputs.items()}

        torch.cuda.synchronize()
        
        return outputs if len(outputs) > 1 else list(outputs.values())[0]
    
    @staticmethod
    def prepare_tensor(tensor, device):
        return ops.convert_to_torch_tensor(tensor, dtype = 'int32').to(device = device)
    
    @staticmethod
    def load_engine(path, ** _):
        from tensorrt_llm.runtime import Session
        
        engine_path = glob.glob(os.path.join(path, '*.engine'))[0]
        
        logger.info('Loading engine from {}'.format(engine_path))
        
        with open(engine_path, 'rb') as f:
            engine_buffer = f.read()
        
        session = Session.from_serialized_engine(engine_buffer)
        
        return session

def _is_batched(inputs):
    return isinstance(inputs, list) and len(ops.shape(inputs[0])) == 1
