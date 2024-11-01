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
import inspect
import logging
import collections

from functools import cache

from loggers import timer
from utils import time_to_string
from utils.keras_utils import ops

TRTLLMInferenceOutput = collections.namedtuple(
    "TRTLLMInferenceOutput", [
        "tokens", "lengths", "offset"
    ]
)

logger = logging.getLogger(__name__)

@cache
def load_trt_engine(directory, use_cpp = True, kv_cache_free_gpu_memory_fraction = 0.3, ** kwargs):
    from tensorrt_llm.runtime import ModelRunnerCpp, ModelRunner
    
    if use_cpp: kwargs['kv_cache_free_gpu_memory_fraction'] = kv_cache_free_gpu_memory_fraction
    runner = ModelRunnerCpp if use_cpp else ModelRunner
    return runner.from_dir(
        engine_dir = directory,
        ** {k : v for k, v in kwargs.items() if k in inspect.signature(runner.from_dir).parameters}
    )


class TRTLLMRunner:
    def __init__(self, directory, runner = None, ** kwargs):
        self.directory  = directory
        
        self.runner = load_trt_engine(directory, ** kwargs) if runner is None else runner
        
        self.infer_signature    = inspect.signature(self.runner.generate).parameters.keys()
        
        self.sos_token  = -1
        self.eos_token  = -1
        self.pad_token  = -1
    
    def set_tokens(self, sos_token = None, eos_token = None, pad_token = None):
        if sos_token not in (-1, None): self.sos_token = sos_token
        if eos_token not in (-1, None): self.eos_token = eos_token
        if pad_token not in (-1, None): self.pad_token = pad_token

    def __repr__(self):
        return '<TRT-LLM engine_dir={}>'.format(self.directory)
    
    @timer(name = 'TRT-LLM inference')
    def __call__(self, inputs, max_input_len = None, stream_callback = None, ** kwargs):
        if max_input_len: self.runner.max_input_len = max_input_len
        
        if not isinstance(inputs, list):
            if not ops.is_torch_tensor(inputs):
                inputs = ops.convert_to_torch_tensor(inputs)

            if len(inputs.shape) == 2:
                if len(inputs) == 1:
                    inputs = [inputs[0]]
                else:
                    inputs = [
                        inp[:length] for length in (inputs != self.pad_token).count_nonzero(1)
                    ]
            else:
                inputs = [inputs]
            
        elif isinstance(inputs[0], int):
            inputs = [ops.convert_to_torch_tensor(inputs)]
        else:
            inputs = [ops.convert_to_torch_tensor(inp) for inp in inputs]

        if 'kwargs' not in self.infer_signature:
            kwargs = {k : v for k, v in kwargs.items() if k in self.infer_signature}
        else:
            kwargs = {k : v for k, v in kwargs.items() if 'prompt' not in k and 'format' not in k}
        kwargs.update({
            'end_id'    : self.eos_token,
            'pad_id'    : self.pad_token,
            'return_dict'   : True,
            'output_sequence_lengths'   : True
        })
        if stream_callback: kwargs['streaming'] = True
        
        inp_lengths = [inp.size(0) for inp in inputs]

        t0 = time.time()
        output = self.runner.generate(inputs, ** kwargs)
        
        if kwargs.get('streaming', False):
            stream = output
            for output in stream:
                stream_callback(TRTLLMInferenceOutput(
                    tokens  = output['output_ids'],
                    lengths = output['sequence_lengths'],
                    offset  = inp_lengths
                ))
        
        if logger.isEnabledFor(logging.INFO):
            n = output['sequence_lengths'].sum().cpu().numpy() - sum(inp_lengths)
            t1 = time.time()
            logger.info('{} tokens generated in {} ({:.3f} tokens/sec)'.format(
                n, time_to_string(t1 - t0), n / (t1 - t0)
            ))
        
        return TRTLLMInferenceOutput(
            tokens  = output['output_ids'],
            lengths = output['sequence_lengths'],
            offset  = inp_lengths
        )
