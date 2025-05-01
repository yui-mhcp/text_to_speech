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
import json

from .runtime import Runtime
from .hf_runtime import HFRuntime
from .onnx_runtime import ONNXRuntime
from .tensorrt_runtime import TensorRTRuntime
from .saved_model_runtime import SavedModelRuntime
from .tensorrt_llm_runtime import TensorRTLLMRuntime
from .tensorrt_llm_bert_runtime import TensorRTLLMBertRuntime

def build_runtime(runtime, path, * args, ** kwargs):
    if runtime not in _runtimes:
        raise ValueError('Unsupported runtime !\n  Accepted : {}\n  Got : {}'.format(
            tuple(_runtimes.keys()), runtime
        ))
    
    if runtime == 'trt_llm' and os.path.exists(os.path.join(path, 'config.json')):
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        if 'bert' in config['pretrained_config']['architecture'].lower():
            return TensorRTLLMBertRuntime(path, * args, ** kwargs)
    
    return _runtimes[runtime](path, * args, ** kwargs)
    
    
_runtimes    = {
    'hf'    : HFRuntime,
    'onnx'  : ONNXRuntime,
    'trt'   : TensorRTRuntime,
    'trt_llm'   : TensorRTLLMRuntime,
    'saved_model'   : SavedModelRuntime
}