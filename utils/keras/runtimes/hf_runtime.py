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

from .. import ops, timer
from .runtime import Runtime

class HFRuntime(Runtime):
    @timer(name = 'Transformers runtime inference')
    def __call__(self, * args, ** kwargs):
        import torch
        
        from transformers import XLMRobertaModel
        
        with torch.no_grad():
            args = [
                ops.convert_to_torch_tensor(arg, 'int32').to(device = self.engine.device)
                for arg in args
            ]
            out = self.engine(* args, ** kwargs)
            if isinstance(self.engine, XLMRobertaModel):
                out = out[0][:, 0]
                out = out / torch.functional.norm(out, dim = -1, keepdim = True)
            
            return out
    
    @staticmethod
    def load_engine(path, *, torch_dtype = 'float16', ** _):
        from transformers import AutoModel
        
        return AutoModel.from_pretrained(path, device_map = 'cuda', torch_dtype = torch_dtype)

