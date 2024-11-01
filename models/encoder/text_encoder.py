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

from .base_encoder import BaseEncoderModel
from utils.keras_utils import TensorSpec, ops
from models.interfaces.base_text_model import BaseTextModel

class TextEncoder(BaseTextModel, BaseEncoderModel):
    pad_value   = BaseTextModel.blank_token_idx
    input_signature = BaseTextModel.text_signature
    prepare_input   = BaseTextModel.prepare_text
    
    def __init__(self, lang = 'multi', pretrained = 'BAAI/bge-m3', ** kwargs):
        kwargs.setdefault('text_encoder', pretrained)
        kwargs.setdefault('pretrained_name', pretrained)
        
        self._init_text(lang, ** kwargs)
        
        super().__init__(pretrained = pretrained, ** kwargs)
    
    def build(self, model = None, pretrained = None, ** kwargs):
        if model is None:
            from custom_architectures.transformers_arch import get_pretrained_transformer

            model = kwargs if not pretrained else get_pretrained_transformer(pretrained, ** kwargs)
            
        super().build(model = model)
        
    def __str__(self):
        return super().__str__() + self._str_text()

    def get_config(self):
        config = super().get_config()
        config.update(self.get_config_text())
            
        return config
