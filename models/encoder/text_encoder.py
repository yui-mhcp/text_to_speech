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

from utils import is_path, expand_path
from utils.text import parse_document, chunks_from_paragraphs
from .base_encoder import BaseEncoderModel
from utils.keras_utils import TensorSpec, ops
from utils.search.vectors import build_vectors_db
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

    def predict(self,
                texts,
                batch_size  = 8,
                *,
                
                primary_key = 'text',
                
                chunk_size  = 512,
                chunk_overlap   = 0.25,
                
                save    = True,
                filename    = 'embeddings.h5',
                directory   = None,
                overwrite   = False,
                
                ** kwargs
               ):
        if isinstance(texts, (str, dict)):  texts = [texts]
        elif hasattr(texts, 'to_dict'):     texts = texts.to_dict('records')
        
        paragraphs = []
        for text in texts:
            if not isinstance(text, str):
                paragraphs.append(text)
            elif '*' not in text and not is_path(text):
                paragraphs.append({'text' : text})
            else:
                file = expand_path(text)
                if not isinstance(file, list): file = [file]
                for f in file:
                    try:
                        paragraphs.extend(parse_document(f, ** kwargs))
                    except Exception as e:
                        logger.warning('An exception occured while parsing file {} : {}'.format(f, e))
                
        if chunk_size:
            paragraphs = chunks_from_paragraphs(
                paragraphs, chunk_size, max_overlap_len = chunk_overlap, tokenizer = self.text_encoder, ** kwargs
            )
        
        vectors = None
        if save:
            if directory is None: directory = self.pred_dir
            os.makedirs(directory, exist_ok = True)
            
            filename = os.path.join(directory, filename)
            vectors  = build_vectors_db(filename)
        
        queries = paragraphs
        if vectors is not None and not overwrite:
            paragraphs = [p for p in paragraphs if p['text'] not in vectors]
        
        if paragraphs:
            vectors = self.embed(
                paragraphs,
                batch_size  = batch_size,
                primary_key = primary_key,
                
                reorder = False,
                initial_results = vectors,
                
                ** kwargs
            )
            if save:
                vectors.save(filename, overwrite = True)
                vectors = vectors[queries]

        return vectors
    
    def get_config(self):
        config = super().get_config()
        config.update(self.get_config_text())
            
        return config
