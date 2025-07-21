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

from abc import ABC, abstractmethod

class Parser(ABC):
    def __init__(self, filename):
        self.filename = filename
    
    def __repr__(self):
        return '<{} file={}>'.format(self.__class__.__name__, self.filename)
    
    @abstractmethod
    def get_paragraphs(self, ** kwargs):
        """
            Extract a list of paragraphs from `self.filename`
            
            A paragraph is a `dict` with (at least) these entries :
                - type  : the paragraph type (e.g., 'text', 'link', 'image', ...)
                - text  : the paragraph text content (only if relevant)
                - filename / width / heigth : the image information (only if "type == image")
        """
    
    def get_text(self, sep = '\n\n', ** kwargs):
        """
            Return raw text from the entire document.
            This may be useful if the parser has a fastest way to extract text-only content.
        """
        text = sep.join([para['text'] for para in self.get_paragraphs(** kwargs) if 'text' in para])
        return [{'text' : text}]
    