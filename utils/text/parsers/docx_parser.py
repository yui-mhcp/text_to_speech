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

from .parser import Parser

class DocxParser(Parser):
    __extensions__ = 'docx'
    
    def get_paragraphs(self, ** kwargs):
        """ Extract a list of paragraphs """
        from docx import Document
        
        return [{'text' : p.text} for p in Document(self.filename).paragraphs]

