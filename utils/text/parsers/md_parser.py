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

import re

from .txt_parser import TxtParser

_hlinks_re   = r'\[(.*?)\]\((.*?)\)'
         
class MarkdownParser(TxtParser):
    __extension__ = 'md'

    def get_text(self, *, remove_hyperlink = True, ** kwargs):
        text = super().get_text(** kwargs)
        return re.sub(_hlinks_re, r'\1', text)
    
    def get_paragraphs(self, ** kwargs):
        """ Extract a list of paragraphs """
        if hasattr(self, 'paragraphs'): return self.paragraphs
        
        lines = self.get_text(** kwargs).split('\n')

        self.paragraphs = []
        text, code_type, section = '', None, []
        for line in lines:
            if line.startswith('```'):
                text = self._maybe_add_paragraph(text, section, code_type)
                if code_type: # end of code block
                    code_type = None
                else:
                    code_type = line[3:].strip() or 'text'
                continue
            elif code_type:
                pass
            elif not line.strip():
                text = self._maybe_add_paragraph(text, section, code_type)
                continue
            elif line.startswith('!['): # skip images
                continue
            elif line.startswith('#'):
                text = self._maybe_add_paragraph(text, section, code_type)

                prefix, _, title = line.partition(' ')
                section = section[: len(prefix) - 1] + [title]

            if text: text += '\n'
            text += line

        self._maybe_add_paragraph(text, section, code_type)

        return self.paragraphs

    def _maybe_add_paragraph(self, text, section, code_type = None):
        if text:
            paragraph = {'type' : 'text', 'text' : text.strip()}
            if section:     paragraph['section'] = section
            if code_type:   paragraph.update({'type' : 'code', 'language' : code_type})
        
            self.paragraphs.append(paragraph)
        
        return ''
