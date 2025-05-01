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
    __extensions__ = 'md'

    def get_text(self, *, remove_hyperlink = True, ** kwargs):
        text = super().get_text(** kwargs)
        return re.sub(_hlinks_re, r'\1', text)
    
    def get_paragraphs(self, ** kwargs):
        """ Extract a list of paragraphs """
        self.text   = ''
        self.code   = None
        self.section_num    = []
        self.section_title  = []
        
        lines = self.get_text(** kwargs).split('\n')
        
        self.paragraphs = []
        for line in lines:
            if line.startswith('```'):
                self._maybe_add_paragraph()
                if self.code:
                    self.code = None
                else:
                    self.code = line[3:].strip() or 'text'
                continue
            elif self.code:
                pass
            elif not line.strip():
                self._maybe_add_paragraph()
                continue
            elif line.startswith('!['): # skip images
                continue
            elif line.startswith('#'):
                self._maybe_add_paragraph()

                title_level = line.split()[0].count('#')
                if title_level <= len(self.section_num):
                    self.section_title = self.section_title[: title_level]
                    self.section_num = self.section_num[: title_level]
                    self.section_num[-1] += 1
                    self.section_title[-1] = line.replace('#', '').strip()
                else:
                    self.section_num.append(1)
                    self.section_title.append(line.replace('#', '').strip())

            if self.text: self.text += '\n'
            self.text += line

        self._maybe_add_paragraph()

        return self.paragraphs

    def _maybe_add_paragraph(self):
        if not self.text: return

        self.paragraphs.append({'text' : self.text.strip(), 'type' : 'text'})
        if self.code:
            self.paragraphs[-1].update({'type' : 'code', 'language' : self.code})

        if self.section_num:
            self.paragraphs[-1].update({
                'section'   : '.'.join([str(num) for num in self.section_num]),
                'section_titles'    : self.section_title
            })

        self.text = ''
