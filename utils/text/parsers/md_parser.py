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

_audio_ext  = ('wav', 'mp3', 'flac', 'opus', 'ogg')
_image_ext  = ('gif', 'png', 'jpeg', 'jpg')
_video_ext  = ('mp4', 'mov', 'ovg', 'avi')

_hlinks_re  = r'\[(.*?)\]\((.*?)\)'
         
class MarkdownParser(TxtParser):
    __extension__ = 'md'

    def get_text(self, *, remove_hyperlink = True, ** kwargs):
        text = super().get_text(** kwargs)
        return re.sub(_hlinks_re, r'\1', text) if remove_hyperlink else False
    
    def get_paragraphs(self, ** kwargs):
        """ Extract a list of paragraphs """
        if hasattr(self, 'paragraphs'): return self.paragraphs
        
        with open(self.filename, 'r', encoding = 'utf-8') as f:
            lines = [l.strip() for l in f]

        self.paragraphs = []
        text, code_type, section = '', None, []
        for line in lines:
            if not line:
                text = self._maybe_add_paragraph(text, section, code_type)
                continue
            if line.startswith('```'):
                text = self._maybe_add_paragraph(text, section, code_type)
                if code_type: # end of code block
                    code_type = None
                else:
                    code_type = line[3:].strip() or 'text'
                continue
            elif code_type:
                pass
            elif line.startswith('!['): # skip images
                text = self._maybe_add_paragraph(text, section, code_type)
                text = self._maybe_add_paragraph(None, section, data = line[2:].split(']')[0])
                continue
            elif line.startswith('#'):
                text = self._maybe_add_paragraph(text, section, code_type)

                prefix, _, title = line.partition(' ')
                section = section[: len(prefix) - 1] + [title]

            if text: text += '\n'
            text += line

        self._maybe_add_paragraph(text, section, code_type)

        return self.paragraphs

    def _maybe_add_paragraph(self, text, section, code_type = None, data = None):
        paragraph = {}
        if text:
            paragraph = {'type' : 'text', 'text' : text.strip()}
            if section:     paragraph['section'] = section
            if code_type:   paragraph.update({'type' : 'code', 'language' : code_type})
        elif data:
            if data.endswith(_image_ext):
                paragraph = {'type' : 'image', 'image' : data, 'section' : section}
            if data.endswith(_audio_ext):
                paragraph = {'type' : 'audio', 'audio' : data, 'section' : section}
            if data.endswith(_video_ext):
                paragraph = {'type' : 'video', 'video' : data, 'section' : section}
            else:
                warnings.warn('Unknown file type : {}'.format(data))
        
        if paragraph:
            self.paragraphs.append(paragraph)
        
        return ''
