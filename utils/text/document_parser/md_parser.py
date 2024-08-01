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

import re

from .parser import parse_document

_hlink_re   = r'\[(.*?)\]\((.*?)\)'

@parse_document.dispatch
def parse_md(filename, encoding = 'utf-8', remove_hyperlink = True, ** kwargs):
    def _maybe_add_paragraph():
        if not text: return
        _text = text.strip()
        if remove_hyperlink:
            _text = re.sub(_hlink_re, r'\1', _text)
        
        paragraphs.append({'text' : _text, 'type' : 'text'})
        if is_code: paragraphs[-1]['type'] = 'code'
        if _section_num:
            paragraphs[-1].update({
                'section'   : '.'.join([str(num) for num in _section_num]),
                'section_titles'    : _section_titles
            })
        return paragraphs[-1]
    
    with open(filename, 'r', encoding = encoding) as f:
        lines = f.read().split('\n')
    
    text    = ''
    is_code = False
    paragraphs  = []
    _section_num, _section_titles = [], []
    for line in lines:
        if line.startswith('```'):
            _maybe_add_paragraph()
            text    = ''
            is_code = not is_code
            continue
        elif is_code:
            pass
        elif not line.strip():
            _maybe_add_paragraph()
            text    = ''
            continue
        elif line.startswith('!['): # skip images
            continue
        elif line.startswith('#'):
            _maybe_add_paragraph()
            text    = ''
            
            title_level = line.split()[0].count('#')
            if title_level <= len(_section_num):
                _section_titles = _section_titles[: title_level]
                _section_num = _section_num[: title_level]
                _section_num[-1] += 1
                _section_titles[-1] = line.replace('#', '').strip()
            else:
                _section_num.append(1)
                _section_titles.append(line.replace('#', '').strip())
        
        if text: text += '\n'
        text += line
    
    _maybe_add_paragraph()
    
    return paragraphs
