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
import warnings

from .parser import register_parser
from .txt_parser import read_txt

_audio_ext  = ('wav', 'mp3', 'flac', 'opus', 'ogg')
_image_ext  = ('gif', 'png', 'jpeg', 'jpg')
_video_ext  = ('mp4', 'mov', 'ovg', 'avi')

_hlinks_re  = r'\[(.*?)\]\((.*?)\)'
_image_re   = re.compile(r'!\[(.*?)\]\((.*?)\)')

@register_parser('md', kind = 'text')
def read_md(filename, *, remove_hyperlink = True, ** kwargs):
    """ Return the raw markdown text, optionally stripping hyperlink targets """
    text = read_txt(filename, ** kwargs)
    return re.sub(_hlinks_re, r'\1', text) if remove_hyperlink else text

@register_parser('md')
def parse_md(filename, ** kwargs):
    """ Extract a list of paragraphs (text / code / image / audio / video) from a markdown file """
    with open(filename, 'r', encoding = 'utf-8') as f:
        lines = [l.strip() for l in f]

    paragraphs = []
    def add(text, section, code_type = None, data = None):
        para = _make_paragraph(text, section, code_type, data)
        if para: paragraphs.append(para)
        return ''

    text, code_type, section = '', None, []
    for line in lines:
        if not line:
            text = add(text, section, code_type)
            continue
        if line.startswith('```'):
            text = add(text, section, code_type)
            if code_type: # end of code block
                code_type = None
            else:
                code_type = line[3:].strip() or 'text'
            continue
        elif code_type:
            pass
        elif line.startswith('!['): # image / media
            text = add(text, section, code_type)
            match = _image_re.match(line)
            data  = match.group(2) if match else line[2:].split(']')[0]
            add(None, section, data = data)
            continue
        elif line.startswith('#'):
            text = add(text, section, code_type)

            prefix, _, title = line.partition(' ')
            section = section[: len(prefix) - 1] + [title]

        if text: text += '\n'
        text += line

    add(text, section, code_type)

    return paragraphs

def _make_paragraph(text, section, code_type = None, data = None):
    if text:
        paragraph = {'type' : 'text', 'text' : text.strip()}
        if section:     paragraph['section'] = section
        if code_type:   paragraph.update({'type' : 'code', 'language' : code_type})
        return paragraph
    elif data:
        if data.endswith(_image_ext):
            return {'type' : 'image', 'image' : data, 'section' : section}
        elif data.endswith(_audio_ext):
            return {'type' : 'audio', 'audio' : data, 'section' : section}
        elif data.endswith(_video_ext):
            return {'type' : 'video', 'video' : data, 'section' : section}
        elif data.endswith(('svg', 'md', 'txt')):
            pass
        else:
            warnings.warn('Unknown file type : {}'.format(data))
    return None
