
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
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
import logging

_wiki_cleaner = r'(\[edit\]|\[[0-9]\])'

def parse_html(text,
               level    = 2,
               
               tag_title    = 'h1',
               title_separator  = ' - ',
               p_separator      = '\n\n',
               tags_to_keep     = ['p', 'li'],
               
               skip_header      = False,
               skip_footer      = False,
               skip_aside       = False,
               skip_hrefs       = False,
               links_threshold  = 0.1,
               
               is_sub_document  = False,
               remove_pattern   = None,
               skip_list_section_keywords   = ['content', 'link', 'reference', 'see also', 'navigation', 'bibliography', 'notes', 'source', 'further reading'],
               
               ** kwargs
              ):
    from bs4 import BeautifulSoup

    def clean_parts(parts, separator = '\n\n'):
        if _remove_regex is not None: parts = [re.sub(_remove_regex, lambda m: '', p) for p in parts]
        text = [' '.join([w for w in p.split(' ') if len(w) > 0]) for p in parts]
        return separator.join([p for p in text if len(p) > 0])
    
    def should_skip_list(title_levels, current_text):
        if len(title_levels) <= 1 and not is_sub_document: return True
        for t in title_levels[1:]:
            for pat in skip_list_section_keywords:
                if pat.lower() in t.lower():
                    return True
        return False
        
        
    def normalize_paragraph(title_levels, current_paragraph_idx, paragraph):
        paragraphs = clean_parts(paragraph.get('text', []), p_separator)
        
        formatted = None if len(paragraphs) == 0 else {
            'title'     : clean_parts(title_levels, title_separator),
            'text'      : paragraphs,
            'end_p_idx'   : current_paragraph_idx,
            ** {k : v for k, v in paragraph.items() if k not in ('title', 'text')},
            ** kwargs
        }

        return {'text' : []}, formatted
    
    def _is_list_of_links(tag, text):
        links = tag.find_all('a')
        if not links: return False
        links_text = '\n'.join([l.text for l in links])
        if len(text) - len(links_text) <= links_threshold * len(text):
            logging.debug('Skipping tag {} because it is a list of links ! (text : {})'.format(
                tag.name, text
            ))
            return True
        return False
    
    if not isinstance(tags_to_keep, (list, tuple)): tags_to_keep = [tags_to_keep]
    _remove_regex = None if not remove_pattern else re.compile(remove_pattern)
    
    parser = BeautifulSoup(text)

    to_skip_tags    = []
    if skip_header: to_skip_tags.append('header')
    if skip_footer: to_skip_tags.append('footer')
    if skip_aside:  to_skip_tags.append('aside')
    
    tag_titles = [tag_title] + ['h{}'.format(i+2) for i in range(level)]
    tags = tags_to_keep + tag_titles + to_skip_tags
    
    title_levels = []
    
    to_skip = []
    
    paragraphs = []
    current_paragraph = {'text' : []}
    current_paragraph_idx = 0
    for tag in parser.find_all(tags):
        if tag.name in tag_titles:
            current_paragraph, parsed = normalize_paragraph(title_levels, current_paragraph_idx, current_paragraph)
            if parsed: paragraphs.append(parsed)
            
            level = tag_titles.index(tag.name)
            title_levels = title_levels[:level] + [tag.text]
            
            continue
        
        if any([skip_tag in tag.name for skip_tag in to_skip_tags]):
            _cleaned_text = '\n'.join([w for w in tag.text.split('\n') if len(w) > 0])
            logging.debug('Adding {} for skiping (text {})'.format(tag.name, _cleaned_text))
            to_skip.append(tag)
            continue
        
        if any([tag in skip.find_all(tag.name) for skip in to_skip]):
            _cleaned_text = '\n'.join([w for w in tag.text.split('\n') if len(w) > 0])
            logging.debug('Skipping tag {} with text {}'.format(tag.name, _cleaned_text))
            continue
        
        text = tag.text.strip()
        if skip_hrefs and _is_list_of_links(tag, text):
            continue
        if 'p' in tag.name:
            if text:
                current_paragraph['text'].append(text)
                current_paragraph.setdefault('start_p_idx', current_paragraph_idx)
            current_paragraph_idx += 1
        
        elif 'li' in tag.name and text:
            if should_skip_list(title_levels, current_paragraph['text']): continue
            current_paragraph.setdefault('start_p_idx', current_paragraph_idx)

            if len(current_paragraph['text']) == 0:
                current_paragraph['text'].append('- ' + text)
            else:
                current_paragraph['text'][-1] += '\n- ' + text
        
    
    current_paragraph, parsed = normalize_paragraph(
        title_levels, current_paragraph_idx, current_paragraph
    )
    if parsed: paragraphs.append(parsed)
    
    return paragraphs