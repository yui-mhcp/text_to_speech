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
import logging

from functools import lru_cache

from loggers import timer, time_logger
from .parser import parse_document

logger = logging.getLogger(__name__)

_wiki_cleaner   = r'\[(edit|[0-9]+|source)\]'
_skip_list_from_sections    = {'content', 'link', 'reference', 'see also', 'navigation', 'bibliography', 'notes', 'source', 'further reading'}

@parse_document.dispatch
@timer
def parse_html(html,
               
               tags = ['p', 'ul'],
               tag_title    = 'h1',
               max_title_level  = 3,
               
               skip_header  = True,
               skip_footer  = True,
               skip_aside   = True,
               skip_hrefs   = True,
               
               remove_pattern   = None,
               skip_list_from_sections  = _skip_list_from_sections,
               
               ** kwargs
              ):
    from bs4 import BeautifulSoup
    
    if tags and not isinstance(tags, (list, tuple, set)): tags = [tags]
    skip_list_from_sections = tuple(set(sec.lower() for sec in skip_list_from_sections))
    _remove_regex = None if not remove_pattern else re.compile(remove_pattern)
    
    for tag in ('script', 'style'):
        html = _remove_tag(html, tag)
    if skip_header: html = _remove_tag(html, 'header')
    if skip_footer: html = _remove_tag(html, 'footer')
    if skip_aside:  html = _remove_tag(html, 'aside')

    with time_logger.timer('parsing'):
        parser = BeautifulSoup(html, features = 'lxml')
    
    tags_title = [tag_title] + ['h{}'.format(i) for i in range(2, max_title_level + 1)]
    
    tags_to_find    = list(tags) + tags_title
    with time_logger.timer('find tags'):
        tags_parsed = parser.find_all(tags_to_find)

    paragraphs = []
    _section_num, _section_titles = [], []
    for tag in tags_parsed:
        if not tag.text.strip(): continue
        
        if tag.name in tags_title:
            level = tags_title.index(tag.name) + 1
            if level <= len(_section_num):
                _section_num, _section_titles = _section_num[:level], _section_titles[:level]
                _section_num[-1] += 1
                _section_titles[-1] = tag.text
            else:
                _section_num.append(1)
                _section_titles.append(tag.text)

        elif 'ul' in tag.name:
            if skip_hrefs and _is_link(tag, threshold = 0.25): continue
            if _should_skip_list(skip_list_from_sections, tuple(_section_titles)): continue
            
            _maybe_add_paragraph(
                paragraphs,
                '\n'.join(['- ' + it.text.strip() for it in tag.children]),
                _section_titles,
                _section_num,
                type = 'list'
            )
        else:
            if skip_hrefs and _is_link(tag, 0.25): continue
            
            _maybe_add_paragraph(
                paragraphs, tag.text, _section_titles, _section_num
            )
    
    if _remove_regex is not None:
        for p in paragraphs:
            p['text'] = re.sub(_remove_regex, '', p['text'])
        
    return paragraphs

@timer
def _remove_tag(html, tag):
    return re.sub('<{tag}.*?>.*?</{tag}>'.format(tag = tag), '', html, flags = re.DOTALL)

def _maybe_add_paragraph(paragraphs, text, section_titles, section_num, ** kwargs):
    if isinstance(text, list): text = '\n'.join(text)
    
    text = text.strip()
    if not text: return None, []

    paragraphs.append({'text' : text, 'type' : 'text', ** kwargs})
    if section_num:
        paragraphs[-1].update({
            'section'   : '.'.join([str(num) for num in section_num]),
            'section_titles'    : section_titles.copy()
        })
    return paragraphs[-1], []

@timer
def _is_link(tag, threshold = 0.25):
    links = tag.find_all('a')
    if not links: return False
    
    links_text = ''.join([l.text for l in links])
    if len(links_text) / len(tag.text) >= threshold:
        logger.debug('Skipping tag {} because it is a list of links ! (text : {})'.format(
            tag.name, tag.text
        ))
        return True
    return False

@timer
@lru_cache(maxsize = 5)
def _should_skip_list(_skip_list_from_sections, titles):
    for title in titles:
        title = title.lower()
        if any(skip in title for skip in _skip_list_from_sections):
            return True
    return False

