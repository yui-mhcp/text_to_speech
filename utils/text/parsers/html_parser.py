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
import logging

from loggers import Timer, timer
from .parser import Parser

logger = logging.getLogger(__name__)

_links  = {}

_link_re    = r'<a [^>]*?href\=([^>\s]+)[^>]*?>(.*?)</a>'
_title_re   = r'\<title\>(.*?)\<\/title\>'
_whitespace_re = re.compile(r'\s+')

class HTMLParser(Parser):
    __extension__ = 'html'

    def __init__(self, filename = None, html = None, ** _):
        """
        Initialize HTML parser
        
        Args:
            html_content: HTML string to parse
            url: Optional URL for context (used as filename)
        """
        assert filename or html
        
        super().__init__(filename or "html_content")
        
        if not html:
            with open(filename, 'r', encoding = 'utf-8') as f:
                html = f.read()
        
        self.html = html

    def get_paragraphs(self, html = None, ** kwargs):
        title, html = prepare_html(self.html, ** kwargs)

        return extract_paragraphs(html, title = title or 'html', ** kwargs)

def extract_title(html):
    match = re.search(_title_re, html, flags = re.DOTALL)
    return match.group(1).strip() if match is not None else None

@timer
def prepare_html(html,
                 *,
                 
                 skip_header    = True,
                 skip_footer    = True,
                 skip_aside     = True,
                 skip_nav       = True,
                 skip_table     = False,
                 skip_form      = True,
                 skip_link_item = True,
                 
                 origin = None,
                 
                 simplify   = True,
                 
                 ** _
                ):
    tags = ['head', 'script', 'style']
    if skip_table:  tags.append('table')
    if skip_aside:  tags.append('aside')
    if skip_form:   tags.append('form')
    if skip_nav:    tags.append('nav')
    
    title = extract_title(html)
    html  = _remove_tags(html, tags)
    if skip_header: html = _remove_tags(html, ['header'], mode = 'first')
    if skip_footer: html = _remove_tags(html, ['footer'], mode = 'last')
    if skip_link_item:  html = _remove_link_items(html)
    if simplify:        html = _simplify_html(html)
    
    if origin:
        html = re.sub(r'\[\d+\]', '', html)
        html = re.sub(_link_re, lambda m: _add_link_ref(m, origin), html, flags = re.DOTALL)
    
    html = html.replace('</', ' </')
    
    return title, html

def _add_link_ref(match, origin):
    link = match.group(1).strip('"')
    if not link.startswith('http'): link = origin + link
    
    if link not in _links:
        _links[link] = len(_links) + 1
    
    text = match.group(2)
    if '<' not in text:
        return '<p>{} [{}]</p>'.format(text, _links[link])
    elif '</h' in text:
        return text.replace('</h', ' [{}] </h'.format(_links[link]), 1)
    elif '</p' in text:
        return text.replace('</p', ' [{}] </p'.format(_links[link]), 1)
    else:
        return text + ' [{}] '.format(_links[link])

def get_link(link_id):
    for link, num in _links.items():
        if num == link_id:
            return link
    return None

@timer
def extract_paragraphs(html, *, title = 'html', skip_table = False, ** _):
    from bs4 import BeautifulSoup
    
    tags = ['p', 'ul', 'ol', 'h1', 'h2', 'h3', 'h4', 'h5']
    if not skip_table: tags.append('table')
    
    with Timer('html parsing'):
        soup = BeautifulSoup(html, 'lxml')
    
    with Timer('find tags'):
        tags = soup.find_all(tags)
    
    titles = []
    parsed = []
    with Timer('tags processing'):
        for tag in tags:
            if tag.decomposed:
                continue
            elif tag.name == 'table':
                rows = _parse_table(tag)
                if rows and rows[0]:
                    parsed.append({'type' : 'table', 'section' : titles, 'rows' : rows})
            elif tag.name in ('ul', 'ol'):
                items = _parse_list(tag)
                if items:
                    parsed.append({'type' : 'list', 'section' : titles, 'items' : items})
            elif tag.name[0] == 'h' and tag.name[1].isdigit():
                titles = _parse_title(tag, titles)
            elif tag.name == 'code':
                text = _extract_text(tag)
                if text: parsed.append({'type' : 'code', 'section' : titles, 'text' : text})
            else:
                text = _extract_text(tag)
                if text: parsed.append({'type' : 'text', 'section' : titles, 'text' : text})

            tag.decompose()
    
    if title:
        for para in parsed: para['title'] = title
    
    return parsed

def _remove_tags(html, tags, mode = 'all'):
    pattern = r'<({})\b[^>]*>.*?</\1>'.format('|'.join(tags))
    
    if mode == 'all':
        return re.sub(pattern, '', html, flags = re.DOTALL | re.IGNORECASE)
    elif mode == 'first':
        return re.sub(pattern, '', html, count = 1, flags = re.DOTALL | re.IGNORECASE)
    elif mode == 'last':
        matches = list(re.finditer(pattern, html, flags = re.DOTALL | re.IGNORECASE))
        if matches:
            return html[: matches[-1].start()] + html[matches[-1].end() :]
        return html

def _remove_link_items(html):
    return re.sub(r'<li\b[^>]*?>\s*<a\b[^>]*?>.*?</a>\s*</li>', '', html, flags = re.DOTALL)

def _simplify_html(html):
    html = re.sub(f'</?(?:div|span)[^>]*?>', '', html)
    html = re.sub(r'<[^>]+?/>', '', html)
    return html

@timer
def _parse_table(tag):
    row_tags = tag.find_all('tr')
    columns = [
        t.get_text().strip() for t in row_tags[0].find_all('td')
    ]
    
    rows = []
    for row_tag in row_tags[1:]:
        rows.append({
            col : _extract_text(t) for col, t in zip(columns, row_tag.find_all('td'))
        })
    
    if len(rows) > 1:
        for col in columns:
            ref = rows[0].get(col, None)
            if all(row.get(col, None) == ref for row in rows[1:]):
                for row in rows: row.pop(col, None)
    
    return rows

@timer
def _parse_list(tag):
    items = [_extract_text(t) for t in tag.find_all('li')]
    return [it for it in items if it]

def _parse_title(tag, titles):
    level = int(tag.name[1]) - 1
    titles = titles[:level]
    if len(titles) != level: titles.extend([''] * (level - len(titles)))
    titles.append(_extract_text(tag))
    return titles

@timer
def _extract_text(tag):
    return re.sub(_whitespace_re, ' ', tag.get_text().strip())
