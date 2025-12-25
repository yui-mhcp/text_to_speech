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

import logging
import warnings
import collections

from copy import deepcopy
from functools import cache

from .cleaners import remove_urls, remove_files
from .text_processing import format_text, split_text

logger  = logging.getLogger(__name__)

_multimodal_types = ('audio', 'image', 'video')

def chunks_from_paragraphs(paragraphs,
                           max_length   = None,
                           *,
                           
                           group_by = None,
                           separator    = '\n\n',
                           
                           max_overlap  = 5,
                           max_overlap_len  = 0.2,
                           
                           ** kwargs
                          ):
    """
        Creates chunks from the given `paragraphs` by splitting then merging them
        
        Arguments :
            - paragraphs    : a list of paragraph (i.e., `dict` containing at least `text` entry)
            - max_length    : maximum length for a given chunk
            
            - group_by  : controls which paragraphs to merge together
                          This allows to only merge paragraph with the same section or filename
                          The value should be a (list of) paragraph's entries to use to group them
            
            - max_overlap[_len] : forwarded to `merge_texts`
            
            - tokenizer : forwarded to `split_text` and `merge_texts`
            - kwargs    : forwarded to `split_text` and `merge_texts`
        Return :
            - chunks    : a list of splitted/merged paragraphs
        
        Note : in order to enforce overlaps, the paragraphs are splitted with `max_length = max_overlap_len / max_overlap` with a sentence tolerance of `max_length`. This means that a paragraph is splitted into sentences of at most `max_overlap_len / max_overlap`, but a single sentence is only splitted if it is longer than `max_length`.
        
        Here is a comprehensive example of this procedure :
            Inputs :
            - 2 paragraphs with 3 sentences each
                1st paragraph sentence lengths : [32, 100, 20] (total 152)
                2nd paragraph sentence lengths : [25, 150, 15] (total 190)
            - max_length    = 200
            - max_overlap   = 50
            
            Splitted paragraphs :
            - 6 paragraphs, as each sentence is <= max_length (200) but both paragraphs are longer than `max_overlap_len / max_overlap` (10)
            
            Output :
            - 3 paragraphs :
                1st output paragraph sentence lengths : [32, 100, 20, 25] (total 177)
                2nd output paragraph sentence lengths : [20, 25, 150] (total 195)
                3rd output paragraph sentence lengths : [15] (total 15)
            
            // Explanations
            - The 1st paragraph now includes an additional sentence as it does not exceeds `max_length`
            The 2nd paragraph starts with the 2 last sentences of the previous paragraph, as their cumulated length is smaller than `max_overlap_len` (45 <= 50)
            The final paragraph only contains 1 sentence without overlap because the last sentence exceeds `max_overlap_len` (150 > 50)
    """
    paragraphs = deepcopy(paragraphs)
    for i, para in enumerate(paragraphs):
        if 'text' not in para:
            text = paragraph_to_text(para)
            if text:
                para['text'] = text
            else:
                assert para.get('type', None) in _multimodal_types, str(para)
    
    # here, all paragraphs must either have a "text" entry, either have "type" in "image/audio/video"
    
    if group_by and all(group_by in p for p in paragraphs):
        groups = group_paragraphs(paragraphs, group_by)
        
        paragraphs = []
        for group in groups.values():
            para = merge_paragraphs(group)
            assert all('type' in c for c in para.get('content', [])), str(para)
            text = paragraph_to_text(para)
            if text: para['text'] = text
            
            paragraphs.append(para)
    
    if not max_length:
        return paragraphs
    
    splitted = []
    for para in paragraphs:
        if 'text' not in para:
            splitted.append(para)
            continue
        elif any(c['type'] in _multimodal_types for c in para.get('content', [])):
            chunks = split_content(
                para['content'],
                max_length  = max_length,

                max_overlap = max_overlap,
                max_overlap_len = max_overlap_len,

                ** kwargs
            )
            
            infos = {k : v for k, v in para.items() if k not in ('text', 'content')}
            for c in chunks: c.update(infos)
            
            splitted.extend(chunks)
        else:
            chunks = split_text(
                para['text'],
                max_length  = max_length,

                max_overlap = max_overlap,
                max_overlap_len = max_overlap_len,

                ** kwargs
            )

            para.pop('content', None)
            splitted.extend(
                {** para, 'text' : text} for text in chunks
            )
    
    return splitted

def group_paragraphs(paragraphs, key):
    """
        Group `paragraphs` into groups that have the same value for `key`(s)
        
        Arguments :
            - paragraphs    : a `list` of `dict`, the paragraphs to group
            - key   : the (list of) key(s) to determine groups
        Return :
            - groups    : an `OrderedDict` of groups, where keys are the groups and values the list of paragraphs belonging to that group
    """
    if isinstance(key, str): key = [key]
    
    groups = collections.OrderedDict()
    for para in paragraphs:
        group = tuple(_to_hashable(para.get(k, ())) for k in key)
        groups.setdefault(group, []).append(para)
    return groups

def merge_paragraphs(paragraphs):
    """
        Merge a `list` of paragraphs into a single paragraph
        
        Arguments :
            - paragraphs    : the list of paragraphs to merge
        Return :
            - merged    : a `dict` with a `content` entry that is the list of individual paragraphs
    """
    if len(paragraphs) <= 1: return paragraphs
    
    common  = set(paragraphs[0].keys())
    content = paragraphs
    for para in content:
        if 'type' not in para: para['type'] = 'text'
        common = common.intersection(set(para.keys()))
    common = common.difference({'type', 'text', 'image', 'audio', 'video'})
    
    merged = {'content' : content}
    for k in common:
        ref = content[0][k]
        if all(not hasattr(c[k], 'shape') and c[k] == ref for c in content[1:]):
            for c in content: c.pop(k)
            merged[k] = ref
    
    return merged

def paragraph_to_text(paragraph, format = None, separator = '\n\n'):
    """ Return a string representing the content of the paragraph """
    if isinstance(paragraph, str): paragraph = {'text' : paragraph}
    else: assert isinstance(paragraph, dict), str(paragraph)
    
    if format:
        return format_text(format, ** kwargs)
    elif 'text' in paragraph:
        return paragraph['text']
    elif 'content' in paragraph:
        if isinstance(paragraph['content'], str):
            return paragraph['content']
        elif any('text' in c for c in paragraph['content']):
            return separator.join(c['text'] for c in paragraph['content'] if 'text' in c)
        else:
            return None
    elif 'type' not in paragraph:
        raise RuntimeError('Paragraphs without "type" should have a "text" entry : {}'.format(
            paragraph
        ))
    elif paragraph['type'] == 'list':
        return '\n- ' + '\n-'.join(paragraph['items'])
    elif paragraph['type'] == 'table':
        return '\n- ' + '\n-'.join(paragraph['rows'])
    elif paragraph['type'] in ('image', 'audio', 'video'):
        return None
    else:
        raise ValueError('Unknown paragraph type : {}'.format(paragraph['type']))

def split_content(content, max_length, ** kwargs):
    splitted, mm_content, text_content = [], [], []
    for c in content:
        if c['type'] in _multimodal_types:
            if text_content:
                chunks = split_text(
                    '\n\n'.join(c['text'] for c in text_content), max_length, ** kwargs
                )
                if mm_content:
                    chunks[0] = {
                        'content' : mm_content + [{'type' : 'text', 'text' : chunks[0]}],
                        'text'    : chunks[0]
                    }
                splitted.extend(chunks)

                mm_content, text_content = [], []
            
            mm_content.append(c)
        else:
            text_content.append(c)
    
    if text_content:
        chunks = split_text(
            '\n\n'.join(c['text'] for c in text_content), max_length, ** kwargs
        )
        if mm_content:
            chunks[0] = {
                'content' : mm_content + [{'type' : 'text', 'text' : chunks[0]}],
                'text'    : chunks[0]
            }
        splitted.extend(chunks)
    
    for i, para in enumerate(splitted):
        if isinstance(para, str): splitted[i] = {'text' : para}
    
    return splitted

def process_paragraphs(paragraphs,
                       *,
                      
                       group_by = None,
                       
                       footnote_mode    = 'keep',
                       footnote_format  = 'Footnote: {text}',
                       
                       skip_urls    = False,
                       skip_files   = False,
                       skip_header  = True,
                       skip_page_number = True,
                       
                       ** kwargs
                      ):
    assert footnote_mode in ('keep', 'skip', 'insert', 'insert_next_sentence', 'insert_last_sentence')
    
    if skip_header:
        paragraphs = [p for p in paragraphs if not p.get('is_header', False)]
    
    if skip_page_number:
        paragraphs = [p for p in paragraphs if not p.get('is_page_number', False)]

    if skip_urls or skip_files:
        for p in paragraphs:
            if 'text' in p and '://' in p['text']:
                if skip_urls: p['text'] = remove_urls(p['text']).strip()
                if skip_files: p['text'] = remove_files(p['text']).strip()
        
        paragraphs = [p for p in paragraphs if 'text' not in p or p['text']]
        
    if footnote_mode == 'skip':
        paragraphs = [p for p in paragraphs if not p.get('is_footnote', False)]
        for p in paragraphs:
            for idx in p.get('footnotes', []):
                p['text'] = p['text'].replace('^{}'.format(idx), '')
    elif 'insert' in footnote_mode:
        footnotes = {
            '{}-{}'.format(p.get('page', None), p['footnote_index']) : p
            for p in paragraphs if p.get('is_footnote', False)
        }
        paragraphs = [p for p in paragraphs if not p.get('is_footnote', False)]

        for p in paragraphs:
            for idx in p.get('footnotes', []):
                p['text'] = _insert_footnote(
                    footnotes, p['text'], p.get('page', None), idx, footnote_mode, footnote_format
                )
    
    if group_by:
        groups = group_paragraphs(paragraphs, group_by)
        paragraphs = []
        for group in groups:
            merged = merge_paragraphs(group, skip = ('text', 'box'))
            merged['text'] = '\n\n'.join([p['text'] for p in group if 'text' in p])
            paragraphs.append(merged)
    
    return paragraphs

def _insert_footnote(_footnotes, paragraph, page, index, mode, format):
    _key = '{}-{}'.format(page, index)
    if _key not in _footnotes:
        logger.info('Footnote {} not found'.format(_key))
        return paragraph
    
    formatted = ' ' + format.format(
        text = _footnotes[_key]['text'], index = index
    ).replace('^{}'.format(index), '', 1).lstrip()
    if mode == 'insert':
        return paragraph.replace('^{}'.format(index), formatted)
    elif mode == 'insert_next_sentence':
        sentences = split_sentences(paragraph)
        for i, sent in enumerate(sentences):
            if '^{}'.format(index) in sent:
                break
        sentences[i] = sentences[i].replace('^{}'.format(index), '')
        sentences.insert(i + 1, formatted)
        return ''.join(sentences)
    elif mode == 'insert_last_sentence':
        return paragraph.replace('^{}'.format(index), '') + formatted

def _to_hashable(x):
    return tuple(x) if isinstance(x, list) else x
