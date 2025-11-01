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

from functools import cache

from .text_processing import format_text, split_text

logger  = logging.getLogger(__name__)

_skip_keys  = ('text', 'box', 'items', 'rows')

def chunks_from_paragraphs(paragraphs,
                           max_length,
                           *,
                           
                           group_by = None,
                           missmatch_mode   = 'ignore',
                           
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
            
            - separator : character used to separate two sentences from different paragraphs
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
    paragraphs = paragraphs.copy()
    for i, para in enumerate(paragraphs):
        if 'text' not in para:
            para = para.copy()
            para['text'] = paragraph_to_text(para)
            paragraphs[i] = para
    
    paragraphs = [p for p in paragraphs if p['text']]
    
    if group_by and all(group_by in p for p in paragraphs):
        groups = group_paragraphs(paragraphs, group_by)
        
        paragraphs = []
        for group in groups:
            para = merge_paragraphs(group, missmatch_mode, skip = _skip_keys)
            para['text'] = '\n\n'.join(para['text'] for para in group)
            paragraphs.append(para)
    
    if not max_length:
        return paragraphs
    
    splitted = []
    for para in paragraphs:
        chunks  = split_text(
            para['text'],
            max_length  = max_length,
            
            max_overlap = max_overlap,
            max_overlap_len = max_overlap_len,
            
            ** kwargs
        )
        
        splitted.extend(
            {** para, 'text' : text} for text in chunks
        )
        
    return splitted

def group_paragraphs(paragraphs, key):
    """ Group `paragraphs` into groups that have the same value for `key` entry(ies) """
    if isinstance(key, str): key = [key]
    
    groups = collections.OrderedDict()
    for para in paragraphs:
        group = tuple(_to_hashable(para.get(k, ())) for k in key)
        groups.setdefault(group, []).append(para)
    return list(groups.values())

def merge_paragraphs(paragraphs, missmatch_mode = 'ignore', skip = None):
    """ Takes the intersection or union of a list of paragraphs """
    if not skip:                    skip = {}
    elif isinstance(skip, str):     skip = {skip}
    else:                           skip = set(skip)
    
    merged = {k : v for k, v in paragraphs[0].items() if k not in skip}
    for para in paragraphs[1:]:
        for k, v in para.items():
            if hasattr(v, 'shape') or hasattr(merged.get(k, None), 'shape'): continue
            
            if k in skip:           continue
            elif k not in merged:   merged[k] = v
            elif merged[k] == v:    continue
            elif missmatch_mode == 'first': continue
            elif missmatch_mode == 'error':
                raise RuntimeError('Values for key {} missmatch : {} vs {}'.format(k, merged[k], v))
            else:
                if missmatch_mode == 'skip':
                    warnings.warn('Values for key {} missmatch : {} vs {}'.format(k, merged[k], v))
                merged.pop(k)
                skip.add(k)
    
    return merged

def paragraph_to_text(paragraph, format = None):
    if isinstance(paragraph, str): return paragraph
    
    assert isinstance(paragraph, dict), str(paragraph)
    
    if format:
        return format_text(format, ** kwargs)
    elif 'text' in paragraph:
        return paragraph['text']
    elif 'type' not in paragraph:
        raise RuntimeError('Paragraphs without "type" should have a "text" entry : {}'.format(
            paragraph
        ))
    elif paragraph['type'] == 'list':
        return '\n- ' + '\n-'.join(paragraph['items'])
    elif paragraph['type'] == 'table':
        return '\n- ' + '\n-'.join(paragraph['rows'])
    elif paragraph['type'] in ('document', 'image', 'audio', 'video'):
        return None
    else:
        raise ValueError('Unknown paragraph type : {}'.format(paragraph['type']))
def _to_hashable(x):
    return tuple(x) if isinstance(x, list) else x
