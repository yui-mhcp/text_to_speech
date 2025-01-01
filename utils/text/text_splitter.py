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
import warnings
import collections

logger  = logging.getLogger(__name__)

_eos_chars = (
    '\n\n',             # typical paragraph end
    r'\.\.\.\s*', r'\?\s*', r'\!\s*', # standard punctuation
    # The basic rule is a dot followed by a space (or newline)
    # The `(?<!\.[a-zA-Z]{1})` rule excludes acronyms / "e.g." / "i.e." / ...
    r'(?<!\.[a-zA-Z]{1})\.\s+',
    r'\n(?=\s*[-\*\dA-Z])' # new line with itemize or starting a new sentence (lowercase excluded)
)
_closing_punctuation = {
    ')' : '(', ']' : '[', '}' : '{', '"' : '"', "'" : "'", "`" : "`"
}
_sentence_split_pattern = (r',(?!\d)', ': ', r'\(.*\)')

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
    if any(isinstance(para, str) for para in paragraphs):
        paragraphs = [{'text' : para} if isinstance(para, str) else para for para in paragraphs]
    
    if group_by:
        groups = group_paragraphs(paragraphs, group_by)
        texts  = [separator.join([para['text'] for para in group]) for group in groups]
        paragraphs  = [merge_paragraphs(group, missmatch_mode, skip = 'text') for group in groups]
        
        for para, text in zip(paragraphs, texts):
            para['text'] = text
    
    splitted = []
    for para in paragraphs:
        chunks  = split_text(
            para['text'],
            max_length  = max_length,
            
            max_overlap = max_overlap,
            max_overlap_len = max_overlap_len,
            
            ** kwargs
        )
        
        for text in chunks:
            splitted.append({** para, 'text' : text})
    
    return splitted

def group_paragraphs(paragraphs, key):
    """ Group `paragraphs` into groups that have the same value for `key` entry(ies) """
    if isinstance(key, str):                key = {key : lambda x: x}
    elif isinstance(key, (list, tuple)):    key = {k : lambda x: x for k in key}
    
    groups = collections.OrderedDict()
    for para in paragraphs:
        group = tuple(v(para.get(k, ())) for k, v in key.items())
        group = tuple(v if not isinstance(v, list) else tuple(v) for v in group)
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

def split_text(text,
               max_length,
               *,
               
               tokens   = None,
               tokenizer    = None,
               
               eos_pattern  = _eos_chars,
               sent_pattern = _sentence_split_pattern,
               
               tolerance    = 0,
               sent_tolerance   = 0,
               
               merge    = True,
               
               err_mode = 'skip',
               return_tokens    = False,
               
               ** kwargs
              ):
    """
        Splits `text` in a recursive way (paragraphs -> sentences -> sub sentences -> words)
        such that a single text in the result has at most `max_length` tokens (with possible tolerance)
        
        Arguments :
            - text  : the text to split (or list of text parts)
            - max_length    : the maximum length for a given text
            
            - tokens    : pre-computed tokens for `text`
            - tokenizer : a tokenization function (or a `TextEncoder` instance)
            
            - eos_pattern   : the split patterns to transform `text` in sentences
            - sent_pattern  : the split pattern to transform a sentence into sub-sentences
            
            - tolerance : the accepted tolerance to keep `text` complete
            - sent_tolerance    : the accepted tolerance to keep a sentence intact

            - merge : whether to merge text parts to form longest possible continuous blocks
            
            - err_mode  : the action to perform when a single word is longer than `max_length`
                          - error   : raise a RuntimeError
                          - skip    : ignore and displays a warning
                          - ignore  : ignore without warning
                          - keep    : adds the text even though it exceeds `max_length`
            
            - kwargs    : forwarded to `merge_texts` (if `merge == True`)
        Return :
            - splitted_texts    : a list of texts
            - splitted_tokens   : a list of text tokens
    """
    if tokenizer is None:
        tokenizer = lambda text: list(text)
    elif hasattr(tokenizer, 'encode'):
        _tokenizer = tokenizer
        tokenizer = lambda text: _tokenizer.encode(text, strip = False, return_type = 'list')
    
    if isinstance(tolerance, float):      tolerance = int(tolerance * max_length)
    if isinstance(sent_tolerance, float): sent_tolerance = int(sent_tolerance * max_length)
    
    max_text_length = max_length + tolerance
    max_sent_length = max_length + sent_tolerance
    
    if tokens is None: tokens = tokenizer(text)
    if len(tokens) <= max_text_length: return [text] if not return_tokens else ([text], [tokens])
    
    splitted    = split_sentences(text, eos_pattern, strip = False)
    tokens      = [tokenizer(txt) for txt in splitted]
    
    result_text, result_tokens = [splitted[0]], [tokens[0]]
    for split, tok in zip(splitted[1:], tokens[1:]):
        # adds the sentence if it is shorter than `max_sent_length`
        if len(tok) <= max_sent_length:
            result_text.append(split)
            result_tokens.append(tok)
        # if the sentence is too long, but we have a second pattern to split it
        elif sent_pattern:
            splitted_sent, splitted_sent_tok = split_text(
                split,
                max_sent_length,

                tokens  = tok,
                tokenizer   = tokenizer,

                eos_pattern = sent_pattern,
                sent_pattern    = ' ' if sent_pattern != ' ' else None,
                
                err_mode    = err_mode,
                return_tokens   = True
            )

            result_text.extend(splitted_sent)
            result_tokens.extend(splitted_sent_tok)
        
        # if `sent_pattern is None`, it probably means that `split` is a single word that is too long
        # it is therefore not possible to split it more, which may raise an error or skkip it
        elif err_mode == 'error':
            raise RuntimeError('It was not possible to split `{}`'.format(split))
        elif err_mode == 'ignore':
            continue
        elif err_mode == 'skip':
            warnings.warn('The text `{}` is skipped as it is too long'.format(split))
            continue
        elif err_mode == 'keep':
            result_text.append(split)
            result_tokens.append(tok)
    
    if merge:
        result_text, result_tokens, _ = merge_texts(
            result_text, max_text_length, tokens = result_tokens, tokenizer = tokenizer, ** kwargs
        )
    
    return result_text if not return_tokens else (result_text, result_tokens)

def merge_texts(texts,
                max_length,
                max_overlap    = 0,
                max_overlap_len    = 0.2,
                     
                *,
                  
                tokens = None,
                tokenizer  = None,
                
                ** _
               ):
    """
        Merges `text` such that the total number of tokens within a chunk
        is smaller or equal than `max_length`.
        The `max_overlap` controls the overlap between 2 consecutive chunks of text
        
        Arguments :
            - texts : list of text (str)
            - max_length    : maximum length for a given text
            - max_overlap   : maximum number of text to use as start overlap
            - max_overlap_len   : maximum length for the start overlap (can be relative to max_length)
            
            - tokens    : pre-computed tokens for `chunks`
            - tokenizer : the tokenization function (or a `TextEncoder` instance)
        Return :
            - chunks            : list of texts
            - chunk_tokens      : list of concatenated tokens
            - merged_indices    : list of list of index, such that chunks[i] is the concatenation of `[texts[idx] for idx in merged_indices[i]]`
    """
    if isinstance(max_overlap_len, float): max_overlap_len = int(max_overlap_len * max_length)
    
    if tokenizer is None:
        tokenizer = lambda text: list(text)
    elif hasattr(tokenizer, 'encode'):
        _tokenizer = tokenizer
        tokenizer = lambda text: _tokenizer.encode(text, strip = False, return_type = 'list')

    if tokens is None:
        tokens = [tokenizer(txt) for txt in texts]
    
    merged_texts    = [[texts[0]]]
    merged_tokens   = [[tokens[0]]]
    merged_indices  = [[0]]
    merged_len      = len(tokens[0])
    for i, (text, tok) in enumerate(zip(texts[1:], tokens[1:]), start = 1):
        # adds `chunk` to the current group as their union is smaller than `max_length`
        if merged_len + len(tok) <= max_length:
            merged_texts[-1].append(text)
            merged_tokens[-1].append(tok)
            merged_indices[-1].append(i)
            merged_len  += len(tok)
        else:
            merged_texts.append([text])
            merged_tokens.append([tok])
            merged_indices.append([i])
            merged_len = len(tok)
            
            # computes an overlap based on the "n" last parts of the current group
            # such that their cumulated length is smaller or equal than `max_overlap`
            if max_overlap > 0 and len(tok) < max_length:
                _max_overlap_len = min(max_overlap_len, max_length - len(tok))

                overlap_len = 0
                for i in range(min(max_overlap, len(merged_texts[-2]))):
                    if overlap_len + len(merged_tokens[-2][- i]) > _max_overlap_len: break

                    merged_texts[-1].insert(0, merged_texts[-2][- i])
                    merged_tokens[-1].insert(0, merged_tokens[-1][- i])
                    merged_indices[-1].insert(0, merged_indices[-2][- i])
                    overlap_len += len(merged_tokens[-2][- i])
                    merged_len  += len(merged_tokens[-2][- i])

    # transform the merged list of list into list of str by concatenating the different groups
    result_texts, result_tokens = [''] * len(merged_texts), [[] for _ in range(len(merged_tokens))]
    for i, (merged_txt, merged_tok) in enumerate(zip(merged_texts, merged_tokens)):
        update_tokens = False
        for txt, tok in zip(merged_txt, merged_tok):
            if result_texts[i] and not result_texts[i].endswith((' ', '\n')):
                update_tokens = True
                txt = ' ' + txt
            result_texts[i] += txt
            result_tokens[i].extend(tok)
        
        if update_tokens:
            result_tokens[i] = tokenizer(result_texts[i])

    return result_texts, result_tokens, merged_indices
    

def split_sentences(text, eos_pattern = _eos_chars, strip = False):
    """
        Splits `text` into sentences with additional post-processing
        
        Arguments :
            - text      : the text to split (str)
            - eos_text  : the "end of sentence" texts (list or tuple)
            - strip     : whether to remove spaces at the end of sentences ('\n' will not be removed)
        Return :
            - sentences : a list of sentences
        
        Note : the returned sentences include the end of sentence character.
        
        Handled cases : these are some special cases that are handled by additional post-processing
            - Section indexes : `1.1. [section title]`
            - Quotes          : `"[quote]."`
            - URL / mails     : `example.com`, `example@example.com`
            - ie / eg         : `e.g., [example]`
        
        Example : 
        ```python
        text     = '1. Fact questions (e.g., "What did Albert Einstein win the Nobel Prize for ?")\n',
        splitted = ['1', '. ', 'Fact questions (e', '.', 'g', '.', ', "What did Albert Einstein win the Nobel Prize for', ' ?', '")', '\n', '']
        output   = ['1. Fact questions (e.g., "What did Albert Einstein win the Nobel Prize for ?")\n']
        ```
    """
    splitted = split_and_join(text.strip(), eos_pattern)

    i = 0
    sentences = []
    while i < len(splitted):
        sent = splitted[i]
        # i % 2 == 1 means it is a split pattern
        if i % 2 == 1 or _is_end_of_quote(sentences, sent):
            if sentences: sentences[-1] += sent
            i += 1
            continue
        elif not sent.strip():
            i += 1
            continue
        
        # merge enumerations starting with "x. [...]"
        # the while loop is to handle sections-like "x.y.z. [...]"
        while i + 2 < len(splitted) and splitted[i].isdigit() and splitted[i + 1].strip() == '.':
            sent = sent + splitted[i + 1] + splitted[i + 2]
            i += 2
        
        sentences.append(sent)
        i += 1
    
    if strip: sentences = [sent.strip(' ') for sent in sentences]
    return sentences

def split_and_join(text, pattern, * args):
    """
        Splits text based on `pattern` (+ args) and returns a list where even indexes are splitted items, and odd ones are split pattern
        
        Example :
        ```python
        splitted = split_and_join("Hello World ! This is an example.", "!", ".")
        print(splitted) # ["Hello World ", "!", " This is an example", "."]
        ```
    """
    if isinstance(pattern, str): pattern = (pattern, )
    if args: pattern = tuple(pattern) + args
    
    return re.split('({})'.format('|'.join([
        re.escape(p) if '\\' not in p else p for p in pattern
    ])), text)


def _is_end_of_quote(sentences, sent):
    if not sentences or not sent.strip(): return False
    prev, sent = sentences[-1], sent.strip().split()[0]
    return all(c in _closing_punctuation and _closing_punctuation[c] in prev for c in sent)
