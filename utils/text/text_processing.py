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

import os
import re
import logging
import warnings

from functools import cache

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
            - text  : the text to split
            - max_length    : the maximum length for a given text
            
            - tokens    : pre-computed tokens for `text`
            - tokenizer : a tokenization function (or a `Tokenizer` instance)
            
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
        tokenizer = list
    elif hasattr(tokenizer, 'tokenize'):
        tokenizer = tokenizer.tokenize
    
    if isinstance(tolerance, float):      tolerance = int(tolerance * max_length)
    if isinstance(sent_tolerance, float): sent_tolerance = int(sent_tolerance * max_length)
    
    max_text_length = max_length + tolerance
    max_sent_length = max_length + sent_tolerance
    
    if tokens is None: tokens = tokenizer(text)
    if len(tokens) <= max_text_length: return [text] if not return_tokens else ([text], [tokens])
    
    splitted    = split_sentences(text, eos_pattern, strip = False)
    tokens      = [tokenizer(sent) for sent in splitted]
    
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
        tokenizer = list
    elif hasattr(tokenizer, 'tokenize'):
        tokenizer = tokenizer.tokenize

    if tokens is None:
        tokens = [tokenizer(txt) for txt in texts]
    
    texts   = [txt.strip(' ') for txt in texts]
    
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
                for i in range(1, 1 + min(max_overlap, len(merged_texts[-2]))):
                    if overlap_len + len(merged_tokens[-2][- i]) > _max_overlap_len: break

                    merged_texts[-1].insert(0, merged_texts[-2][- i])
                    merged_tokens[-1].insert(0, merged_tokens[-2][- i])
                    merged_indices[-1].insert(0, merged_indices[-2][- i])
                    overlap_len += len(merged_tokens[-2][- i])
                    merged_len  += len(merged_tokens[-2][- i])
    
    result_texts = [' '.join(texts) for texts in merged_texts]
    result_tokens = []
    for i, list_tokens in enumerate(merged_tokens):
        result_tokens.append([])
        for toks in list_tokens: result_tokens[-1].extend(toks)
    
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

def format_text(format, ** kwargs):
    """
        Apply `format` with `kwargs`:
        - if `format` is a python-like format, calls `format.format(** kwargs)`
        - if `format` is a `jinja` format, calls `jinja2.ImmutableSandboxedEnvironment.render(** kwargs)`
        - Otherwise, return `format`
    """
    if '{' not in format:
        return format
    elif '{%' in format or '{{' in format:
        return compile_jinja_template(format).render(** kwargs)
    elif re.search(r'\{[^\s\'\"]+\}', format):
        return format.format(** kwargs)
    else:
        return format

@cache
def compile_jinja_template(template):
    import jinja2

    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
    env.globals['raise_exception'] = raise_exception
    env.globals['basename'] = os.path.basename
    return env.from_string(template)


def get_pairs(text, n = 2):
    """ Creates a n-gram """
    return [tuple(text[i : i + n]) for i in range(0, len(text) - n + 1)]

def bpe(token, bpe_ranks, end_of_word = None):
    """ Computes the byte-pair-encoding (BPE) algorithm """
    word    = tuple(token) if end_of_word is None else tuple(token[:-1]) + (token[-1] + end_of_word, )
    pairs   = get_pairs(word)

    if not pairs:
        if not end_of_word: end_of_word = ''
        return token + end_of_word
    
    while True:
        bigram = min(pairs, key = lambda pair: bpe_ranks.get(pair, float('inf')))
        if bigram not in bpe_ranks: break
        
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break
            
            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1: break
        else: pairs = get_pairs(word)
    return word

def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def _is_end_of_quote(sentences, sent):
    if not sentences or not sent.strip(): return False
    prev, sent = sentences[-1], sent.strip().split()[0]
    return all(c in _closing_punctuation and _closing_punctuation[c] in prev for c in sent)
