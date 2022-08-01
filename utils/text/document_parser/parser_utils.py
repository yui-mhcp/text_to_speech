
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

from utils.text.text_processing import multi_split, simple_text_split, split_text

def clean_paragraphs(paragraphs):
    cleaned = {}
    for page_idx, para_list in paragraphs.items():
        cleaned[page_idx] = []
        for p in para_list:
            p_clean = p.copy()
            if 'text' in p_clean:
                p_clean['text'] = p_clean['text'].strip()
                if len(p_clean['text']) == 0: continue
            
            cleaned[page_idx].append(p_clean)
    
    return cleaned

def infer_pages(paragraphs,
                start_number    = 0,
                max_paragraph_per_page  = 10,
                max_line_per_page       = 45,
                max_word_per_page       = 512,
                tqdm    = lambda x: x,
                ** kwargs
               ):
    """
        Split a list of paragraphs in pages.
        Arguments :
            - paragraphs    : a list of dict representing paragraphs containing (at least) field 'text'. 
            - start_number  : first page number. 
            - max_paragraph_per_page    : maximum number of paragraphs per page. 
            - max_line_per_page     : maximum number of line per page (line delimited by \n)
            - max_word_per_page     : maximum number of words per pages*.
            - tqdm  : progress bar
            - kwargs    : unused kwargs
        Return :
            a dict where keys are page numbers and values are a list of dict (paragraphs of this page). 
            
        * the function do not split big paragraphs so if a single paragraph have more words, it will be in a single page (the only paragraph for this page). 
    """    
    pages = {}
    
    page_number, total_p_words, total_p_lines = start_number, 0, 0
    current_paragraphs = []

    for paragraph in tqdm(paragraphs):
        text = paragraph['text']
        if text is not None and len(text) == 0: continue
        
        
        n_words, n_lines = len(text.split(' ')), len(text.split('\n'))
        
        if (
            len(current_paragraphs) >= max_paragraph_per_page or
            total_p_words + n_words >= max_word_per_page or 
            total_p_lines + n_lines >= max_line_per_page
        ):
            pages[page_number] = current_paragraphs
            
            page_number += 1
            current_paragraphs = []
            total_p_words, total_p_lines = 0, 0
        
        current_paragraphs.append(paragraph)
        total_p_words += n_words
        total_p_lines += n_lines
    
    if len(current_paragraphs) > 0:
        pages[page_number] = current_paragraphs
    
    return pages

def split_paragraphs(pages, max_paragraph_length):
    result = {}
    for page_number, paragraphs in pages.items():
        splitted = []
        for para in paragraphs:
            if 'text' not in para or len(para['text']) <= max_paragraph_length:
                splitted.append(para)
                continue
            
            parts = split_text(para['text'], max_paragraph_length)
            
            splitted.extend([{** para, 'text' : part} for part in parts])
        
        result[page_number] = splitted
            
    return result
