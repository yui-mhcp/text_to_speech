import re

_wiki_cleaner = r'(\[edit\]|\[[0-9]\])'

def parse_html(text,
               level    = 2,
               
               tag_title    = 'h1',
               title_separator  = ' - ',
               p_separator      = '\n\n',
               
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
        if len(title_levels) == 1 and not is_sub_document: return True
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
    
    _remove_regex = None if not remove_pattern else re.compile(remove_pattern)
    
    parser = BeautifulSoup(text)
    
    tag_titles = [tag_title] + ['h{}'.format(i+2) for i in range(level)]
    tags = ['p', 'li'] + tag_titles
    
    title_levels = []
    
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
        
        text = tag.text.strip()
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
        
    
    current_paragraph, parsed = normalize_paragraph(title_levels, current_paragraph_idx, current_paragraph)
    if parsed: paragraphs.append(parsed)
    
    return paragraphs