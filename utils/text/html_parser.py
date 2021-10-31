def parse_html(text, level = 2, title_separator = ' - ', p_separator = '\n\n', tag_title = 'h1', ** kwargs):
    def normalize_paragraph(title_levels, current_paragraph_idx, paragraph):
        if not paragraph.get('text', []): return {'text' : []}, None
        formatted = {
            'title'     : title_separator.join(title_levels).replace('[edit]', ''),
            'text'      : p_separator.join(paragraph['text']),
            'end_p_idx'   : current_paragraph_idx,
            ** {k : v for k, v in paragraph.items() if k not in ('title', 'text')},
            ** kwargs
        }
        return {'text' : []}, formatted
    
    from bs4 import BeautifulSoup
    
    parser = BeautifulSoup(text)
    
    tag_titles = [tag_title] + ['h{}'.format(i+2) for i in range(level)]
    tags = ['p'] + tag_titles
    
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
        if text:
            current_paragraph['text'].append(text)
            current_paragraph.setdefault('start_p_idx', current_paragraph_idx)
        current_paragraph_idx += 1
    
    current_paragraph, parsed = normalize_paragraph(title_levels, current_paragraph_idx, current_paragraph)
    if parsed: paragraphs.append(parsed)
    
    return paragraphs