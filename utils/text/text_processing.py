import tensorflow as tf

_max_length = 150

_end_sentence = ('...', '.', ' ?', ' !', '?', '!')

def create_padding_mask(seq, pad_value = 0, dtype = tf.float32):
    """
        Return padding mask matching attention shape [batch_size, 1, 1, seq_len]
    """
    mask = tf.cast(tf.math.equal(seq, pad_value), dtype = dtype)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(batch_size, size, dtype = tf.float32):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.tile(tf.expand_dims(mask, axis = 0), [batch_size, 1, 1])
    
    return tf.cast(mask, dtype = dtype)

def create_combined_mask(target, pad_value = 0):
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[0], tf.shape(target)[1])
    padding_mask    = create_padding_mask(target, pad_value = pad_value)
    return tf.maximum(look_ahead_mask, padding_mask)

def create_transformer_masks(inp, target, input_pad_value = 0, target_pad_value = 0):
    """
        Return 2 masks used in Transformer (Encoder + Decoder) architecture
        
        Arguments : 
            - inp       : input sequence (encoder input)
            - target    : target sequence (decoder input)
            - {input / target}_pad_value    : padding value for input / target sequence
        Return : [enc_padding_mask, combined_mask]
            - enc_padding_mask  : padding mask for encoder attention blocks
            - combined_mask     : combination of look_ahead_mask + padding mask on decoder input (target) for the 1st attention block of decoder layer
        
        Note : enc_padding_mask is used in encoder's MHA but also in the 2nd block of MHA in decoders layers
    """
    padding_mask    = create_padding_mask(inp, pad_value = input_pad_value)
    
    combined_mask   = create_combined_mask(target, target_pad_value)
    
    return padding_mask, combined_mask
    

def multi_split(text, * separators):
    """
        Split a text (str) based on multiple separators and return a list of tuple (part, separator) 
    """
    liste = [(text, '')]
    for sep in separators:
        new_liste = []
        for text, end_c in liste:
            parts = text.split(sep)
            for sub_part in parts[:-1]:
                new_liste.append((sub_part, sep))
            new_liste.append((parts[-1], end_c))
        liste = new_liste
    return liste
    
def simple_text_split(text, max_length = _max_length):
    """
        Split a text (word based) such that each part have at most 'max_length' caracters
    """
    mots = text.split(" ")

    text_parts = []
    length, parts = 0, []
    for mot in mots:
        parts.append(mot)
        length += len(mot)

        if length >= max_length:
            text_parts.append(" ".join(parts))
            length, parts = 0, []
    if length > 0: text_parts.append(" ".join(parts))
    
    return text_parts

def split_text(text, max_length = _max_length):
    """
        Split a text such that each parts have at most 'max_length' caracters. 
        The split is based on different criteria : 
        1) Split based on sentence ('_end_sentence' used as delimiters)
        2) If sentences are longer than 'max_length', split them based on comma
        3) If parts are still too long, split them on words
    """
    if isinstance(text, list):
        return [split_text(t, max_length) for t in text]
    
    text = text.replace('\n', ' ').strip()
    if len(text) == 0: return []
    elif len(text) <= max_length: return [text]
    
    if text[-1] in _end_sentence: text += ' '

    parts = []
    for part, end_char in multi_split(text, *_end_sentence):
        part = part.strip()
        # Skip empty parts
        if len(part) == 0: continue
        
        if len(part) <= max_length:
            # If part <= max_length, directly add it
            if len(parts) == 0 or len(parts[-1]) + len(part) > max_length:
                parts.append(part + end_char)
            else:
                parts[-1] += ' ' + part + end_char
                
        elif ', ' in part:
            # If part is longer but contains comma, split it based on commas
            splitted_part = part.split(", ")
            for i, sub_part in enumerate(splitted_part):
                sub_part = sub_part.strip()
                
                end_sub_part = end_char if i == len(splitted_part) -1 else ","
                if len(sub_part) <= max_length:
                    if len(parts) == 0 or len(parts[-1]) + len(sub_part) > max_length:
                        parts.append(sub_part + end_sub_part)
                    else:
                        parts[-1] += ' ' + sub_part + end_sub_part
                else:
                    sub_splitted = simple_text_split(sub_part, max_length)
                    sub_splitted[-1] += end_sub_part
                    for sub in sub_splitted:
                        sub = sub.strip()
                        if len(parts) == 0 or len(parts[-1]) + len(sub) > max_length:
                            parts.append(sub)
                        else:
                            parts[-1] += ' ' + sub
        else:
            splitted_part = simple_text_split(part, max_length)
            splitted_part[-1] += end_char
            for sub_part in splitted_part:
                sub_part = sub_part.strip()
                if len(parts) == 0 or len(parts[-1]) + len(sub_part) > max_length:
                    parts.append(sub_part)
                else:
                    parts[-1] += ' ' + sub_part
    
    return [p for p in parts if len(p) > 0]

