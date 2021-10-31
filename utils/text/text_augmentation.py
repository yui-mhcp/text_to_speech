import tensorflow as tf

def random_mask(tokens,
                mask_token_idx,
                min_idx = 0,
                max_idx = None,
                nb_mask = 1,
                min_mask_length = 1,
                max_mask_length = 1
               ):
    """ Randomly mask `randrange(min_mask_length, max_mask_length)` tokens in `tokens[min_idx : max_idx]`"""
    def tf_randrange(min_val, max_val):
        range_val = tf.cast(tf.maximum(0, max_val - min_val), tf.float32)
        return tf.cast(tf.math.floor(tf.random.uniform(()) * range_val), tf.int32) + min_val
    
    if max_idx is None: max_idx = len(tokens)
    if max_idx < 0: max_idx = len(tokens) + max_idx
    if nb_mask == 0 or max_mask_length < min_mask_length: return tokens
    
    mask = tf.cast([mask_token_idx], tf.int32)
    
    masked = tokens
    for i in range(nb_mask):
        # if min_idx > max_idx: break
        mask_length = tf_randrange(min_mask_length, max_mask_length)
        mask_idx = tf_randrange(min_idx, max_idx - mask_length + 1)
        
        max_idx = max_idx - mask_length + 1
        masked = tf.concat([masked[:mask_idx], mask, masked[mask_idx + mask_length :]], axis = 0)
    
    return masked

    