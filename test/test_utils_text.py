from utils.text.cleaners import *

from unitest import Test, set_sequential, assert_equal, assert_function

_default_sentences  = [
    "Hello World !",
    "Bonjour à tous !",
    "1, 2, 3, 4, 5, 6, 7, 8, 9 et 10 !",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
]

_sentences = None

def _maybe_load_dataset():
    global _sentences
    if _sentences is None:
        try:
            from datasets import get_dataset
            _dataset = get_dataset('snli', modes = 'valid')['valid']

            _sentences = _dataset['text_x'].values.tolist() + _dataset['text_y'].values.tolist()
            _sentences = [sent.strip() for sent in _sentences]
        except ImportError:
            print("Module datasets is not available, using default sentences")
            _sentences = _default_sentences
    return _sentences

def test_transformers_encoder(name):
    try:
        import torch
    except OSError as e:
        print("Error importing torch, cannot make the TextEncoder tests : {}".format(str(e)))
        return
    
    from transformers import AutoTokenizer
    from utils.text import TextEncoder

    transformers_encoder = AutoTokenizer.from_pretrained(name)
    text_encoder = TextEncoder.from_transformers_pretrained(name)
    text_encoder.rstrip = False
    
    sentences = _maybe_load_dataset()
    
    set_sequential()
    for sent in sentences:
        tokens_1, tokens_2 = text_encoder.tokenize(sent), transformers_encoder.tokenize(sent)
        
        assert_equal(text_encoder.tokenize, transformers_encoder.tokenize, sent)
        
        assert_equal(text_encoder.encode, lambda txt: transformers_encoder(txt)['input_ids'], sent)

def test_text_encoder(encoder):
    set_sequential()
    for sent in _default_sentences:
        assert_function(encoder.encode, sent)
        assert_function(lambda text: encoder.decode(encoder.encode(text)), sent)
        assert_function(encoder.split, sent, max_length = 150)
    
    assert_function(encoder.join, * _default_sentences)
    
@Test
def test_cleaners():
    text_en = " HellO WOrld ! "
    text_fr = "Bonjour à tous, comment ça va?"
    
    assert_equal("HellO WOrld !", strip, text_en)
    assert_equal("HellO WOrld ! ", lstrip, text_en)
    assert_equal(" HellO WOrld !", rstrip, text_en)

    assert_equal("Bonjour à tous ,  comment ça va ?", detach_punctuation, text_fr)
    
    assert_equal("Bonjour à tous comment ça va", remove_punctuation, text_fr)
    
    assert_equal("Bonjour    , comment   va?", remove_tokens, text_fr, tokens = ['à', 'tous', 'ça'])
    
    assert_equal( " hello world ! ", lowercase, text_en)
    
    assert_equal("Bonjour a tous, comment ca va?", convert_to_ascii, text_fr)
    assert_equal(text_fr, fr_convert_to_ascii, text_fr)

@Test
def test_english_text_encoder():
    from utils.text import default_english_encoder
    
    test_text_encoder(default_english_encoder())

@Test
def test_french_encoder():
    from utils.text import default_french_encoder
    
    test_text_encoder(default_french_encoder())
    
@Test
def test_bert_cased_encoder():
    test_transformers_encoder('bert-base-cased')

@Test
def test_bert_uncased_encoder():
    test_transformers_encoder('bert-base-uncased')

@Test
def test_bart_encoder():
    test_transformers_encoder('facebook/bart-large')

@Test
def test_f1():
    from utils.text import TextEncoder, f1_score
    
    assert_equal([1, 1, 1, 1], f1_score("Hello World !", "Hello ! World"))
    assert_equal([0, 1, 1, 1], f1_score("Hello World !", "Hello ! World", normalize = False))
    assert_equal([0, 2 / 3, 2 / 3, 2 / 3], f1_score("Hello World !", "Hello ! world", normalize = False))
    assert_equal([1, 1, 1, 1], f1_score("Hello World !", "Hello world"))
    assert_equal([0, 1, 1, 1], f1_score([0, 1, 2], [0, 2, 1]))
    assert_equal([1, 1, 1, 1], f1_score([0, 1, 2], [0, 2], exclude = [1]))
    assert_equal([0, 0.8, 1, 2 / 3], f1_score([0, 1, 2], [0, 2]))

