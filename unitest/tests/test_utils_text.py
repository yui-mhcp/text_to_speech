
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

import logging

from tqdm import tqdm

from utils.text.cleaners import *
from utils.text import TextEncoder, default_french_encoder, default_english_encoder, f1_score
from unitest import Test, set_sequential, assert_equal, assert_function

logger = logging.getLogger(__name__)

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
            _dataset = get_dataset('snli', modes = 'valid').sample(1000, random_state = 0)

            _sentences = [
                sent.strip() for sent in (
                    _dataset['text_x'].values.tolist() + _dataset['text_y'].values.tolist()
                )
            ]
        except ImportError:
            logger.warning("Module datasets is not available, using default sentences")
            _sentences = _default_sentences
    return _sentences

def test_transformers_encoder(name, use_sos_and_eos = None):
    import torch
    
    from transformers import AutoTokenizer

    transformers_encoder = AutoTokenizer.from_pretrained(name)
    text_encoder = TextEncoder.from_transformers_pretrained(name)
    
    if use_sos_and_eos is not None: text_encoder.use_sos_and_eos = use_sos_and_eos
    
    sentences = _maybe_load_dataset()
    
    set_sequential()
    for sent in sentences:
        assert_equal(transformers_encoder.tokenize, text_encoder.tokenize, sent, name = 'tokenize')
        
        assert_equal(
            lambda txt: transformers_encoder(txt)['input_ids'],
            text_encoder.encode, sent, name = 'encode'
        )

def test_text_encoder(encoder):
    set_sequential()
    for sent in _default_sentences:
        assert_function(encoder.encode, sent, name = 'encode')
        assert_function(encoder.invert, sent, name = 'decode')
        assert_function(encoder.split, sent, max_length = 150, name = 'split')
    
    assert_function(encoder.join, * _default_sentences, name = 'join')
    
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
    assert_equal("Bonjour a tous, comment ça va?", fr_convert_to_ascii, text_fr)

@Test
def test_english_text_encoder():
    test_text_encoder(default_english_encoder())

@Test
def test_french_encoder():
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
def test_gpt2_encoder():
    test_transformers_encoder('gpt2', False)

@Test
def test_f1():
    assert_equal([1, 1, 1, 1], f1_score("Hello World !", "Hello ! World"))
    assert_equal([0, 1, 1, 1], f1_score("Hello World !", "Hello ! World", normalize = False))
    assert_equal([0, 2 / 3, 2 / 3, 2 / 3], f1_score("Hello World !", "Hello ! world", normalize = False))
    assert_equal([1, 1, 1, 1], f1_score("Hello World !", "Hello world"))
    assert_equal([0, 1, 1, 1], f1_score([0, 1, 2], [0, 2, 1]))
    assert_equal([1, 1, 1, 1], f1_score([0, 1, 2], [0, 2], exclude = [1]))
    assert_equal([0, 0.8, 1, 2 / 3], f1_score([0, 1, 2], [0, 2]))

