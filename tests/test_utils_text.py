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
import logging
import unittest
import numpy as np

from absl.testing import parameterized
try:
    from transformers import AutoTokenizer
except:
    AutoTokenizer   = None

from utils.text import *
from . import CustomTestCase, data_dir, reproductibility_dir, is_tensorflow_available

_default_texts  = [
    "Hello World !",
    "Bonjour à tous !",
    "1, 2, 3, 4, 5, 6, 7, 8, 9 et 10 !",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
]

class TestNumbersCleaners(CustomTestCase, parameterized.TestCase):
    @parameterized.parameters(
        ('1g', 'one gram'),
        ('2g', 'two grams'),
        ('3m', 'three meters'),
        ('4l', 'four liters'),
        ('5mi', 'five miles'),
        ('6 t', 'six tons'),
        
        ('7 mm', 'seven milimeters'),
        ('8 kg', 'eight kilograms'),
        ('9 Mo', 'nine megaoctets'),
        ('10 Gb', 'ten gigabits'),
        
        ('5cm/s',   'five centimeters per second'),
        ('10km/h',  'ten kilometers per hour')
    )
    def test_math(self, text, target):
        self.assertEqual(target, normalize_numbers(text))

    @parameterized.parameters(
        ('-1', ' minus one'),
        ('+1', ' plus one'),
        ('1+1', 'one plus one'),
        ('1 + 1', 'one plus one'),
        ('1-1', 'one - one'),
        ('1 - 1', 'one minus one'),
        ('-1 - -1', ' minus one minus minus one'),
        ('-1 * -1', ' minus one times minus one'),
        ('-1.5 / - 2.5', ' minus one punt five divide by minus two punt five'),
    )
    def test_math(self, text, target):
        self.assertEqual(target, normalize_numbers(text))
        
    def test_money(self):
        self.assertEqual(normalize_numbers('$10'), 'ten dollars')
        self.assertEqual(normalize_numbers('$1', 'fr'), 'un dollar')

    @parameterized.parameters(
        ('en', '1 sec', 'one second'),
        ('en', '10sec', 'ten seconds'),
        ('en', '1min', 'one minute'),
        ('en', '2 min 1sec', 'two minutes and one second'),
        ('en', '1h', 'one hour'),
        ('en', '2 h 2min', 'two hours and two minutes'),
        ('en', '10 h 10 sec', 'ten hours and ten seconds'),
        ('en', '23h 59min 59sec', 'twenty-three hours and fifty-nine minutes and fifty-nine seconds'),
        
        ('fr', '1 sec', 'une seconde'),
        ('fr', '10sec', 'dix secondes'),
        ('fr', '1min', 'une minute'),
        ('fr', '2 min 1sec', 'deux minutes et une seconde'),
        ('fr', '1h', 'une heure'),
        ('fr', '2 h 2min', 'deux heures et deux minutes'),
        ('fr', '10 h 10 sec', 'dix heures et dix secondes'),
        ('fr', '23h 59min 59sec', 'vingt-trois heures et cinquante-neuf minutes et cinquante-neuf secondes'),
    )
    def test_time(self, lang, text, target):
        self.assertEqual(target, normalize_numbers(text, lang = lang))

    @parameterized.parameters(
        ('en', '3rd', 'third'),
        ('en', '2nd', 'second'),
        ('en', '10ème', 'tenth'),
        
        ('fr', '2nd', 'deuxième'),
        ('fr', '3rd', 'troisième'),
        ('fr', '10ième', 'dixième'),
        
        ('be', '1er', 'premier'),
        ('be', '3rd', 'troisième'),
        ('be', '70ème', 'septantième'),
        ('be', '91ème', 'nonante et unième')
    )
    def test_ordinal(self, lang, text, target):
        self.assertEqual(target, normalize_numbers(text, lang = lang))

    @parameterized.parameters(
        ('1, 2, 3, 4 and 5 !', 'one, two, three, four and five !'),
        ('1 000', 'one thousand'),
        ('1 000 000', 'one million'),
        ('1.5', 'one punt five'),
        ('put during 3-4 min', 'put during three - four minutes')
    )
    def test_others(self, text, target):
        self.assertEqual(target, normalize_numbers(text))


class TestCleaners(CustomTestCase):
    def test_strip(self):
        self.assertEqual(strip(' Hello  World !  '), 'Hello  World !')
        self.assertEqual(lstrip(' Hello  World ! '), 'Hello  World ! ')
        self.assertEqual(rstrip(' Hello  World ! '), ' Hello  World !')
        self.assertEqual(collapse_whitespace(' Hello  World   !'), ' Hello World !')
    
    def test_punctuation(self):
        self.assertEqual(detach_punctuation('Bonjour, comment ça va?'), 'Bonjour ,  comment ça va ?')
        self.assertEqual(remove_punctuation('Bonjour, comment ça va?'), 'Bonjour comment ça va')

    def test_replacements(self):
        self.assertEqual(replace_words('Ceci es un test', {'es' : 'est'}), 'Ceci est un test')
        self.assertEqual(replace_words('Ceci es un test', {'est' : ''}), 'Ceci es un test')
        self.assertEqual(replace_words('C\'est un c test', {'c' : ''}), '\'est un  test')
        self.assertEqual(replace_words(
            'C\'est un C test', {'c' : ''}, pattern_format = r"\b({})\b(?!')"
        ), 'C\'est un  test')
        self.assertEqual(expand_abreviations('Mr test', lang = 'en'), 'mister test')
        self.assertEqual(expand_abreviations('Mr. test', lang = 'en'), 'mister test')
        
        self.assertEqual(remove_tokens('Hello World !', ['hello']), ' World !')
        self.assertEqual(remove_tokens('Hello the World !', ['this', 'the']), 'Hello  World !')

class TestMetrics(CustomTestCase):
    def test_text_f1(self):
        self.assertEqual([1, 1, 1, 1], text_f1("Hello World !", "Hello ! World"))
        self.assertEqual([0, 1, 1, 1], text_f1("Hello World !", "Hello ! World", normalize = False))
        self.assertEqual(
            [0, 2 / 3, 2 / 3, 2 / 3], text_f1("Hello World !", "Hello ! world", normalize = False)
        )
        self.assertEqual([1, 1, 1, 1], text_f1("Hello World !", "Hello world"))
        self.assertEqual([0, 1, 1, 1], text_f1([0, 1, 2], [0, 2, 1]))
        self.assertEqual([1, 1, 1, 1], text_f1([0, 1, 2], [0, 2], exclude = [1]))
        self.assertEqual([0, 0.8, 1, 2 / 3], text_f1([0, 1, 2], [0, 2]))

class TestTextSplitter(CustomTestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ('single', 'Hello World !', 1),
        ('simple_1', 'Hello World ! This is a test', 2),
        ('simple_2', 'Hello World ? This is a test', 2),
        ('simple_3', 'Hello World. This is a test', 2),
        ('simple_4', 'Hello World... This is a test.', 2),
        
        ('url', 'This is an url : http://example.example.com', 1),
        ('mail', 'This is an email : example.example@example.com', 1),
        
        ('enum_1', '1. First item.\n2. Second item.\n3. 3rd item.', 3),
        ('enum_2', 'Examples :\n1. First item.\n2. Second item.\n3. 3rd item.', 4),
        ('enum_3', 'Examples : \n1. First item.\n2. Second item.\n3. 3rd item.', 4),
        ('enum_multi', 'Example :\n1. First item\n    1.1 First item A\n    1.2 First item B\n2. Second item', 5),
        ('enum_single_line', 'Items are : 1) First item 2) Second item 3) Third item', 1),
        ('itemize_1', 'List of items :\n- First item\n- Second item\n- Third item', 4),
        ('itemize_2', 'Equations :\n- 1 + 1 = 2\n- 1 - 1 = 0\n- -1 * 2 = -2', 4),
        
        ('decimal_1', 'Equation : 1.2 + 1.8 = 3.0', 1),
        ('decimal_2', 'Equation 1 : 1.2 + 1.8 = 3. \nEquation 2 : 1.8 - 1.8 = 0.\nend', 3),
        ('decimal_3', '1.2 + 1.3 = 2.5. 1.3 + 1.2 = 2.5. Addition is commutative', 3),
        
        ('quote_1', 'She said "Hello World !"', 1),
        ('quote_2', 'E.g., "Hello World !"', 1),
        ('quote_3', 'E.g. "Hello World !"', 1),
        
        ('acronym', 'M.H.C.P. stands for "Mental Health Counsuling Program"', 1)
    )
    def test_split_sentences(self, text, target):
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), target, 'Result : {}'.format(sentences))
    
    @parameterized.parameters(
        (['a', 'b', 'c', 'd'], 2, [[0, 1], [2, 3]]),
        (['a', 'b', 'c', 'd'], 3, [[0, 1, 2], [3]]),
        (['ab', 'c', 'def', 'g'], 3, [[0, 1], [2], [3]])
    )
    def test_merging_simple(self, text, max_length, target):
        merged, _, indices = merge_texts(text, max_length)
        self.assertEqual(indices, target, 'Merged : {}'.format(merged))
    
    @parameterized.parameters(
        (['a', 'b', 'c', 'd'], 2, [[0, 1], [2, 3]]),
        (['a', 'b', 'c', 'd'], 3, [[0, 1, 2], [3]]),
        (['ab', 'c', 'def', 'g'], 3, [[0, 1, 2], [3]]),
        (['Hello World', '!'], 3, [[0, 1]]),
        (['Hello', 'World', '!', 'This', 'is a test'], 3, [[0, 1, 2], [3], [4]])
    )
    def test_merging_words(self, text, max_length, target):
        merged, _, indices = merge_texts(text, max_length, tokenizer = lambda text: text.split())

class TestTokensProcessing(CustomTestCase):
    def test_text_filtering(self):
        texts   = np.tile(np.arange(10)[np.newaxis], [10, 1]).astype(np.int32)
        lengths = np.array([3, 5, 7, 1, 2, 4, 8, 9, 6, 10], dtype = np.int32)
        for i, l in enumerate(lengths): texts[i, l:] = -1
        
        self.assertEqual(
            filter_texts(texts, lengths), (texts, lengths)
        )
        
        self.assertEqual(
            filter_texts(texts, lengths, max_text_length = 5),
            (texts[lengths <= 5, :5], lengths[lengths <= 5])
        )
        mask = np.logical_and(lengths <= 5, lengths >= 2)
        self.assertEqual(
            filter_texts(texts, lengths, min_text_length = 2, max_text_length = 5),
            (texts[mask, :5], lengths[mask])
        )
        
        self.assertEqual(
            filter_texts(texts, lengths, min_text_length = 2, max_total_length = 5),
            (texts[[0], :3], lengths[[0]])
        )
        self.assertEqual(
            filter_texts(texts, lengths, min_text_length = 2, max_total_length = 5, sort_by_length = True),
            (texts[[0, 4], :3], lengths[[0, 4]])
        )
        self.assertEqual(
            filter_texts(texts, lengths, max_total_length = 8),
            (texts[[0, 1], :5], lengths[[0, 1]])
        )
        self.assertEqual(
            filter_texts(texts, lengths, max_total_length = 8, sort_by_length = True),
            (texts[[0, 3, 4], :3], lengths[[0, 3, 4]])
        )

        
        self.assertEqual(filter_texts(
            texts, lengths, max_total_length = 8, required_idx = 1
        ), (texts[[0, 1], :5], lengths[[0, 1]]))
        self.assertEqual(filter_texts(
            texts, lengths, max_total_length = 8, sort_by_length = True, required_idx = 1
        ), (texts[[1, 3, 4], :5], lengths[[1, 3, 4]]))
        self.assertEqual(
            filter_texts(texts, lengths, max_text_length = 4, required_idx = 1),
            (texts[[]], lengths[[]])
        )
    
    def test_logits_filtering(self):
        logits = np.random.normal(size = (5, 25)).astype(np.float32)
        for i in range(0, 25, 5):
            with self.subTest(i = i):
                filtered = logits.copy()
                filtered[:, i :] = - np.inf
                self.assertEqual(
                    mask_slice_tokens(logits, i, remove_after = True), filtered
                )

                filtered = logits.copy()
                filtered[:, : i] = - np.inf
                self.assertEqual(
                    mask_slice_tokens(logits, i, remove_after = False), filtered
                )

                indexes = [i, 24, 1]
                filtered = logits.copy()
                filtered[:, indexes] = - np.inf
                self.assertEqual(
                    mask_batch_tokens(logits, indexes), filtered
                )

class TestTokenizer(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.tokenizer = default_english_tokenizer(
            vocab_size  = 150,
            pad_token   = '_',
            sos_token   = '<s>',
            eos_token   = '</s>',
            use_sos_and_eos = True
        )
    
    def test_attributes(self):
        self.assertEqual(150, len(self.tokenizer))
        self.assertEqual(150, self.tokenizer.vocab_size)
        self.assertEqual(en_symbols, self.tokenizer.vocab[: len(en_symbols)])
        self.assertEqual(en_symbols, [self.tokenizer[i] for i in range(len(en_symbols))])
        
        self.assertEqual(en_symbols[0], self.tokenizer.blank_token)
        self.assertEqual(0, self.tokenizer.blank_token_idx)
        self.assertEqual(148, self.tokenizer.sos_token_idx)
        self.assertEqual(149, self.tokenizer.eos_token_idx)
    
    @parameterized.parameters(* _default_texts)
    def test_encode(self, text):
        self.assertEqual(
            ['<s>'] + list(text) + ['</s>'],
            self.tokenizer.tokenize(
                self.tokenizer.sos_token + text + self.tokenizer.eos_token, cleaned = True
            )
        )
        self.assertEqual(
            [self.tokenizer[c] for c in text if c in self.tokenizer],
            self.tokenizer.encode(text, cleaned = True, add_sos_and_eos = False)
        )
        self.assertEqual(
            [148] + [self.tokenizer[c] for c in text if c in self.tokenizer] + [149],
            self.tokenizer.encode(text, cleaned = True)
        )
        
    @parameterized.parameters(* _default_texts)
    def _test_format(self, text):
        self.assertEqual(
            self.tokenizer.format('{text}', text = text, fake = 'hello !', cleaned = True),
            [148] + [self.tokenizer[c] for c in text if c in self.tokenizer] + [149]
        )
        self.assertEqual(
            self.tokenizer.format('Text : {text}', text = text, fake = 'hello !', cleaned = True),
            [148] + [self.tokenizer[c] for c in 'Text : ' + text if c in self.tokenizer] + [149]
        )
        
    def test_batched_encode(self):
        all_encoded = [self.tokenizer.encode(txt, cleaned = True) for txt in _default_texts]
        all_encoded_padded = np.full(
            (len(all_encoded), max([len(e) for e in all_encoded])), self.tokenizer.blank_token_idx
        ).astype(np.int32)
        for i, enc in enumerate(all_encoded): all_encoded_padded[i, : len(enc)] = enc

        self.assertEqual(
            all_encoded, self.tokenizer.encode(_default_texts, cleaned = True, return_type = 'list')
        )
        self.assertEqual(
            all_encoded_padded,
            self.tokenizer.encode(_default_texts, cleaned = True, return_type = 'np')
        )
    
    @unittest.skipIf(not is_tensorflow_available(), 'tensorflow is not available')
    def test_tf_function(self):
        import tensorflow as tf
        
        pipe = tf.data.Dataset.from_tensor_slices(_default_texts).map(self.tokenizer.encode)
        for text, encoded in zip(_default_texts, pipe):
            self.assertTfTensor(encoded)
            self.assertEqual(self.tokenizer.encode(text), encoded)
        
        self.assertEqual(
            self.tokenizer.encode(_default_texts, return_type = 'np'),
            tf.function(self.tokenizer.encode)(tf.reshape(_default_texts, [-1]), shape = (None, None)),
        )

    @unittest.skipIf(AutoTokenizer is None, 'The `transformers` library is unavailable !')
    @parameterized.named_parameters(
        ('bart', 'facebook/bart-large'),
        ('mbart', 'moussaKam/barthez'),
        ('bert_uncased', 'bert-base-uncased'),
        ('bert_cased', 'bert-base-cased'),
        ('gpt2', 'gpt2', False),
        ('falcon', 'tiiuae/falcon-7b', False),
        ('flan_t5', 'google/flan-t5-large'),
        ('xlm_roberta', 'BAAI/bge-m3'),
        ('mistral', 'bofenghuang/vigostral-7b-chat', True, False)
    )
    def test_transformers_tokenizer(self, model_name, use_sos_and_eos = None, add_eos = None):
        transformers_encoder    = AutoTokenizer.from_pretrained(model_name)
        encoder = Tokenizer.from_transformers_pretrained(model_name)

        if use_sos_and_eos is not None: encoder.use_sos_and_eos = use_sos_and_eos
        
        for sent in _default_texts:
            self.assertEqual(
                transformers_encoder.tokenize(sent),
                encoder.tokenize(sent)
            )
            self.assertEqual(
                transformers_encoder(sent)['input_ids'],
                encoder.encode(sent, add_eos = add_eos)
            )

