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

import os
import logging
import unittest

from absl.testing import parameterized

from utils.text import *
from utils.text.cleaners import *
from unitests import CustomTestCase, data_dir, reproductibility_dir

_default_texts  = [
    "Hello World !",
    "Bonjour à tous !",
    "1, 2, 3, 4, 5, 6, 7, 8, 9 et 10 !",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
]

def is_tensorflow_available():
    try:
        import tensorflow as tf
        return True
    except:
        return False

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
    
    def test_money(self):
        self.assertEqual(expand_numbers('$10'), 'ten dollars')
        self.assertEqual(expand_numbers('$1', 'fr'), 'un dollar')

    def test_numbers(self):
        # Test for simple number expansion
        self.assertEqual(expand_numbers('1, 2, 3, 4 and 5 !'), 'one, two, three, four and five !')
        self.assertEqual(expand_numbers('1 000'), 'one thousand')
        self.assertEqual(expand_numbers('1 000 000'), 'one million')
        self.assertEqual(expand_numbers('1.5'), 'one punt five')
        self.assertEqual(
            expand_numbers('1, 2, 3, 4 et 5 !', 'fr'), 'un, deux, trois, quatre et cinq !'
        )
        self.assertEqual(expand_numbers('1.5', 'fr'), 'un virgule cinq')
        
    def test_time(self):
        self.assertEqual(expand_numbers('1h'), 'one hour')
        self.assertEqual(expand_numbers('2 min'), 'two minutes')
        self.assertEqual(expand_numbers('30 sec'), 'thirty seconds')
        self.assertEqual(expand_numbers('1h1min'), 'one hour and one minute')
        self.assertEqual(expand_numbers('5h 1min 5sec'), 'five hours one minute and five seconds')

        self.assertEqual(expand_numbers('1h', 'fr'), 'une heure')
        self.assertEqual(expand_numbers('2 min', 'fr'), 'deux minutes')
        self.assertEqual(expand_numbers('30 sec', 'fr'), 'trente secondes')
        self.assertEqual(expand_numbers('1h1min', 'fr'), 'une heure et une minute')
        self.assertEqual(
            expand_numbers('5h 1min 5sec', 'fr'), 'cinq heures une minute et cinq secondes'
        )


class TestTextEncoder(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.encoder = default_english_encoder(
            vocab_size  = 150,
            pad_token   = '_',
            sos_token   = '<s>',
            eos_token   = '</s>',
            use_sos_and_eos = True
        )
    
    def test_attributes(self):
        self.assertEqual(len(self.encoder), 150)
        self.assertEqual(self.encoder.vocab[: len(en_symbols)], en_symbols)
        self.assertEqual([self.encoder[i] for i in range(len(en_symbols))], en_symbols)
        
        self.assertEqual(self.encoder.blank_token, en_symbols[0])
        self.assertEqual(self.encoder.blank_token_idx, 0)
        self.assertEqual(self.encoder.sos_token_idx, 148)
        self.assertEqual(self.encoder.eos_token_idx, 149)
    
    @parameterized.parameters(* _default_texts)
    def test_encode(self, text):
        self.assertEqual(
            self.encoder.tokenize(self.encoder.sos_token + text + self.encoder.eos_token, cleaned = True),
            ['<s>'] + list(text) + ['</s>']
        )
        self.assertEqual(
            self.encoder.encode(text, cleaned = True, add_sos_and_eos = False),
            [self.encoder[c] for c in text if c in self.encoder]
        )
        self.assertEqual(
            self.encoder.encode(text, cleaned = True),
            [148] + [self.encoder[c] for c in text if c in self.encoder] + [149]
        )
        
    @parameterized.parameters(* _default_texts)
    def test_format(self, text):
        self.assertEqual(
            self.encoder.format('{text}', text = text, fake = 'hello !', cleaned = True),
            [148] + [self.encoder[c] for c in text if c in self.encoder] + [149]
        )
        self.assertEqual(
            self.encoder.format('Text : {text}', text = text, fake = 'hello !', cleaned = True),
            [148] + [self.encoder[c] for c in 'Text : ' + text if c in self.encoder] + [149]
        )
        
    def test_multi_functions(self):
        all_encoded = [
            self.encoder.encode(txt, cleaned = True) for txt in _default_texts
        ]
        all_encoded_padded = np.full(
            (len(all_encoded), max([len(e) for e in all_encoded])), self.encoder.blank_token_idx
        ).astype(np.int32)
        for i, enc in enumerate(all_encoded): all_encoded_padded[i, : len(enc)] = enc

        self.assertEqual(
            self.encoder.multi_encode(_default_texts, cleaned = True, return_type = 'list'),
            [all_encoded, [len(enc) for enc in all_encoded]]
        )
        self.assertEqual(
            self.encoder.multi_encode(_default_texts, cleaned = True, return_type = 'np'),
            (all_encoded_padded, np.array([len(enc) for enc in all_encoded], dtype = np.int32))
        )
        
        self.assertEqual(
            self.encoder.multi_format('{text}', text = _default_texts, cleaned = True, return_type = 'list'),
            [all_encoded, [len(enc) for enc in all_encoded]]
        )
        self.assertEqual(
            self.encoder.multi_format('{text}', text = _default_texts, cleaned = True, return_type = 'np'),
            (all_encoded_padded, np.array([len(enc) for enc in all_encoded], dtype = np.int32))
        )
    
    @unittest.skipIf(not is_tensorflow_available(), 'tensorflow is not available')
    def test_tf_function(self):
        import tensorflow as tf
        
        pipe = tf.data.Dataset.from_tensor_slices(_default_texts).map(self.encoder.encode)
        for text, encoded in zip(_default_texts, pipe):
            self.assertEqual(encoded, self.encoder.encode(text))
        
        self.assertEqual(
            tf.function(self.encoder.multi_encode)(tf.reshape(_default_texts, [-1])),
            self.encoder.multi_encode(_default_texts)
        )

        pipe = tf.data.Dataset.from_tensor_slices(tf.reshape(_default_texts, [-1, 1])).map(
            self.encoder.multi_encode
        )
        for text, encoded in zip(_default_texts, pipe):
            self.assertEqual(encoded, self.encoder.multi_encode([text]))

        pipe = tf.data.Dataset.from_tensor_slices(_default_texts).map(
            lambda txt: self.encoder.format('Text : {text}', text = txt)
        )
        for text, encoded in zip(_default_texts, pipe):
            self.assertEqual(encoded, self.encoder.encode('Text : {}'.format(text)))
            self.assertEqual(encoded, self.encoder.format('Text : {text}', text = text))
        
        self.assertEqual(
            tf.function(self.encoder.multi_format)('Text : {text}', text = _default_texts),
            self.encoder.multi_format('Text : {text}', text = _default_texts)
        )

        pipe = tf.data.Dataset.from_tensor_slices([_default_texts]).map(
            lambda txt: self.encoder.multi_format('Text : {text}', text = txt)
        )
        for text, encoded in zip([_default_texts], pipe):
            self.assertEqual(encoded, self.encoder.multi_format('Text : {text}', text = text))
    
class TestTextProcessing(CustomTestCase):
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
                    remove_slice_tokens(logits, i, remove_after = True), filtered
                )

                filtered = logits.copy()
                filtered[:, : i] = - np.inf
                self.assertEqual(
                    remove_slice_tokens(logits, i, remove_after = False), filtered
                )

                indexes = [i, 24, 1]
                filtered = logits.copy()
                filtered[:, indexes] = - np.inf
                self.assertEqual(
                    remove_batch_tokens(logits, indexes), filtered
                )

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
        self.assertEqual(indices, target, 'Merged : {}'.format(merged))