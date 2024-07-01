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
import tensorflow as tf

from utils.text import *
from utils.text.cleaners import *
from unitests import CustomTestCase, data_dir, reproductibility_dir

_default_texts  = [
    "Hello World !",
    "Bonjour à tous !",
    "1, 2, 3, 4, 5, 6, 7, 8, 9 et 10 !",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
]

class TestText(CustomTestCase):
    def test_cleaners(self):
        self.assertEqual(strip(' Hello  World !  '), 'Hello  World !')
        self.assertEqual(lstrip(' Hello  World ! '), 'Hello  World ! ')
        self.assertEqual(rstrip(' Hello  World ! '), ' Hello  World !')
        self.assertEqual(collapse_whitespace(' Hello  World   !'), ' Hello World !')
        
        self.assertEqual(detach_punctuation('Bonjour, comment ça va?'), 'Bonjour ,  comment ça va ?')
        self.assertEqual(remove_punctuation('Bonjour, comment ça va?'), 'Bonjour comment ça va')

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
        
        # Test for simple number expansion
        self.assertEqual(expand_numbers('1, 2, 3, 4 and 5 !'), 'one, two, three, four and five !')
        self.assertEqual(expand_numbers('1 000'), 'one thousand')
        self.assertEqual(expand_numbers('1 000 000'), 'one million')
        self.assertEqual(expand_numbers('1.5'), 'one punt five')
        self.assertEqual(
            expand_numbers('1, 2, 3, 4 et 5 !', 'fr'), 'un, deux, trois, quatre et cinq !'
        )
        self.assertEqual(expand_numbers('1.5', 'fr'), 'un virgule cinq')
        # Test for _expand_time (en)
        self.assertEqual(expand_numbers('1h'), 'one hour')
        self.assertEqual(expand_numbers('2 min'), 'two minutes')
        self.assertEqual(expand_numbers('30 sec'), 'thirty seconds')
        self.assertEqual(expand_numbers('1h1min'), 'one hour and one minute')
        self.assertEqual(expand_numbers('5h 1min 5sec'), 'five hours one minute and five seconds')
        # Test for _expand_time (fr)
        self.assertEqual(expand_numbers('1h', 'fr'), 'une heure')
        self.assertEqual(expand_numbers('2 min', 'fr'), 'deux minutes')
        self.assertEqual(expand_numbers('30 sec', 'fr'), 'trente secondes')
        self.assertEqual(expand_numbers('1h1min', 'fr'), 'une heure et une minute')
        self.assertEqual(
            expand_numbers('5h 1min 5sec', 'fr'), 'cinq heures une minute et cinq secondes'
        )
        
        self.assertEqual(expand_numbers('$10'), 'ten dollars')
        self.assertEqual(expand_numbers('$1', 'fr'), 'un dollar')

    def test_text_encoder(self):
        encoder = default_english_encoder(
            vocab_size  = 150,
            pad_token   = '_',
            sos_token   = '<s>',
            eos_token   = '</s>',
            use_sos_and_eos = True
        )
        
        self.assertEqual(len(encoder), 150)
        self.assertEqual(encoder.vocab[: len(en_symbols)], en_symbols)
        self.assertEqual([encoder[i] for i in range(len(en_symbols))], en_symbols)
        
        self.assertEqual(encoder.blank_token, en_symbols[0])
        self.assertEqual(encoder.blank_token_idx, 0)
        self.assertEqual(encoder.sos_token_idx, 148)
        self.assertEqual(encoder.eos_token_idx, 149)
        
        for text in _default_texts:
            with self.subTest(text = text):
                self.assertEqual(
                    encoder.tokenize(encoder.sos_token + text + encoder.eos_token, cleaned = True),
                    ['<s>'] + list(text) + ['</s>']
                )
                self.assertEqual(
                    encoder.encode(text, cleaned = True, add_sos_and_eos = False),
                    [encoder[c] for c in text if c in encoder]
                )
                self.assertEqual(
                    encoder.encode(text, cleaned = True),
                    [148] + [encoder[c] for c in text if c in encoder] + [149]
                )
                
                self.assertEqual(
                    encoder.format('{text}', text = text, fake = 'hello !', cleaned = True),
                    [148] + [encoder[c] for c in text if c in encoder] + [149]
                )
                self.assertEqual(
                    encoder.format('Text : {text}', text = text, fake = 'hello !', cleaned = True),
                    [148] + [encoder[c] for c in 'Text : ' + text if c in encoder] + [149]
                )

        all_encoded = [
            encoder.encode(txt, cleaned = True) for txt in _default_texts
        ]
        all_encoded_padded = np.full(
            (len(all_encoded), max([len(e) for e in all_encoded])), encoder.blank_token_idx
        ).astype(np.int32)
        for i, enc in enumerate(all_encoded): all_encoded_padded[i, : len(enc)] = enc

        self.assertEqual(
            encoder.multi_encode(_default_texts, cleaned = True, return_type = 'list'),
            [all_encoded, [len(enc) for enc in all_encoded]]
        )
        self.assertEqual(
            encoder.multi_encode(_default_texts, cleaned = True, return_type = 'np'),
            (all_encoded_padded, np.array([len(enc) for enc in all_encoded], dtype = np.int32))
        )
        
        self.assertEqual(
            encoder.multi_format('{text}', text = _default_texts, cleaned = True, return_type = 'list'),
            [all_encoded, [len(enc) for enc in all_encoded]]
        )
        self.assertEqual(
            encoder.multi_format('{text}', text = _default_texts, cleaned = True, return_type = 'np'),
            (all_encoded_padded, np.array([len(enc) for enc in all_encoded], dtype = np.int32))
        )
        
        
        pipe = tf.data.Dataset.from_tensor_slices(_default_texts).map(encoder.encode)
        for text, encoded in zip(_default_texts, pipe):
            self.assertEqual(encoded, encoder.encode(text))
        
        self.assertEqual(
            tf.function(encoder.multi_encode)(tf.reshape(_default_texts, [-1])),
            encoder.multi_encode(_default_texts)
        )

        pipe = tf.data.Dataset.from_tensor_slices(tf.reshape(_default_texts, [-1, 1])).map(
            encoder.multi_encode
        )
        for text, encoded in zip(_default_texts, pipe):
            self.assertEqual(encoded, encoder.multi_encode([text]))

        pipe = tf.data.Dataset.from_tensor_slices(_default_texts).map(
            lambda txt: encoder.format('Text : {text}', text = txt)
        )
        for text, encoded in zip(_default_texts, pipe):
            self.assertEqual(encoded, encoder.encode('Text : {}'.format(text)))
            self.assertEqual(encoded, encoder.format('Text : {text}', text = text))
        
        self.assertEqual(
            tf.function(encoder.multi_format)('Text : {text}', text = _default_texts),
            encoder.multi_format('Text : {text}', text = _default_texts)
        )

        pipe = tf.data.Dataset.from_tensor_slices([_default_texts]).map(
            lambda txt: encoder.multi_format('Text : {text}', text = txt)
        )
        for text, encoded in zip([_default_texts], pipe):
            self.assertEqual(encoded, encoder.multi_format('Text : {text}', text = text))
    
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
