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
import unittest

try:
    from transformers import AutoTokenizer
except:
    AutoTokenizer   = None
from absl.testing import parameterized
    
from utils.text import *
from unitests import CustomTestCase

test_with_dataset   = False

_default_texts  = [
    "Hello World !",
    "Bonjour à tous !",
    "1, 2, 3, 4, 5, 6, 7, 8, 9 et 10 !",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
]
dataset_texts = []
if test_with_dataset:
    from datasets import get_dataset
    dataset = get_dataset('squad', modes = 'valid')
    dataset_texts = dataset['question'].values.tolist() + dataset['context'].values.tolist()
    del dataset

sentences   = _default_texts + dataset_texts
sentences   = [' '.join(sent.split()) for sent in sentences]

@unittest.skipIf(AutoTokenizer is None, 'The `transformers` library is unavailable !')
class TestTokenizers(CustomTestCase, parameterized.TestCase):
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
    def test_transformers_encoder(self, model_name, use_sos_and_eos = None, add_eos = None):
        transformers_encoder    = AutoTokenizer.from_pretrained(model_name)
        encoder = TextEncoder.from_transformers_pretrained(model_name)

        if use_sos_and_eos is not None: encoder.use_sos_and_eos = use_sos_and_eos
        
        for sent in sentences:
            self.assertEqual(
                encoder.tokenize(sent),
                transformers_encoder.tokenize(sent)
            )
            self.assertEqual(
                encoder.encode(sent, add_eos = add_eos),
                transformers_encoder(sent)['input_ids']
            )

