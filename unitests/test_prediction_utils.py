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
import glob
import unittest

from absl.testing import parameterized

try:
    from models.utils import *
    err = None
except Exception as e:
    err = e

from utils import path_to_unix
from unitests import CustomTestCase, data_dir, reproductibility_dir

_data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

@unittest.skipIf(err is not None, 'Module failed to be loaded : {}'.format(err))
class TestPredictionUtils(CustomTestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ('single', ['test'], [0]),
        ('multi', ['test1', 'test2'], [0, 1]),
        ('dict', [{'text' : 'test1'}, {'text' : 'test2', 'other' : 'test_fail'}], [0, 1]),
        ('missing', [{'text' : 'test1'}, {'other' : 'failure'}], [0, 1]),
        ('duplicated', ['test1', 'test2', 'test1'], [0, 1]),
    )
    def test_preparation_simple(self, data, _indexes):
        results, inputs, indexes, entries, duplicates, filtered = prepare_prediction_results(
            data, {}, primary_key = 'text'
        )

        _inputs = [data[idx] for idx in indexes]
        _entries = [d if not isinstance(d, dict) else d.get('text', None) for d in _inputs]

        _duplicates = {}
        for i, d in enumerate(data):
            entry = d if isinstance(d, str) else d.get('text', None)
            if entry: _duplicates.setdefault(entry, []).append(i)

        self.assertEqual(
            (results, inputs, indexes, entries, duplicates, filtered),
            ([None] * len(data), _inputs, _indexes, _entries, _duplicates, [])
        )
    
    def test_preparation_transform(self):
        results, inputs, indexes, entries, duplicates, filtered = prepare_prediction_results(
            ['test', 'test\\a', {'filename' : 'test\\b'}],
            {},
            primary_key = 'filename',
            expand_files    = False,
            normalize_entry = lambda f: f.replace('\\', '/')
        )
        
        self.assertEqual(entries, ['test', 'test/a', 'test/b'])
        self.assertEqual(inputs, ['test', 'test\\a', {'filename' : 'test\\b'}])
        
        results, inputs, indexes, entries, duplicates, filtered = prepare_prediction_results(
            ['test\\a'],
            {'test/a' : 'success'},
            primary_key = 'filename',
            expand_files    = False,
            normalize_entry = lambda f: f.replace('\\', '/'),
            overwrite   = False,
            required_keys   = ()
        )
        
        self.assertEqual(len(inputs), 0)
        self.assertEqual(results, [('success', None)])

    @parameterized.named_parameters(
        ('file', [_data_files[0]], [path_to_unix(_data_files[0])]),
        ('directory', [data_dir], [path_to_unix(f) for f in _data_files]),
        (
            'formatted',
            [data_dir + '/*.jpg'],
            [path_to_unix(f) for f in glob.glob(data_dir + '/*.jpg')]
        ),
        (
            'duplicated',
            [_data_files[0], data_dir],
            _data_files
        )
    )
    def test_preparation_path(self, files, target):
        results, inputs, indexes, entries, duplicates, filtered = prepare_prediction_results(
            files, {}, primary_key = 'filename'
        )
        
        self.assertEqual(entries, target)

    @parameterized.named_parameters(
        ('simple', ['a', 'bb', 'c'], lambda t: len(t) > 1, ['a', 'c'], ['bb']),
        ('empty', ['a', 'b'], lambda t: True, [], ['a', 'b']),
        (
            'dict',
            ['a', 'bb', {'text' : 'ccc'}, 'dddd'],
            {
                'short' : lambda t: len(t) == 1,
                'long'  : lambda t: len(t) > 2,
                'failed': lambda t: False
            },
            ['bb'],
            {'short' : ['a'], 'long' : ['ccc', 'dddd']}
        )
    )
    def test_filters(self, entries, filters, valids, _filtered):
        results, inputs, indexes, entries, duplicates, filtered = prepare_prediction_results(
            entries, {}, primary_key = 'text', filters = filters
        )
        
        self.assertEqual(entries, valids)
        self.assertEqual(len(results), len(valids))
        self.assertEqual(filtered, _filtered)
