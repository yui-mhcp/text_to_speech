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
import glob
import numpy as np
import pandas as pd

from absl.testing import parameterized

from . import CustomTestCase, data_dir
from utils.file_utils import is_path, path_to_unix, expand_path, hash_file, load_data, dump_data

class TestFPathProcessing(CustomTestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ('root', '/', os.path.exists('/')),
        ('current', '.', True),
        ('parent', '..', True),
        ('home', '~', True),
        ('home_formatted', '~/*', True),
        ('folder', data_dir, True),
        ('file', os.path.join(data_dir, 'audio_test.wav'), True),
        ('unix_format', os.path.join(data_dir, '*.wav'), True),
        
        ('none', None, False),
        ('empty', '', False),
        ('wrong', ' ', False),
        ('wrong_symbol', '?%%µ*%`"', False),
        ('py_format', os.path.join(data_dir, 'audio_{}.wav'), False)
    )
    def test_is_path(self, path, target):
        self.assertEqual(target, is_path(path))
    
    def test_hash_file(self):
        self.assertEqual(
            hash_file(os.path.join(data_dir, 'audio_test.wav')),
            hash_file(os.path.join(data_dir, 'audio_test.wav'), 1024)
        )
    
    def test_expand_path(self):
        self.assertEqual([], expand_path(None))
        self.assertEqual([], expand_path(''))
        
        files = [f for f in glob.glob(os.path.join(data_dir, '*')) if os.path.isfile(f)]
        files_rec = sorted(files + glob.glob(os.path.join(data_dir, '**', '*')))
        self.assertEqual(files, expand_path(data_dir, recursive = False, unix = False))
        self.assertEqual(
            [f.replace(os.path.sep, '/') for f in files],
            expand_path(data_dir, recursive = False, unix = True)
        )
        self.assertEqual(
            set(f.replace(os.path.sep, '/') for f in files_rec),
            set(expand_path(data_dir, recursive = True, unix = True))
        )

        
class TestFilesIO(CustomTestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ('txt', 'txt', None),
        ('md', 'md', None),
        ('py', 'py', None),
        ('json', 'json', {"a" : 1, "b" : 2, "c" : True, "d" : "Hello World !", "e" : {"aa" : 1.}, "f" : [1, 2., "3"]}),
        ('npy', 'npy', np.arange(5, dtype = 'int32')),
        ('csv', 'csv', pd.DataFrame([{'a' : 1, 'b' : 2}, {'a' : 2, 'b' : 3}])),
        ('tsv', 'tsv', pd.DataFrame([{'a' : 1, 'b' : 2}, {'a' : 2, 'b' : 3}]))
    )
    def test_load(self, ext, target):
        path = os.path.join(data_dir, 'files', 'test.' + ext)
        if target is None:
            with open(path, 'r', encoding = 'utf-8') as file:
                target = file.read()
        
        self.assertEqual(target, load_data(path))
        pass
    
    @parameterized.named_parameters(
        ('txt', 'txt', None),
        ('md', 'md', None),
        ('py', 'py', None),
        ('json', 'json', {"a" : 1, "b" : 2, "c" : True, "d" : "Hello World !", "e" : {"aa" : 1.}, "f" : [1, 2., "3"]}),
        ('npy', 'npy', np.arange(5, dtype = 'int32')),
        ('csv', 'csv', pd.DataFrame([{'a' : 1, 'b' : 2}, {'a' : 2, 'b' : 3}])),
        ('tsv', 'tsv', pd.DataFrame([{'a' : 1, 'b' : 2}, {'a' : 2, 'b' : 3}]))
    )
    def test_dump(self, ext, data):
        tar_file = os.path.join(data_dir, 'files', 'test.' + ext)
        tmp_file = os.path.join(data_dir, 'files', 'test-dump.' + ext)
        
        if data is None:
            with open(tar_file, 'r', encoding = 'utf-8') as f:
                data = f.read()
        
        try:
            dump_data(tmp_file, data)
            self.assertTrue(os.path.exists(tmp_file))
            mode = 'rb' if ext.endswith('npy') else 'r'
            with open(tmp_file, mode) as f1, open(tar_file, mode) as f2:
                target  = f2.read()
                value   = f1.read()
                
            self.assertEqual(target, value)
        finally:
            if os.path.exists(tmp_file): os.remove(tmp_file)
            
