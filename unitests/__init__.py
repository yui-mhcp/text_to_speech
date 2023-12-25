# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
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
import unittest
import warnings

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import load_data, dump_data, is_equal

data_dir    = os.path.join(os.path.dirname(__file__), '__data')
reproductibility_dir    = os.path.join(os.path.dirname(__file__), '__reproduction')

class CustomTestCase(unittest.TestCase):
    def assertEqual(self, value, target, msg = None, ** kwargs):
        self.assertTrue(* is_equal(target, value, ** kwargs))
    
    def assertReproductible(self, value, file):
        file = os.path.join(reproductibility_dir, file)
        if not os.path.exists(file):
            os.makedirs(reproductibility_dir, exist_ok = True)
            dump_data(filename = file, data = value)
        self.assertEqual(load_data(file), value)
    
