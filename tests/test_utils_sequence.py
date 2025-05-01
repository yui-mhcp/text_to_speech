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

import numpy as np
import keras.ops as K

from absl.testing import parameterized

from . import CustomTestCase
from utils.sequence_utils import pad_batch, pad_to_multiple

class TestSequences(CustomTestCase, parameterized.TestCase):
    def test_pad_batch_simple(self):
        self.assertEqual(
            np.array([[1, 2, 0], [1, 2, 3]]), pad_batch([[1, 2], [1, 2, 3]])
        )
        self.assertEqual(
            np.array([[1, 2, 3], [1, 2, 0]]), pad_batch([[1, 2, 3], [1, 2]])
        )

        self.assertEqual(
            np.array([[1, 2, -1], [1, 2, 3]]), pad_batch([[1, 2], [1, 2, 3]], pad_value = -1.)
        )
        self.assertEqual(
            np.array([[0, 1, 2], [1, 2, 3]]), pad_batch([[1, 2], [1, 2, 3]], pad_mode = 'before')
        )

    @parameterized.named_parameters(
        ('vectors', (4, 5), [(1, ), (2, ), (5, ), (4, )]),
        ('matrix',  (4, 8, 8), [(2, 3), (3, 4), (5, 8), (8, 3)]),
        ('images',  (4, 32, 32, 3), [(16, 16, 3), (16, 16, 3), (32, 32, 3), (32, 32, 3)])
    )
    def test_pad_batch_array(self, target_shape, shapes):
        inputs  = [np.ones(s) for s in shapes]
        batch   = pad_batch(inputs)
        self.assertEqual(target_shape, batch.shape)
        for s, inp, b in zip(shapes, inputs, batch):
            self.assertEqual(
                inp, b[tuple(slice(0, dim) for dim in s)]
            )

    @parameterized.named_parameters(
        ('vectors', (4, 5), [(1, ), (2, ), (5, ), (4, )]),
        ('matrix',  (4, 8, 8), [(2, 3), (3, 4), (5, 8), (8, 3)]),
        ('images',  (4, 32, 32, 3), [(16, 16, 3), (16, 16, 3), (32, 32, 3), (32, 32, 3)])
    )
    def test_pad_batch_tensors(self, target_shape, shapes):
        inputs  = [K.ones(s) for s in shapes]
        batch   = pad_batch(inputs)
        self.assertEqual(target_shape, batch.shape)
        for s, inp, b in zip(shapes, inputs, batch):
            self.assertEqual(
                inp, b[tuple(slice(0, dim) for dim in s)]
            )
