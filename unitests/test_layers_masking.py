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

import keras
import unittest
import numpy as np
import keras.ops as K

from keras import layers
from absl.testing import parameterized

from unitests import CustomTestCase
try:
    from custom_layers import CustomEmbedding, masked_1d
    from custom_architectures import get_architecture
    err = None
except Exception as e:
    err = e

@unittest.skipIf(err is not None, 'The module import failed due to {}'.format(err))
class TestMaskedConv1D(CustomTestCase, parameterized.TestCase):
    def setUp(self):
        self.inputs = K.arange(1, 32, dtype = 'float32')[None]
    
    @parameterized.product(
        n_layers    = (1, 2),
        n_padding   = (1, 4, 9),
        kernel_size = list(range(1, 6)),
        strides     = list(range(1, 4)),
        padding     = ('same', 'valid'),
        dilation_rate   = list(range(1, 2))
    )
    def test_layer(self, n_layers, n_padding, padding, ** kwargs):
        self.padded = K.pad(self.inputs, [(0, 0), (0, 16)], constant_values = 0.)

        model = keras.Sequential([
            layers.Input(shape = (None, ), dtype = 'int32'),
            CustomEmbedding(64, 512, mask_value = 0)
        ])
        _padding = padding
        for _ in range(n_layers):
            if padding == 'same' and kwargs['kernel_size'] % 2 == 1:
                _padding = 'valid'
                k_half = kwargs['kernel_size'] // 2
                model.add(masked_1d.MaskedZeroPadding1D((k_half, k_half)))
            model.add(masked_1d.MaskedConv1D(512, padding = _padding, ** kwargs))
        
        with keras.device('cpu'):
            out1 = model(self.inputs)
        out_length  = out1.shape[1]
        
        if n_layers == 1:
            self.assertEqual(
                masked_1d._compute_new_len(self.inputs.shape[1], padding = padding, ** kwargs),
                out_length
            )
        if getattr(out1, '_keras_mask', None) is not None:
            self.assertTrue(np.all(K.convert_to_numpy(out1._keras_mask)))
        
        with keras.device('cpu'):
            out2 = model(self.padded)
        
        if getattr(out2, '_keras_mask', None) is not None:
            self.assertTrue(np.all(K.convert_to_numpy(out2._keras_mask[:, : out_length])))
            self.assertTrue(np.all(~K.convert_to_numpy(out2._keras_mask[:, out_length :])))

        self.assertEqual(out2[:, : out1.shape[1]], out1, max_err = 1e-5)
        self.assertEqual(out2[:, out1.shape[1] :], K.zeros_like(out2)[:, out1.shape[1] :])
        
