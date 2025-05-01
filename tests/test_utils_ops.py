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

import keras
import unittest
import numpy as np
import keras.ops as K

from absl.testing import parameterized

from . import CustomTestCase, is_tensorflow_available
from utils.keras.ops import Ops

class TestOps(CustomTestCase, parameterized.TestCase):
    def test_init(self):
        fn = Ops('shape')
        
        self.assertFalse(fn.built)
        self.assertTrue(fn.numpy_fn is not None)
        self.assertEqual(fn.name, 'shape')
        
        self.assertTrue(fn.numpy_fn is np.shape, str(fn.numpy_fn))
        self.assertTrue(fn.keras_fn is K.shape, str(fn.keras_fn))
    
    @parameterized.named_parameters(
        ('tuple', (0, 1, 2)),
        ('list',  [0, 1, 2]),
        ('range', range(3)),
        ('array', np.arange(3)),
        ('tensor', K.arange(3))
    )
    def test_call_simple(self, x):
        fn = Ops('sum')
        
        self.assertEqual(3, fn(x))
        if not K.is_tensor(x):
            self.assertFalse(fn.built)
            self.assertArray(fn(x))
        else:
            self.assertTrue(fn.built)
            self.assertTensor(fn(x))

    @parameterized.named_parameters(
        ('tuple', (0, 1, 2)),
        ('list',  [0, 1, 2]),
        ('range', range(3)),
        ('array', np.arange(3)),
        ('tensor', K.arange(3))
    )
    def test_disable_np(self, x):
        fn = Ops('sum', disable_np = True)
        
        out = fn(x)
        self.assertEqual(3, out)
        self.assertTensor(out)
        self.assertTrue(fn.built)
        self.assertTrue(fn.numpy_fn is None, str(fn.numpy_fn))
        
    @unittest.skipIf(keras.backend.backend() != 'tensorflow', 'Backend should be `tensorflow`')
    def test_tensorflow_shortcut(self):
        import tensorflow as tf
        
        fn = Ops('shape')
        fn.build()
        
        self.assertTrue(fn.keras_fn is K.shape)
        self.assertTrue(fn.call_keras_fn is fn.keras_fn)
        
        fn = Ops(
            'sum',
            tensorflow_fn = 'reduce_sum',
            keras_fn = lambda *_: self.fail('keras should not be called')
        )
        fn.build()
        
        self.assertTrue(fn.tensorflow_fn is tf.reduce_sum, str(fn.tensorflow_fn))
        self.assertTrue(fn.call_keras_fn is tf.reduce_sum, fn.call_keras_fn)
        self.assertEqual(3, fn(tf.range(3)))
        
        self.assertArray(fn(range(3)))
        self.assertTensor(fn(tf.range(3)))
    
    @unittest.skipIf(
        keras.backend.backend() == 'tensorflow' or not is_tensorflow_available(),
        'Backend should not be `tensorflow`'
    )
    def test_graph_with_other_backend(self):
        import tensorflow as tf
        
        def foo(x):
            return fn(x)
        
        @tf.function(reduce_retracing = True)
        def tf_graph_foo(x):
            return foo(x)
        
        fn = Ops('sum')
        fn.build()
        
        self.assertFalse(
            fn.call_keras_fn is fn.keras_fn, 'The tensorflow-graph check should be enabled'
        )
        
        # the function should behave normally for eager execution
        self.assertArray(foo(range(3)))
        self.assertArray(foo(np.arange(3)))
        self.assertTensor(foo(K.arange(3)))
        
        # the argument should be treated as a constant by the tensorflow graph
        #self.assertArray(tf_graph_foo(range(3)))
        #self.assertArray(tf_graph_foo(np.arange(3)))
        
        # the operation should use tensorflow functions when executed in a `tf.function`
        self.assertTfTensor(tf_graph_foo(tf.range(3)))
        self.assertTfTensor(tf_graph_foo(tf.range(4)))
        self.assertTfTensor(tf_graph_foo(tf.range(5)))
        
        try:
            tf_graph_foo(K.arange(3))
            self.fail('The function should raise an exception as it should call `tensorflow` function on a non-tensorflow `Tensor`')
        except:
            pass
