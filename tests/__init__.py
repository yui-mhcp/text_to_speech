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
import sys
import unittest
import warnings
import numpy as np

from functools import cache, wraps

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import load_data, dump_data, is_equal

data_dir    = os.path.join(os.path.dirname(__file__), 'data')
reproductibility_dir    = os.path.join(os.path.dirname(__file__), '__reproduction')

@cache
def is_tensorflow_available():
    """ Return whether `tensorflow` is available without importing it ! """
    if 'tensorflow' in sys.modules: return True
    
    for finder in sys.meta_path:
        spec = finder.find_spec('tensorflow', None)
        if spec is not None: return True
    return False

@cache
def get_graph_function(fn, reduce_retracing = True, jit_compile = False):
    import tensorflow as tf
    
    @tf.function(reduce_retracing = reduce_retracing, jit_compile = jit_compile)
    def foo(* args, ** kwargs):
        return fn(* args, ** kwargs)
    
    foo.__name__ = fn.name if hasattr(fn, 'name') else fn.__name__
    return foo

def get_xla_function(fn):
    return get_graph_function(fn, jit_compile = True)

def convert_to_tf_tensor(x):
    import keras
    import keras.ops as K
    import tensorflow as tf
    
    if K.is_tensor(x):
        if keras.backend.backend() == 'tensorflow': return x
        return tf.convert_to_tensor(K.convert_to_numpy(x))
    elif isinstance(x, np.ndarray):
        return tf.convert_to_tensor(x)
    return x

_graph_failed = {}

class CustomTestCase(unittest.TestCase):
    def assertEqual(self, target, value, msg = None, max_err = 1e-6, ** kwargs):
        eq, err_msg = is_equal(target, value, max_err = max_err, ** kwargs)
        self.assertTrue(eq, '{}{}'.format((msg + '\n') if msg else '', err_msg))
    
    def assertNotEqual(self, target, value, max_err = 1e-6, ** kwargs):
        eq, err_msg = is_equal(target, value, max_err = max_err, ** kwargs)
        self.assertFalse(eq, 'Values should be different but are equal')
    
    def assertReproductible(self, value, file, max_err = 1e-6, ** kwargs):
        file = os.path.join(reproductibility_dir, file)
        if not os.path.exists(file):
            os.makedirs(reproductibility_dir, exist_ok = True)
            dump_data(filename = file, data = value)
        self.assertEqual(load_data(file), value, max_err = max_err, ** kwargs)

    def assertArray(self, x):
        self.assertTrue(
            isinstance(x, (np.ndarray, np.integer, np.floating)),
            'The function must return a `np.ndarray`, got {}'.format(type(x))
        )

    def assertTensor(self, x):
        import keras.ops as K
        
        self.assertTrue(
            K.is_tensor(x), 'The function must return a `Tensor`, got {}'.format(type(x))
        )
    
    def assertTfTensor(self, x):
        import tensorflow as tf
        
        self.assertTrue(
            tf.is_tensor(x), 'The function must return a `tf.Tensor`, got {}'.format(type(x))
        )
    
    def assertGraphCompatible(self,
                              fn,
                              * args,
                              
                              target    = None,
                              is_random = False,
                              is_tensor_output  = True,
                              
                              jit_compile   = False,
                              
                              ** kwargs
                             ):
        if fn in _graph_failed: return
        elif not is_tensorflow_available():
            self.skipTest('`tensorflow` should be available')
        
        import keras
        import tensorflow as tf
        
        name = fn.name if hasattr(fn, 'name') else fn.__name__
        with self.subTest('{} compatible : {}'.format('XLA' if jit_compile else 'Graph', name)):
            graph_fn = get_graph_function(fn, jit_compile = jit_compile)
            
            tf_args    = keras.tree.map_structure(convert_to_tf_tensor, args)
            tf_kwargs  = keras.tree.map_structure(convert_to_tf_tensor, kwargs)
            
            try:
                result  = graph_fn(* tf_args, ** tf_kwargs)
            except Exception as e:
                result = e
            
            if isinstance(result, Exception):
                _graph_failed[fn] = True
                self.fail('The function does not support graph mode :\n{}'.format(result))
            
            if target is None:
                target = fn(* args, ** kwargs)
            
            if is_tensor_output:
                if isinstance(result, (list, tuple)):
                    self.assertTrue(
                        isinstance(result, type(target)), 'Expected : {} - got : {}'.format(type(target), result)
                    )
                    [self.assertTfTensor(res) for res in result]
                else:
                    self.assertTfTensor(result)
            else:
                self.assertTrue(
                    isinstance(result, type(target)), 'Expected : {} - got : {}'.format(type(target), result)
                )
            
            if not is_random:
                self.assertEqual(target, keras.tree.map_structure(lambda t: t.numpy(), result))
            else:
                self.assertEqual(
                    keras.tree.map_structure(lambda t: tuple(t.shape), target),
                    keras.tree.map_structure(lambda t: tuple(t.shape), result)
                )
    
    def assertXLACompatible(self, fn, * args, ** kwargs):
        self.assertGraphCompatible(fn, * args, _xla = True ** kwargs)


