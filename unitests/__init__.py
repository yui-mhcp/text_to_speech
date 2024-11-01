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
import keras
import unittest
import warnings
import threading
import numpy as np
import keras.ops as K

from functools import cache, wraps

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import load_data, dump_data, is_equal, get_fn_name

data_dir    = os.path.join(os.path.dirname(__file__), '__data')
reproductibility_dir    = os.path.join(os.path.dirname(__file__), '__reproduction')

class TimeoutException(Exception):
    pass

def timeout(t):
    def wrapper(fn):
        @wraps(fn)
        def inner(self):
            thread = threading.Thread(target = fn, args = (self, ), daemon = True)
            thread.start()
            thread.join(timeout = t)
            if thread.is_alive(): self.fail('timeout !')
        return inner
    return wrapper

@cache
def get_graph_function(fn):
    import tensorflow as tf
    return tf.function(fn, reduce_retracing = True)

@cache
def get_xla_function(fn):
    import tensorflow as tf
    return tf.function(fn, reduce_retracing = True, jit_compile = True)

def convert_to_tf_tensor(x):
    import tensorflow as tf
    if K.is_tensor(x):
        if keras.backend.backend() == 'tensorflow': return x
        return tf.convert_to_tensor(K.convert_to_numpy(x))
    elif isinstance(x, np.ndarray):
        return tf.convert_to_tensor(x)
    return x

_graph_failed = {}

class CustomTestCase(unittest.TestCase):
    def assertEqual(self, value, target, msg = None, max_err = 1e-6, ** kwargs):
        eq, err_msg = is_equal(target, value, max_err = max_err, ** kwargs)
        self.assertTrue(eq, '{}{}'.format((msg + '\n') if msg else '', err_msg))
    
    def assertNotEqual(self, value, target, max_err = 1e-6, ** kwargs):
        eq, err_msg = is_equal(target, value, max_err = max_err, ** kwargs)
        self.assertFalse(eq, 'Values should be different but are equal')
    
    def assertReproductible(self, value, file, max_err = 1e-3, ** kwargs):
        file = os.path.join(reproductibility_dir, file)
        if not os.path.exists(file):
            os.makedirs(reproductibility_dir, exist_ok = True)
            dump_data(filename = file, data = value)
        self.assertEqual(load_data(file), value, max_err = max_err, ** kwargs)
    
    def assertTensor(self, x, nested = False):
        if nested:
            return keras.tree.map_structure(self.assertTensor, x)
        self.assertTrue(
            K.is_tensor(x), 'The function must return a `Tensor`, got {}'.format(type(x))
        )
    
    def assertTfTensor(self, x, nested = False):
        if nested:
            return keras.tree.map_structure(self.assertTfTensor, x)
        import tensorflow as tf
        self.assertTrue(
            tf.is_tensor(x), 'The function must return a `tf.Tensor`, got {}'.format(type(x))
        )

    def assertArray(self, x, nested = False):
        if nested:
            return keras.tree.map_structure(self.assertArray, x)
        import tensorflow as tf
        is_tensor = K.is_tensor(x) or tf.is_tensor(x)
        self.assertFalse(
            is_tensor, 'The function must return a `np.ndarray`, got {}'.format(type(x))
        )

    
    def assertTfPipeline(self, fn, inputs, target = None, expected_shape = None, ** kwargs):
        def fn_with_python_kwargs(* args):
            return fn(* args, ** kwargs)
        
        import tensorflow as tf
        
        inputs  = keras.tree.map_structure(convert_to_tf_tensor, inputs)
        ds      = tf.data.experimental.from_list([inputs]).map(fn_with_python_kwargs)
        
        if expected_shape is not None:
            if expected_shape is True:
                assert target is not None
                expected_shape = keras.tree.map_structure(
                    lambda x: getattr(x, 'shape', ()), target
                )
            
            shape = keras.tree.map_structure(lambda sign: sign.shape, ds.element_spec)
            self.assertTrue(
                shape == expected_shape,
                'Shape differs\n  Expected : {}\n  Found : {}'.format(expected_shape, shape)
            )
        
        if target is not None:
            self.assertEqual(next(iter(ds)), target)
        
    def assertGraphCompatible(self,
                              fn,
                              * args,
                              
                              _xla = False,
                              
                              target = None,
                              target_shape  = None,
                              nested = None,
                              
                              ** kwargs
                             ):
        if fn in _graph_failed: return
        
        import tensorflow as tf
        
        with self.subTest('{} compatible : {}'.format('XLA' if _xla else 'Graph', get_fn_name(fn))):
            if _xla:
                graph_fn    = get_xla_function(fn)
            else:
                graph_fn    = get_graph_function(fn)
            
            args    = keras.tree.map_structure(convert_to_tf_tensor, args)
            kwargs  = keras.tree.map_structure(convert_to_tf_tensor, kwargs)
            
            err = None
            try:
                result  = graph_fn(* args, ** kwargs)
            except Exception as e:
                err = e
            
            if err is not None:
                _graph_failed[fn] = True
                self.fail('The function does not support graph mode :\n{}'.format(err))
            
            if nested is None and target is not None:
                nested = isinstance(target, (list, tuple, dict))
            self.assertTfTensor(result, nested = nested)
            
            if target is not None:
                self.assertEqual(keras.tree.map_structure(lambda t: t.numpy(), result), target)
    
    def assertXLACompatible(self, fn, * args, ** kwargs):
        self.assertGraphCompatible(fn, * args, _xla = True ** kwargs)
