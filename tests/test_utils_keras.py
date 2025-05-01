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
from utils.keras import ops

class TestCustomOperation(CustomTestCase, parameterized.TestCase):
    def _test_ops(self, keras_ops, custom_ops, * args, target_type = None, ** kwargs):
        target  = keras_ops(* args, ** kwargs) if callable(keras_ops) else keras_ops
        value   = custom_ops(* args, ** kwargs)
        
        self.assertEqual(target, value)
        if target_type:
            check_fn = self.assertArray if target_type == 'array' else self.assertTensor
            if isinstance(target, (list, tuple)):
                [check_fn(v) for v in value]
            else:
                check_fn(value)
        
        self.assertGraphCompatible(custom_ops, * args, target = target, ** kwargs)
        
        return value

class TestRandomOps(CustomTestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ('beta', 'beta', 0., 1.),
        ('binomial', 'binomial', 10, 0.1),
        ('gamma', 'gamma', 0.5),
        ('normal', 'normal'),
        ('randint', 'randint', 5, 100),
        ('truncated_normal', 'truncated_normal'),
        ('uniform', 'uniform')
    )
    def test_tensor_creation(self, name, * args):
        fn = getattr(ops.random, name)
        self.assertTrue(fn.numpy_fn is None, 'numpy should be disabled')
        
        self.assertEqual(fn((5, 6), * args, seed = 0).shape, (5, 6))
        if keras.backend.backend() != 'torch':
            self.assertGraphCompatible(fn, (5, 6), * args, is_random = True, seed = 0)
        else:
            self.skipTest('The random operations are not usable in graph mode with `torch` backend')

    
    @parameterized.named_parameters(
        ('categorical', 'categorical', (4, 5), (4, 16), 5),
        ('dropout', 'dropout', (5, 6), (5, 6), 0.5),
        ('shuffle', 'shuffle', (5, 6), (5, 6), 0)
    )
    def test_tensor_modification(self, name, target_shape, input_shape, * args):
        fn = getattr(ops.random, name)
        self.assertTrue(fn.numpy_fn is None, 'numpy should be disabled')
        
        inp = keras.random.normal(input_shape)
        self.assertEqual(fn(inp, * args, seed = 0).shape, target_shape)
        if keras.backend.backend() != 'torch':
            self.assertGraphCompatible(fn, inp, * args, is_random = True, seed = 0)
        else:
            self.skipTest('The random operations are not usable in graph mode with `torch` backend')
    
class TestCoreOps(TestCustomOperation):
    @parameterized.product(
        operation   = ('array', 'empty', 'zeros', 'ones', 'full'),
        shape       = [(), (5, ), (5, 5)],
        dtype       = ('uint8', 'int32', 'float16', 'float32', 'bool')
    )
    def test_tensor_creation(self, operation, shape, dtype):
        fn = getattr(ops, operation)
        self.assertTrue(fn.numpy_fn is None, 'numpy should be disabled')
        
        args = (shape, )
        if operation == 'full': args += (5, )
        
        target  = getattr(K, operation)(* args, dtype = dtype)
        value   = fn(* args, dtype = dtype)

        self.assertTensor(value)
        
        graph_kwargs = {'target' : target}
        if operation != 'empty' or keras.backend.backend() == 'tensorflow':
            self.assertEqual(target, value)
        else:
            self.assertEqual(shape, tuple(value.shape))
            graph_kwargs['is_random'] = True
        
        self.assertGraphCompatible(fn, * args, dtype = dtype, ** graph_kwargs)

    @parameterized.product(
        (
            {'start' : 5, 'end' : None, 'step' : None},
            {'start' : 5, 'end' : 10, 'step' : None},
            {'start' : -5, 'end' : 10, 'step' : None},
            {'start' : 0, 'end' : 10, 'step' : 2},
            {'start' : 10, 'end' : 0, 'step' : -1}
        ),
        dtype = ('int32', 'float32')
    )
    def test_range(self, start, end = None, step = None, dtype = 'float32'):
        self.assertTrue(ops.arange.numpy_fn is None, 'numpy should be disabled')

        args    = [a for a in (start, end, step) if a is not None]
        self._test_ops(K.arange, ops.arange, * args, dtype = dtype, target_type = 'tensor')

    @parameterized.parameters('int32', 'bool', 'float32')
    def test_eye(self, dtype):
        self.assertTrue(ops.eye.numpy_fn is None, 'numpy should be disabled')

        self._test_ops(K.eye, ops.eye, 5, dtype = dtype, target_type = 'tensor')

    @parameterized.named_parameters(
        ('int', 1, ()),
        ('float', 1, ()),
        ('bool', True, ()),
        
        ('list', [1, 2], (2, )),
        ('nested_list', [[1], [2]], (2, 1)),
        
        ('array_1d', np.arange(3), (3, )),
        ('array_2d', np.ones((5, 5)), (5, 5)),
        
        ('tensor_1d', K.arange(3), (3, )),
        ('tensor_2d', K.ones((5, 5)), (5, 5))
    )
    def test_shape(self, data, target):
        shape = ops.shape(data)
        self.assertTrue(isinstance(shape, tuple), str(type(shape)))
        self.assertEqual(target, shape)
        
        self.assertGraphCompatible(
            ops.shape, data, target = target, is_tensor_output = hasattr(data, 'shape')
        )
        
        rank = ops.rank(data)
        self.assertTrue(isinstance(rank, int), str(type(rank)))
        self.assertEqual(len(target), rank)
        
        self.assertGraphCompatible(ops.rank, data, target = len(target))
    
    @parameterized.named_parameters([
        ('int',     0, 'int32'),
        ('float',   0., 'float32'),
        ('bool',    True, 'bool'),
        ('string',  'Hello World !', 'string'),
        ('list_int',    [0, 1, 2], 'int32'),
        ('list_float',  [0., 1., 2.], 'float32'),
        ('list_mixed',  [0, 1., 2.], 'int32'), # should take the type of the 1st item
        ('nested_list_int', [[0], [1]], 'int32'),
        ('nested_list_float',   [[0.], [1.]], 'float32'),
        
        ('array_bool',  np.zeros((5, ), dtype = 'bool'), 'bool'),
        ('array_uint8', np.zeros((5, ), dtype = 'uint8'), 'uint8'),
        ('array_int32', np.zeros((5, ), dtype = 'int32'), 'int32'),
        ('array_int64', np.zeros((5, ), dtype = 'int64'), 'int32'),
        ('array_float32',   np.zeros((5, ), dtype = 'float16'), 'float32'),
        ('array_float64',   np.zeros((5, ), dtype = 'float32'), 'float32'),
        
        ('tensor_bool', K.zeros((5, ), dtype = 'bool'), 'bool'),
        ('tensor_uint8',    K.zeros((5, ), dtype = 'uint8'), 'uint8'),
        ('tensor_int32',    K.zeros((5, ), dtype = 'int32'), 'int32'),
        ('tensor_float16',  K.zeros((5, ), dtype = 'float16'), 'float16'),
        ('tensor_float32',  K.zeros((5, ), dtype = 'float32'), 'float32'),
    ])
    def test_get_convertion_type(self, data, target):
        self.assertEqual(target, ops.dtype_to_str(ops.get_convertion_dtype(data)))

    @parameterized.named_parameters(
        ('integer', 1, False, False),
        ('float', 2.5, False, False),
        ('bool', True, False, False),
        ('string', 'Hello World !', False, False),
        ('list', [1, 'hello', True], False, False),
        ('array', np.ones((5, 5)), True, False),
        ('tensor', K.ones((5, 5)), True, True),
        ('list_of_tensors', [K.ones((5, 5))], False, False)
    )
    def test_is_tensor(self, value, is_array, is_tensor):
        self.assertEqual(is_array, ops.is_array(value))
        self.assertEqual(is_tensor, ops.is_tensor(value))

    @parameterized.parameters(
        'uint8', 'int32', 'int64', 'float', 'float16', 'float32', 'float64', 'bool'
    )
    def test_cast(self, dtype):
        tensor = K.zeros((5, 5), dtype = 'float32')
        array  = np.zeros((5, 5), dtype = 'float32')
        
        array_cast  = ops.cast(array, dtype)
        self.assertArray(array_cast)

        tensor_cast = ops.cast(tensor, dtype)
        self.assertTensor(tensor_cast)

        if dtype == 'float': dtype = keras.backend.floatx()
        self.assertEqual(ops.dtype_to_str(array_cast.dtype), dtype)

        if keras.backend.backend() != 'jax' or '64' not in dtype:
            self.assertEqual(ops.dtype_to_str(tensor_cast.dtype), dtype)
        
        if hasattr(np, dtype):
            self.assertEqual(
                ops.dtype_to_str(ops.cast(array, getattr(np, dtype)).dtype), dtype
            )
            if keras.backend.backend() != 'jax' or '64' not in dtype:
                self.assertEqual(
                    ops.dtype_to_str(ops.cast(tensor, getattr(np, dtype)).dtype), dtype
                )

        if dtype == 'float32':
            self.assertTrue(
                array is array_cast, 'The function creates a new instance for same dtype'
            )
            self.assertTrue(
                tensor is tensor_cast, 'The function creates a new instance for same dtype'
            )

        self.assertEqual(ops.is_int(tensor_cast), 'int' in dtype)
        self.assertEqual(ops.is_float(tensor_cast), 'float' in dtype)
        self.assertEqual(ops.is_numeric(tensor_cast), dtype != 'bool')
        self.assertEqual(ops.is_bool(tensor_cast), dtype == 'bool')

        self.assertEqual(ops.is_int(array_cast), 'int' in dtype)
        self.assertEqual(ops.is_float(array_cast), 'float' in dtype)
        self.assertEqual(ops.is_bool(array_cast), dtype == 'bool')
        self.assertEqual(ops.is_numeric(array_cast), dtype != 'bool')
    
        self.assertGraphCompatible(ops.cast, array, dtype, target = array_cast)

    @parameterized.product(
        data  = (np.ones((5, ), dtype = 'float16'), K.ones((5, ), dtype = 'float16'), (256, 256)),
        dtype = (None, 'int32', 'float', 'float16', 'float32')
    )
    def test_convert_to_numpy(self, data, dtype):
        array   = ops.convert_to_numpy(data, dtype)
        self.assertArray(array)
        if dtype not in (None, 'float'):
            self.assertEqual(dtype, array.dtype.name)
        
        if isinstance(data, np.ndarray) and dtype in (None, 'float', 'float16'):
            self.assertTrue(array is data, 'the function created a new array')
        
        if not isinstance(data, tuple):
            self.assertGraphCompatible(ops.convert_to_numpy, data, dtype, target = array)

    @parameterized.product(
        data  = (np.ones((5, ), dtype = 'float16'), K.ones((5, ), dtype = 'float16'), (256, 256)),
        dtype = (None, 'int32', 'float', 'float16', 'float32')
    )
    def test_convert_to_tensor(self, data, dtype):
        tensor  = ops.convert_to_tensor(data, dtype)
        self.assertTensor(tensor)
        if dtype not in (None, 'float'):
            self.assertEqual(dtype, ops.dtype_to_str(tensor.dtype))
            if not isinstance(data, tuple):
                self.assertGraphCompatible(ops.convert_to_tensor, data, dtype, target = tensor)

        if K.is_tensor(data) and dtype in (None, 'float', 'float16'):
            self.assertTrue(tensor is data, 'the function created a new tensor')
        
    
    @unittest.skipIf(not is_tensorflow_available(), '`tensorflow` is not available')
    @parameterized.product(
        data  = (np.ones((5, ), dtype = 'float16'), K.ones((5, ), dtype = 'float16'), (256, 256)),
        dtype = (None, 'int32', 'float', 'float16', 'float32')
    )
    def test_convert_to_tf_tensor(self, data, dtype):
        tensor  = ops.convert_to_tf_tensor(data, dtype)
        self.assertTfTensor(tensor)
        if dtype not in (None, 'float'):
            self.assertEqual(dtype, ops.dtype_to_str(tensor.dtype))
        
        if K.is_tensor(data) and dtype in (None, 'float', 'float16') and keras.backend.backend() == 'tensorflow':
            self.assertTrue(tensor is data, 'the function created a new tensor')

    @parameterized.named_parameters(
        ('array', np.arange(16).reshape(2, 4, 2).astype('int32')),
        ('tensor', K.reshape(K.arange(16), [2, 4, 2]))
    )
    def test_slice(self, data):
        target_type = 'tensor' if K.is_tensor(data) else 'array'
        self._test_ops(K.slice, ops.slice, data, [0, 0, 0], list(data.shape), target_type = target_type)
        self._test_ops(K.slice, ops.slice, data, [0, 0, 0], [1, 3, 2], target_type = target_type)
        self._test_ops(K.slice, ops.slice, data, [1, 2, 0], [1, 1, 2], target_type = target_type)
    
    @parameterized.named_parameters(
        ('array', np.arange(16).reshape(2, 4, 2).astype('int32')),
        ('tensor', K.reshape(K.arange(16), [2, 4, 2]))
    )
    def test_slice_update(self, data):
        target_type = 'tensor' if K.is_tensor(data) else 'array'

        update  = np.arange(4).reshape(2, 2, 1).astype('int32')
        value   = self._test_ops(
            K.slice_update, ops.slice_update, data, [0, 1, 0], update, target_type = target_type
        )
        
        if not K.is_tensor(data):
            self.assertTrue(value is data, 'The numpy slice update should be inplace')
    
    def test_stack(self):
        self._test_ops(
            K.stack, ops.stack, [0, 1, 2], axis = 0, target_type = 'array'
        )
        self._test_ops(
            K.stack, ops.stack, [np.array([1, 2]), np.array([2, 3])], axis = 0, target_type = 'array'
        )
        self._test_ops(
            K.stack, ops.stack, [np.array([1, 2]), np.array([2, 3])], axis = 1, target_type = 'array'
        )
        self._test_ops(
            K.stack, ops.stack, [np.array([[1], [2]]), np.array([[2], [3]])], axis = 0, target_type = 'array'
        )
        
        self._test_ops(
            K.stack, ops.stack, [K.array([1, 2]), K.array([2, 3])], axis = 0, target_type = 'tensor'
        )
        self._test_ops(
            K.stack, ops.stack, [K.array([1, 2]), K.array([2, 3])], axis = 1, target_type = 'tensor'
        )
        self._test_ops(
            K.stack, ops.stack, [K.array([[1], [2]]), K.array([[2], [3]])], axis = 0, target_type = 'tensor'
        )

    def test_unstack(self):
        self._test_ops(
            lambda x, axis: list(x), ops.unstack, np.arange(6).reshape((3, 2)), axis = 0, target_type = 'array'
        )
        self._test_ops(
            lambda x, axis: list(x.T), ops.unstack, np.arange(6).reshape((3, 2)), axis = 1, target_type = 'array'
        )
        
        self._test_ops(
            K.unstack, ops.unstack, K.reshape(np.arange(6), (3, 2)), axis = 0, target_type = 'tensor'
        )
        self._test_ops(
            K.unstack, ops.unstack, K.reshape(np.arange(6), (3, 2)), axis = 1, target_type = 'tensor'
        )
    

class TestNumpyOps(TestCustomOperation):
    @parameterized.named_parameters([
        (name, name) for name in ('norm', 'normalize', 'abs', 'sum', 'min', 'max', 'mean', 'std')
    ])
    def test_unary_operation(self, name):
        x = np.random.normal(size = (5, 6)).astype('float32')
        self._test_ops(
            getattr(K, name), getattr(ops, name), x, target_type = 'array'
        )
        self._test_ops(
            getattr(K, name), getattr(ops, name), K.convert_to_tensor(x), target_type = 'tensor'
        )
    
    @parameterized.named_parameters([
        (name, name) for name in ('add', 'subtract', 'multiply', 'divide', 'divide_no_nan')
    ])
    def test_binary_operation(self, name):
        x = np.random.normal(size = (5, 6)).astype('float32')
        y = np.random.normal(size = (1, 6)).astype('float32')
        self._test_ops(
            getattr(K, name), getattr(ops, name), x, y, target_type = 'array'
        )
        self._test_ops(
            getattr(K, name), getattr(ops, name), K.convert_to_tensor(x), y, target_type = 'tensor'
        )
        self._test_ops(
            getattr(K, name), getattr(ops, name), x, K.convert_to_tensor(y), target_type = 'tensor'
        )
        self._test_ops(
            getattr(K, name), getattr(ops, name), K.convert_to_tensor(x), K.convert_to_tensor(y), target_type = 'tensor'
        )

    def test_unique(self):
        self._test_ops(
            np.unique, ops.unique, np.array([1, 2, 1, 3, 5, 1, 3]), target_type = 'array'
        )
        self._test_ops(
            np.unique, ops.unique, K.array([1, 2, 1, 3, 5, 1, 3]), target_type = 'tensor'
        )
        
    @parameterized.product(
        shape = [(16, ), (4, 4), (4, 2, 2), (2, 2, 2, 2)],
        dtype   = ('uint8', 'int32', 'float32')
    )
    def test_take(self, shape, dtype):
        x = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
        
        for axis in range(-1, len(shape)):
            self._test_ops(
                K.take, ops.take, x, [1, 0], axis = axis, target_type = 'array'
            )
            self._test_ops(
                K.take, ops.take, K.convert_to_tensor(x), [1, 0], axis = axis, target_type = 'tensor'
            )

    @parameterized.product(
        shape = [(16, ), (4, 4), (4, 2, 2), (2, 2, 2, 2)],
        to_tensor   = (True, False)
    )
    def test_take_along_axis(self, shape, to_tensor):
        array = np.arange(np.prod(shape)).astype(np.int32)
        shuffled    = array.copy()
        np.random.shuffle(shuffled)
        array, shuffled = array.reshape(shape), shuffled.reshape(shape)
        if to_tensor:
            x       = K.convert_to_tensor(array)
            x_shuffled    = K.convert_to_tensor(shuffled)
        else:
            x, x_shuffled = array, shuffled

        for axis in range(-1, len(shape)):
            indices = np.argsort(shuffled, axis = axis).astype(np.int32)

            value   = ops.take_along_axis(x_shuffled, indices, axis = axis)
            target  = np.take_along_axis(shuffled, indices, axis = axis)
            self.assertEqual(value, target)
            if axis == 0:
                self.assertEqual(
                    ops.take_along_axis(x_shuffled, indices[: 2], axis = axis),
                    np.take_along_axis(shuffled, indices[: 2], axis = axis)
                )
            if to_tensor:
                self.assertTensor(value)
                self.assertGraphCompatible(
                    ops.take_along_axis, x_shuffled, indices, axis = axis, target = target
                )
            else:
                self.assertArray(value)

            _min        = np.argmin(array, axis = axis, keepdims = True)
            _max        = np.argmax(array, axis = axis, keepdims = True)
            _min_max    = np.concatenate([_min, _max], axis = axis)
            for indices in (_min, _max, _min_max):
                target  = np.take_along_axis(array, indices, axis = axis).astype(np.int32)
                self.assertEqual(
                    ops.take_along_axis(x, indices, axis = axis), target
                )

                self.assertGraphCompatible(
                    ops.take_along_axis, x, indices, axis = axis, target = target
                )
class TestMathOps(TestCustomOperation):
    @parameterized.named_parameters(
        * [(op, op) for op in ('sum', 'min', 'max', 'mean', 'argsort')]
    )
    def test_segment(self, name):
        segment_fn, np_fn = getattr(ops, 'segment_' + name), getattr(np, name)
        
        data        = np.random.uniform(0., 10., size = (64, 64)).astype('float32')
        data_t      = K.convert_to_tensor(data, 'float32')
        segment_ids = np.repeat(np.arange(8), 8).astype('int32')
        num_segments    = 8
        
        for axis in (0, 1):
            with self.subTest(axis = axis):
                target = data if axis == 0 else data.T
                
                target = [
                    np_fn(target[segment_ids == id_i], axis = 0)
                    for id_i in range(num_segments)
                ]
                if name == 'argsort':
                    target = np.concatenate(target, axis = 0).astype('int32')
                else:
                    target = np.array(target)
                
                if axis == 1: target = target.T
                
                self._test_ops(
                    target, segment_fn, data, segment_ids, num_segments, axis = axis,
                    target_type = 'array'
                )
                self._test_ops(
                    target, segment_fn, data_t, segment_ids, num_segments, axis = axis,
                    target_type = 'tensor'
                )
