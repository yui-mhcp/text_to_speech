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

from absl.testing import parameterized

from utils.keras_utils import ops
from unitests import CustomTestCase

class TestCustomOps(CustomTestCase, parameterized.TestCase):
    @parameterized.product(
        operation   = ('array', 'empty', 'zeros', 'ones', 'full'),
        shape       = [(), (5, ), (5, 5), (5, 5, 5)],
        dtype       = ('uint8', 'int32', 'float16', 'float32', 'bool')
    )
    def test_tensor_creation(self, operation, shape, dtype):
        self.assertTrue(hasattr(ops, operation), 'Operator {} is missing'.format(operation))
        
        fn = getattr(ops, operation)
        self.assertTrue(getattr(fn, 'numpy_function', None) is None, 'numpy should be disabled')
        
        args = (shape, )
        if operation == 'full': args += (5, )
        
        target  = getattr(K, operation)(* args, dtype = dtype)
        value   = fn(* args, dtype = dtype)

        self.assertTensor(value)
        if operation != 'array':
            self.assertEqual(value.shape, shape)
        
        if operation != 'empty' or keras.backend.backend() == 'tensorflow':
            self.assertEqual(value, target)
            self.assertGraphCompatible(fn, * args, dtype = dtype, target = target)
                        

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
        self.assertTrue(hasattr(ops, 'arange'), 'Operator `arange` is missing')
        self.assertTrue(ops.arange.numpy_function is None, 'numpy should be disabled')

        args    = [a for a in (start, end, step) if a is not None]
        target  = K.arange(* args, dtype = dtype)
        value   = ops.arange(* args, dtype = dtype)
        
        self.assertTensor(value)
        self.assertEqual(value, target)
        self.assertGraphCompatible(ops.arange, * args, dtype = dtype, target = target)

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
        self.assertEqual(ops.is_array(value), is_array)
        self.assertEqual(ops.is_tensor(value), is_tensor)

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

        if keras.backend.backend() == 'jax' and '64' in dtype: return
        self.assertEqual(ops.dtype_to_str(tensor_cast.dtype), dtype)
        
        if hasattr(np, dtype):
            self.assertEqual(
                ops.dtype_to_str(ops.cast(tensor, getattr(np, dtype)).dtype), dtype
            )
            self.assertEqual(
                ops.dtype_to_str(ops.cast(array, getattr(np, dtype)).dtype), dtype
            )

        if dtype == 'float32':
            self.assertTrue(
                tensor is tensor_cast, 'The function creates a new instance for same dtype'
            )
            self.assertTrue(
                array is array_cast, 'The function creates a new instance for same dtype'
            )

        self.assertEqual(ops.is_int(tensor_cast), 'int' in dtype)
        self.assertEqual(ops.is_float(tensor_cast), 'float' in dtype)
        self.assertEqual(ops.is_numeric(tensor_cast), dtype != 'bool')
        self.assertEqual(ops.is_bool(tensor_cast), dtype == 'bool')

        self.assertEqual(ops.is_int(array_cast), 'int' in dtype)
        self.assertEqual(ops.is_float(array_cast), 'float' in dtype)
        self.assertEqual(ops.is_bool(array_cast), dtype == 'bool')
        self.assertEqual(ops.is_numeric(array_cast), dtype != 'bool')
    
    @parameterized.parameters([
        (0, 'int32'),
        (0., 'float32'),
        (True, 'bool'),
        ('Hello World !', 'string'),
        ([0, 1, 2], 'int32'),
        ([0., 1., 2.], 'float32'),
        ([0, 1., 2.], 'int32'), # should take the type of the 1st item
        ([[0], [1]], 'int32'),
        (np.zeros((5, ), dtype = 'uint8'), 'uint8'),
        (np.zeros((5, ), dtype = 'int32'), 'int32'),
        (np.zeros((5, ), dtype = 'int64'), 'int32'),
        (np.zeros((5, ), dtype = 'float16'), 'float32'),
        (np.zeros((5, ), dtype = 'float32'), 'float32'),
        (np.zeros((5, ), dtype = 'bool'), 'bool'),
        (K.zeros((5, ), dtype = 'uint8'), 'uint8'),
        (K.zeros((5, ), dtype = 'int32'), 'int32'),
        (K.zeros((5, ), dtype = 'float16'), 'float16'),
        (K.zeros((5, ), dtype = 'float32'), 'float32'),
        (K.zeros((5, ), dtype = 'bool'), 'bool'),
    ])
    def test_get_convertion_type(self, data, target):
        self.assertEqual(ops.get_convertion_dtype(data), target)
    
    @parameterized.product(
        data  = (np.ones((5, ), dtype = 'float16'), K.ones((5, ), dtype = 'float16'), (256, 256)),
        dtype = (None, 'int32', 'float', 'float16', 'float32')
    )
    def test_convert_to(self, data, dtype):
        # the test on dtype is not applied for `float` because defaul floatx() is `float32`
        # the objective is to not create a new instance if the input type is already a float
        # even if the actual float is not the default one (in this case, float16)
        array   = ops.convert_to_numpy(data, dtype)
        self.assertArray(array)
        if dtype not in (None, 'float'):
            self.assertTrue(array.dtype.name == dtype)
        if isinstance(data, np.ndarray) and dtype in (None, 'float', 'float16'):
            self.assertTrue(array is data, 'the function created a new array')
        
        tensor  = ops.convert_to_tensor(data, dtype)
        self.assertTensor(tensor)
        if dtype not in (None, 'float'):
            self.assertTrue(ops.dtype_to_str(tensor.dtype) == dtype)
        if K.is_tensor(data) and dtype in (None, 'float', 'float16'):
            self.assertTrue(tensor is data, 'the function created a new tensor')
        
        try:
            tf_tensor = ops.convert_to_tf_tensor(data, dtype)
            self.assertTfTensor(tensor)
            if dtype not in (None, 'float'):
                self.assertTrue(ops.dtype_to_str(tensor.dtype) == dtype)
        except ops.TensorflowNotAvailable:
            self.skip('Tensorflow is not available, skipping the convert_to_tf_tensor test')
    
    @parameterized.parameters(([16], ), ([4, 4], ), ([4, 2, 2], ), ([2, 2, 2, 2], ))
    def test_shapes(self, shape):
        tensor  = K.arange(16, dtype = 'int32')
        array   = np.arange(16).astype(np.int32)
        
        target = np.reshape(array, shape)
        for x in (tensor, array):
            _assert_valid_type = self.assertTensor if x is tensor else self.assertArray

            reshaped    = ops.reshape(x, shape)

            self.assertEqual(reshaped, target)
            self.assertEqual(ops.shape(reshaped), shape)
            self.assertEqual(ops.rank(reshaped), len(shape))
            self.assertEqual(ops.size(reshaped), 16)
            _assert_valid_type(reshaped)

            self.assertGraphCompatible(ops.shape, reshaped, target = shape)
            self.assertGraphCompatible(ops.rank, reshaped, target = len(shape))
            self.assertGraphCompatible(ops.size, reshaped, target = 16)
            self.assertGraphCompatible(ops.reshape, x, shape, target = target)
            
            with self.subTest('expand_dims', is_tensor = x is tensor):
                for axis in range(-1, len(shape) + 1):
                    expanded = ops.expand_dims(reshaped, axis)
                    self.assertEqual(expanded, np.expand_dims(target, axis = axis))
                    _assert_valid_type(expanded)

                    self.assertGraphCompatible(
                        ops.expand_dims, reshaped, axis = axis, target = expanded
                    )

            with self.subTest('unstack_and_stack', is_tensor = x is tensor):
                for axis in range(-1, len(shape)):
                    unstacked = ops.unstack(reshaped, axis = axis)
                    self.assertEqual(len(unstacked), shape[axis])
                    self.assertEqual(unstacked, ops.array_ops._np_unstack(target, axis = axis))
                    _assert_valid_type(unstacked, nested = True)

                    rebuilt = ops.stack(unstacked, axis = axis)
                    self.assertEqual(rebuilt, reshaped)
                    _assert_valid_type(rebuilt)

                    self.assertGraphCompatible(
                        ops.unstack, reshaped, num = len(unstacked), axis = axis, target = unstacked
                    )
                    self.assertGraphCompatible(
                        ops.stack, unstacked, axis = axis, target = reshaped
                    )

            with self.subTest('split_and_concat', shape = shape, is_tensor = x is tensor):
                for axis in range(-1, len(shape)):
                    splitted = ops.split(reshaped, 2, axis = axis)
                    self.assertEqual(len(splitted), 2)
                    self.assertEqual(splitted, np.split(target, 2, axis = axis))
                    _assert_valid_type(splitted, nested = True)

                    rebuilt = ops.concatenate(splitted, axis = axis)
                    self.assertEqual(rebuilt, reshaped)
                    _assert_valid_type(rebuilt)

                    self.assertGraphCompatible(
                        ops.split, reshaped, 2, axis = axis, target = splitted
                    )
                    self.assertGraphCompatible(
                        ops.concat, splitted, axis = axis, target = reshaped
                    )
                    

    @parameterized.named_parameters(
        ('test_array', np.arange(16).reshape(2, 4, 2).astype('int32')),
        ('test_tensor', K.reshape(K.arange(16), [2, 4, 2]))
    )
    def test_slicing(self, data):
        self.assertEqual(ops.slice(data, [0, 0, 0], list(data.shape)), data)
        self.assertEqual(ops.slice(data, [0, 0, 0], [1, 3, 2]), data[: 1, :3, :2])
        self.assertEqual(ops.slice(data, [1, 2, 0], [1, 1, 2]), data[1:2, 2:3, :])
        if K.is_tensor(data):
            self.assertTensor(ops.slice(data, [0, 0, 0], list(data.shape)))
        else:
            self.assertArray(ops.slice(data, [0, 0, 0], list(data.shape)))
        

        self.assertGraphCompatible(
            ops.slice, data, [0, 0, 0], [1, 3, 2], target = data[: 1, :3, :2]
        )
        
        update  = np.arange(4).reshape(2, 2, 1).astype('int32')
        updated = ops.update_slice(data, [0, 1, 0], update)
        
        self.assertEqual(updated[:2, 1:3, :1], update)
        
        if K.is_tensor(data):
            self.assertTensor(ops.slice(updated, [0, 0, 0], list(data.shape)))
        else:
            self.assertArray(ops.slice(updated, [0, 0, 0], list(data.shape)))
            self.assertTrue(updated is data, 'The numpy slice update should be inplace')

        self.assertGraphCompatible(
            ops.update_slice, data, [0, 1, 0], update, target = updated
        )

    @parameterized.product(
        method  = ('argmin', 'argmax', 'argsort'),
        shape = [(16, ), (4, 4), (4, 2, 2), (2, 2, 2, 2)],
        to_tensor   = (True, False)
    )
    def test_indexing(self, method, shape, to_tensor):
        array = np.arange(np.prod(shape))
        shuffled    = array.copy()
        np.random.shuffle(shuffled)
        array, shuffled = array.reshape(shape), shuffled.reshape(shape)
        if to_tensor:
            array       = K.convert_to_tensor(array)
            shuffled    = K.convert_to_tensor(shuffled)
        
        dtype = 'int32' if to_tensor else 'int64'
        for axis in  range(-1, len(shape)):
            self.assertEqual(
                getattr(ops, method)(array, axis = axis),
                K.convert_to_numpy(getattr(K, method)(array, axis = axis)).astype(dtype)
            )
            self.assertEqual(
                getattr(ops, method)(shuffled, axis = axis),
                K.convert_to_numpy(getattr(K, method)(shuffled, axis = axis)).astype(dtype)
            )

            if to_tensor:
                self.assertTensor(getattr(ops, method)(array, axis = axis))
                self.assertGraphCompatible(
                    getattr(ops, method), shuffled, axis = axis,
                    target = getattr(K, method)(shuffled, axis = axis)
                )
            else:
                self.assertArray(getattr(ops, method)(array, axis = axis))

    @parameterized.product(
        shape = [(16, ), (4, 4), (4, 2, 2), (2, 2, 2, 2)],
        to_tensor   = (True, False),
        dtype   = ('uint8', 'int32', 'float32')
    )
    def test_take(self, shape, to_tensor, dtype):
        arr = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
        x = K.convert_to_tensor(arr) if to_tensor else arr
        
        for axis in range(-1, len(shape)):
            value  = ops.take(x, [1, 0], axis = axis)
            target = np.take(arr, [1, 0], axis = axis)
            
            self.assertEqual(value, target)
            if to_tensor:
                self.assertTensor(value)
                self.assertGraphCompatible(
                    ops.take, x, [1, 0], axis = axis, target = target
                )
            else:
                self.assertArray(value)


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

