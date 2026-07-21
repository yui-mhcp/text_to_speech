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

""" Standalone, framework-agnostic assertions for the test-suite.

Every function here raises a plain ``AssertionError`` on failure (and
``unittest.SkipTest`` to skip), so it works identically inside a
``unittest.TestCase`` (see :class:`tests.CustomTestCase`) or inside a plain
``pytest`` function / fixture. This is the single source of truth : the
``CustomTestCase`` methods are thin wrappers delegating here.

The heavy imports (``keras`` / ``tensorflow``) stay lazy so importing this
module never pulls a deep-learning backend in.
"""

import os
import warnings
import numpy as np

from unittest import SkipTest

from utils import load_data, dump_data, is_equal
from ._helpers import (
    reproductibility_dir, is_tensorflow_available, get_graph_function, convert_to_tf_tensor,
    _graph_failed
)

# environment flag controlling whether a missing reproducibility golden is an
# error (CI / shared server) rather than being silently created (local dev).
STRICT_GOLDEN_ENV = 'TESTS_STRICT_GOLDEN'

def assert_equal(target, value, msg = None, *, max_err = 1e-6, ** kwargs):
    eq, err_msg = is_equal(target, value, max_err = max_err, ** kwargs)
    assert eq, '{}{}'.format((msg + '\n') if msg else '', err_msg)

def assert_not_equal(target, value, *, max_err = 1e-6, ** kwargs):
    eq, _ = is_equal(target, value, max_err = max_err, ** kwargs)
    assert not eq, 'Values should be different but are equal'

def assert_array(x):
    assert isinstance(x, (np.ndarray, np.integer, np.floating)), \
        'The function must return a `np.ndarray`, got {}'.format(type(x))

def assert_tensor(x):
    import keras.ops as K
    assert K.is_tensor(x), 'The function must return a `Tensor`, got {}'.format(type(x))

def assert_tf_tensor(x):
    import tensorflow as tf
    assert tf.is_tensor(x), 'The function should return a `tf.Tensor`, got {}'.format(type(x))

def assert_reproducible(value, file, *, directory = None, max_err = 1e-6, strict = None, ** kwargs):
    """ Compare ``value`` against a stored golden file, creating it on first run.

        - ``directory`` : where golden files live (defaults to `reproductibility_dir`)
        - ``strict``    : if the golden is missing, fail instead of creating it.
                          Defaults to the ``TESTS_STRICT_GOLDEN`` environment flag,
                          so a shared server / CI never silently freezes a value.
    """
    if directory is None:    directory = reproductibility_dir
    if strict is None:       strict = os.environ.get(STRICT_GOLDEN_ENV, '') not in ('', '0', 'false', 'False')

    path = os.path.join(directory, file)
    if not os.path.exists(path):
        if strict:
            raise AssertionError(
                'Reproducibility golden `{}` is missing and strict mode is enabled '
                '(set `{}=0` to allow creating it).'.format(file, STRICT_GOLDEN_ENV)
            )
        warnings.warn('Creating missing reproducibility golden : {}'.format(path))
        os.makedirs(directory, exist_ok = True)
        dump_data(filename = path, data = value)

    assert_equal(load_data(path), value, max_err = max_err, ** kwargs)

def assert_graph_compatible(fn,
                            * args,

                            target    = None,
                            is_random = False,
                            is_tensor_output  = True,

                            jit_compile   = False,

                            ** kwargs
                           ):
    """ Assert that ``fn`` runs inside a `tf.function` (graph / XLA) and matches eager output. """
    if fn in _graph_failed: return
    elif not is_tensorflow_available():
        raise SkipTest('`tensorflow` should be available')

    import keras
    import tensorflow as tf

    graph_fn = get_graph_function(fn, jit_compile = jit_compile)

    tf_args    = keras.tree.map_structure(convert_to_tf_tensor, args)
    tf_kwargs  = keras.tree.map_structure(convert_to_tf_tensor, kwargs)

    try:
        result  = graph_fn(* tf_args, ** tf_kwargs)
    except Exception as e:
        result = e

    if isinstance(result, Exception):
        _graph_failed[fn] = True
        raise AssertionError('The function does not support graph mode :\n{}'.format(result))

    if target is None:
        target = fn(* args, ** kwargs)

    if is_tensor_output:
        if isinstance(result, (list, tuple)):
            assert isinstance(result, type(target)), \
                'Expected : {} - got : {}'.format(type(target), result)
            for res in result: assert_tf_tensor(res)
        else:
            assert_tf_tensor(result)
    else:
        assert isinstance(result, type(target)), \
            'Expected : {} - got : {}'.format(type(target), result)

    if not is_random:
        assert_equal(target, keras.tree.map_structure(lambda t: t.numpy(), result))
    else:
        assert_equal(
            keras.tree.map_structure(lambda t: tuple(t.shape), target),
            keras.tree.map_structure(lambda t: tuple(t.shape), result)
        )

def assert_xla_compatible(fn, * args, ** kwargs):
    assert_graph_compatible(fn, * args, jit_compile = True, ** kwargs)
