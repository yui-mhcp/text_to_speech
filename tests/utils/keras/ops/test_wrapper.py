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

""" Pure-``pytest`` tests for the :class:`utils.keras.ops.Ops` wrapper *mechanism*.

This is the heart of the module : a callable that dispatches to a pure-``numpy``
implementation when every input is a ``np.ndarray`` / scalar (so ``keras`` is
never imported), and to ``keras`` (built lazily) as soon as one input is a
tensor — with an extra ``tensorflow`` graph path for ``tf.data`` pipelines.

Backend handling :
  - numpy-only dispatch is checked **keras-free** (unmarked tests) ;
  - cases that require a tensor are marked ``keras`` ;
  - the graph / backend-specialization behaviour is marked ``tensorflow``.
Both keras / tensorflow markers are auto-skipped (root ``conftest``) when the
dependency is missing.
"""

import os
import sys
import subprocess

import numpy as np
import pytest

from utils.keras import ops
from utils.keras.ops import Ops

# inputs that never trigger the keras path (kept unmarked = keras-free)
_NON_TENSOR = [
    pytest.param((0, 1, 2),    id = 'tuple'),
    pytest.param([0, 1, 2],    id = 'list'),
    pytest.param(range(3),     id = 'range'),
    pytest.param(np.arange(3), id = 'array'),
]


# --- construction ------------------------------------------------------------------

def test_init_resolves_numpy_fn_eagerly():
    """ ``Ops(name)`` resolves its numpy implementation at construction, keras-free. """
    fn = Ops('shape')

    assert not fn.built
    assert fn.name == 'shape'
    assert fn.numpy_fn is np.shape

@pytest.mark.keras
def test_init_resolves_keras_fn_lazily():
    import keras.ops as K

    fn = Ops('shape')
    assert fn.keras_fn is K.shape


# --- dispatch (numpy <-> keras) ----------------------------------------------------

@pytest.mark.parametrize('x', _NON_TENSOR)
def test_call_dispatches_to_numpy(x, asserts):
    """ Non-tensor inputs take the numpy path : correct value, ``ndarray`` out, never built. """
    fn = Ops('sum')

    asserts.assert_equal(3, fn(x))
    asserts.assert_array(fn(x))
    assert not fn.built, 'the numpy path must never build the keras function'

@pytest.mark.keras
def test_call_dispatches_to_keras_on_tensor(asserts):
    """ A tensor input takes the keras path : ``Tensor`` out, function gets built. """
    import keras.ops as K

    fn = Ops('sum')
    x  = K.arange(3)

    asserts.assert_equal(3, fn(x))
    asserts.assert_tensor(fn(x))
    assert fn.built

@pytest.mark.keras
@pytest.mark.parametrize('x', _NON_TENSOR + [pytest.param(None, id = 'tensor')])
def test_disable_np_always_uses_keras(x, asserts):
    """ With ``disable_np=True`` even non-tensor inputs go through keras. """
    import keras.ops as K
    if x is None: x = K.arange(3)

    fn = Ops('sum', disable_np = True)
    out = fn(x)

    asserts.assert_equal(3, out)
    asserts.assert_tensor(out)
    assert fn.built
    assert fn.numpy_fn is None


# --- backend specialization (tensorflow) -------------------------------------------

@pytest.mark.tensorflow
def test_backend_specific_op(asserts):
    """ The keras call is specialized at build time depending on the active backend, and
        an explicit ``tensorflow_fn`` is used in the tensorflow-graph path.
    """
    import keras
    import keras.ops as K
    import tensorflow as tf

    fn = Ops('shape')
    fn.build()

    assert fn.keras_fn is K.shape
    if keras.backend.backend() == 'tensorflow':
        assert fn.call_keras_fn is fn.keras_fn, \
            'the tensorflow-graph check should be disabled to directly call keras_fn'
    else:
        assert fn.call_keras_fn is not fn.keras_fn, \
            'the tensorflow-graph check should be enabled to pick keras_fn or tensorflow_fn'

    fn = Ops(
        'sum',
        tensorflow_fn = 'reduce_sum',
        keras_fn      = lambda * _: pytest.fail('keras should not be called')
    )
    fn.build()

    asserts.assert_equal(3, fn(range(3)))
    asserts.assert_array(fn(range(3)))
    assert fn.tensorflow_fn is tf.reduce_sum

    if keras.backend.backend() == 'tensorflow':
        assert fn.call_keras_fn is tf.reduce_sum
        asserts.assert_equal(3, fn(tf.range(3)))
        asserts.assert_tensor(fn(tf.range(3)))


# --- graph compatibility (tensorflow) ----------------------------------------------

@pytest.mark.tensorflow
def test_graph_execution(asserts):
    """ The op behaves normally eagerly, and switches to tensorflow functions inside a
        ``tf.function`` (graph mode), whatever the keras backend.
    """
    import keras
    import keras.ops as K
    import tensorflow as tf

    fn = Ops('sum')
    fn.build()

    def foo(x):
        return fn(x)

    @tf.function(reduce_retracing = True)
    def tf_graph_foo(x):
        return foo(x)

    if keras.backend.backend() != 'tensorflow':
        assert fn.call_keras_fn is not fn.keras_fn, 'the tensorflow-graph check should be enabled'

    # eager execution behaves like a normal call
    asserts.assert_array(foo(range(3)))
    asserts.assert_array(foo(np.arange(3)))
    asserts.assert_tensor(foo(K.arange(3)))

    # inside a `tf.function` the op must use tensorflow ops (and retrace cleanly)
    asserts.assert_tf_tensor(tf_graph_foo(tf.range(3)))
    asserts.assert_tf_tensor(tf_graph_foo(tf.range(4)))
    asserts.assert_tf_tensor(tf_graph_foo(tf.range(5)))

    if keras.backend.backend() != 'tensorflow':
        with pytest.raises(Exception):
            # calling a tensorflow op on a non-tensorflow tensor must fail
            tf_graph_foo(K.arange(3))


# --- nested_arg dispatch -----------------------------------------------------------
# `stack` / `concatenate` / `while_loop` take their tensors *inside* a list argument,
# so `_is_numpy` must look one level deeper (the `nested_arg` index) to choose the path.

def test_nested_arg_all_numpy_uses_numpy(asserts):
    """ A list of pure-numpy arrays (the ``nested_arg``) stays on the numpy path. """
    fn  = Ops('stack', nested_arg = 0)
    out = fn([np.arange(3), np.arange(3)])

    asserts.assert_array(out)
    assert not fn.built, 'a fully-numpy nested arg must not build the keras path'

@pytest.mark.keras
def test_nested_arg_with_one_tensor_uses_keras(asserts):
    """ As soon as **one** element of the nested arg is a tensor, the keras path is taken. """
    import keras.ops as K

    fn  = Ops('stack', nested_arg = 0)
    out = fn([np.arange(3), K.arange(3)])

    asserts.assert_tensor(out)
    assert fn.built


# --- dynamic loading & alias resolution (package level) ----------------------------
# `utils.keras.ops.__getattr__` exposes *any* keras op lazily as an `Ops`, applying the
# conventional aliases and disabling numpy for creation ops — all keras-free.

def test_getattr_returns_lazy_ops():
    """ An arbitrary op is exposed lazily as an `Ops`, resolved keras-free, not built. """
    fn = ops.cos
    assert isinstance(fn, Ops)
    assert fn.name == 'cos'
    assert fn.numpy_fn is np.cos
    assert not fn.built

def test_getattr_disables_numpy_for_creation_ops():
    """ Creation ops obtained dynamically always build a tensor (``numpy_fn is None``). """
    assert ops.linspace.numpy_fn is None
    assert ops.tri.numpy_fn is None

@pytest.mark.parametrize('alias,target', [
    pytest.param('reduce_sum',  'sum',     id = 'reduce_sum'),
    pytest.param('reduce_mean', 'mean',    id = 'reduce_mean'),
    pytest.param('gather',      'take',    id = 'gather'),
    pytest.param('range',       'arange',  id = 'range'),
    pytest.param('acos',        'arccos',  id = 'acos'),
    pytest.param('absolute',    'abs',     id = 'absolute'),
    pytest.param('clip_by_value', 'clip',  id = 'clip_by_value'),
])
def test_alias_resolution(alias, target):
    """ Conventional (tensorflow-style) aliases resolve to their canonical op name. """
    assert getattr(ops, alias).name == target


# --- keras-free guarantee (flexibility) --------------------------------------------

def test_numpy_path_never_imports_keras():
    """ The pure-numpy dispatch must never import keras / tensorflow / torch.

        This is *the* flexibility guarantee (a project that only feeds ``np.ndarray``
        never pulls a deep-learning backend). It can only be asserted in a **fresh**
        interpreter — the in-process ``sys.modules`` is polluted by the keras-marked
        tests — so we run a tiny script through a subprocess and check the leaked modules.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), * (['..'] * 4)))
    code = (
        'import sys, numpy as np\n'
        'from utils.keras import ops\n'
        'assert int(ops.sum(np.arange(3))) == 3\n'
        "ops.divide_no_nan(np.ones(3, 'float32'), np.zeros(3, 'float32'))\n"
        'ops.stack([np.arange(3), np.arange(3)])\n'
        "leaked = [m for m in ('keras', 'tensorflow', 'torch') if m in sys.modules]\n"
        "assert not leaked, 'numpy path imported : ' + ', '.join(leaked)\n"
    )
    env = dict(os.environ, PYTHONPATH = repo_root + os.pathsep + os.environ.get('PYTHONPATH', ''))
    proc = subprocess.run(
        [sys.executable, '-c', code],
        capture_output = True, text = True, env = env, timeout = 60
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
