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

""" Pure-``pytest`` tests for :mod:`utils.keras.compile`.

This module covers the multi-framework compilation layer — the equivalent of
``tf.function`` / ``jax.jit`` / ``torch.compile`` exposed through a single API :

  - :func:`graph_compile`    : the main decorator (eager / graph / XLA dispatch,
    type-hint casting, ``run_eagerly`` / ``use_xla`` / ``recompile`` switches) ;
  - :func:`execute_eagerly`  : run a plain python function inside a ``tf.function``
    via ``tf.py_function`` / ``tf.numpy_function`` ;
  - :func:`compile_function` : the low-level per-backend compile dispatch ;
  - :func:`replace_kwargs`   : pure-python signature rewriting helper ;
  - :class:`TensorSpec`      : the shape/dtype annotation value-object.

Tests are backend-aware : assertions about the execution *mode* branch on the
active backend, since "graph without XLA" is only reachable with `tensorflow`.
Tests needing `tensorflow` are marked ``@pytest.mark.tensorflow`` and auto-skipped
by the suite ``conftest`` when it is not installed.
"""

import time
import inspect
import threading

import keras
import numpy as np
import pytest

from utils.keras import (
    TensorSpec, ops, graph_compile, execute_eagerly, compile_function, replace_kwargs
)
from utils.keras.compile import _infer_execution_mode, _cast_arg, ExecutionMode
from utils.keras.ops.execution_contexts import XLAExecution

from tests import asserts


def _run_capturing(errors, fn, ** kwargs):
    """ Run ``fn`` in a worker thread, recording any exception (assertions raised
        inside a `threading.Thread` would otherwise be swallowed silently). """
    try:
        fn(** kwargs)
    except BaseException as e:  # noqa: BLE001 - we re-raise from the main thread
        errors.append(e)


# ---------------------------------------------------------------------------
# execute_eagerly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('numpy', [False, True])
def test_execute_eagerly_simple(numpy):
    @execute_eagerly(Tout = 'float32', numpy = numpy)
    def foo(x, y):
        assert ops.executing_eagerly()
        if not numpy: asserts.assert_tensor(x)
        else:         asserts.assert_array(x)
        assert isinstance(y, float)
        return x * y

    foo(ops.arange(5, dtype = 'float32'), y = 2.)


@pytest.mark.tensorflow
@pytest.mark.parametrize('numpy', [False, True])
def test_execute_eagerly_tensorflow_simple(numpy):
    import tensorflow as tf

    @tf.function
    def foo(x, y):
        assert not tf.executing_eagerly()
        assert ops.is_tensorflow_graph()
        out = foo_eager(x, y)
        assert out.dtype == tf.float32, 'Dtype is {}'.format(out.dtype)
        assert out.shape.rank is None, 'The shape should be unknown but is {}'.format(out.shape)
        return out

    @execute_eagerly(Tout = 'float32', numpy = numpy)
    def foo_eager(x, y):
        assert tf.executing_eagerly()
        assert ops.executing_eagerly()
        assert not ops.is_tensorflow_graph()
        if not numpy:
            asserts.assert_tf_tensor(x)
            asserts.assert_tf_tensor(y)
        else:
            asserts.assert_array(x)
            asserts.assert_array(y)
        return tf.cast(x, 'float32') * y

    foo(tf.range(5), tf.constant(2, 'float32'))


@pytest.mark.tensorflow
@pytest.mark.parametrize('numpy', [False, True])
def test_execute_eagerly_tensorflow_with_kwargs(numpy):
    import tensorflow as tf

    @tf.function
    def foo(x, y):
        assert not tf.executing_eagerly()
        out = foo_eager(x, y = y)
        assert out.dtype == tf.float32, 'Dtype is {}'.format(out.dtype)
        assert out.shape.rank is None, 'The shape should be unknown but is {}'.format(out.shape)
        return out

    @execute_eagerly(Tout = 'float32', numpy = numpy)
    def foo_eager(x, *, y = None):
        assert tf.executing_eagerly()
        assert ops.executing_eagerly()
        assert y is not None, 'y has not been forwarded correctly'
        if not numpy:
            asserts.assert_tf_tensor(x)
            asserts.assert_tf_tensor(y)
        else:
            asserts.assert_array(x)
            asserts.assert_array(y)
        return tf.cast(x, 'float32') * y

    foo(tf.range(5), tf.constant(2, 'float32'))


@pytest.mark.tensorflow
@pytest.mark.parametrize('numpy', [False, True])
def test_execute_eagerly_tensorflow_with_signature(numpy):
    import tensorflow as tf

    @tf.function
    def foo(x, y):
        assert not tf.executing_eagerly()
        out = foo_eager(x, y)
        assert isinstance(out, (list, tuple)), 'Got type {}'.format(type(out))
        assert out[0].dtype == tf.float32, 'The dtype is {}'.format(out[0].dtype)
        assert len(out[0].shape) == 1, 'The shape is {}'.format(out[0].shape)
        assert out[1].dtype == tf.int32, 'The dtype is {}'.format(out[1].dtype)
        assert len(out[1].shape) == 0, 'The shape is {}'.format(out[1].shape)
        return out

    @execute_eagerly(signature = [
        TensorSpec(shape = (None, ), dtype = 'float32'),
        TensorSpec(shape = (), dtype = 'int32')
    ], numpy = numpy)
    def foo_eager(x, y):
        assert tf.executing_eagerly()
        assert ops.executing_eagerly()
        assert y is not None, 'y has not been forwarded correctly'
        if not numpy:
            asserts.assert_tf_tensor(x)
            asserts.assert_tf_tensor(y)
        else:
            asserts.assert_array(x)
            asserts.assert_array(y)
        return tf.cast(x, 'float32') * y, np.array(len(x), dtype = 'int32')

    foo(tf.range(5), tf.constant(2, 'float32'))


def test_executing_eagerly_detects_xla_context():
    """ ``compile_function(..., jit_compile = True)`` should run the body in a
        non-eager (XLA) context, detected by ``ops.executing_eagerly``. """
    def foo(x):
        if ops.is_tensorflow_backend():
            import tensorflow as tf
            assert not tf.executing_eagerly()
            assert ops.is_tensorflow_graph()
        assert not ops.executing_eagerly()
        return x ** 2

    xla_foo = compile_function(foo, jit_compile = True)
    assert ops.executing_eagerly()
    with XLAExecution():
        # the context manager has no effect on the tensorflow backend
        if ops.is_tensorflow_backend():
            assert ops.executing_eagerly()
        else:
            assert not ops.executing_eagerly()
        xla_foo(2)

    assert ops.executing_eagerly()


# ---------------------------------------------------------------------------
# graph_compile - execution modes
# ---------------------------------------------------------------------------

def test_run_eagerly():
    @graph_compile
    def foo(x : TensorSpec(), eager = False):
        if eager:
            assert ops.executing_eagerly(), 'This should be executed eagerly'
        else:
            assert not ops.executing_eagerly(), 'This should be executed in graph'

    assert ops.executing_eagerly()
    foo(2)
    assert ops.executing_eagerly()
    foo(2, True, run_eagerly = True)


def test_run_eagerly_nested():
    """ ``run_eagerly`` must propagate to nested ``graph_compile``d functions. """
    @graph_compile
    def foo(x : TensorSpec(), eager = False):
        nested_foo(eager)

    @graph_compile
    def nested_foo(eager):
        if eager:
            assert ops.executing_eagerly(), 'This should be executed eagerly'
        else:
            assert not ops.executing_eagerly(), 'This should be executed in graph'

    foo(2)
    assert ops.executing_eagerly()
    foo(2, True, run_eagerly = True)


@pytest.mark.timeout(30)
def test_multi_threading():
    """ The eager/graph state is thread-local : a thread forced to ``run_eagerly``
        must not leak that state into a graph-running thread. """
    @graph_compile
    def foo_eager():
        assert ops.executing_eagerly()
        time.sleep(0.01)
        assert ops.executing_eagerly()

    @graph_compile
    def foo():
        assert not ops.executing_eagerly(), \
            'The separation between threads of `run_eagerly` seems not working'

    errors = []
    for _ in range(5):
        t = threading.Thread(
            target = _run_capturing,
            args   = (errors, foo_eager),
            kwargs = {'run_eagerly' : True, 'recompile' : True}
        )
        t.start()
        time.sleep(1e-3)

        foo(recompile = True)   # main thread : its assertion propagates normally
        t.join()

    assert not errors, 'Worker thread(s) failed : {}'.format(errors)


def test_use_xla_false_execution_mode():
    """ With ``use_xla = False``, only `tensorflow` can run a plain graph ; other
        backends fall back to eager (graph-without-XLA is unsupported there). """
    @graph_compile
    def foo(x : TensorSpec()):
        if ops.is_tensorflow_backend():
            assert not ops.executing_eagerly(), 'tensorflow should run in graph'
        else:
            assert ops.executing_eagerly(), 'non-tf backends fall back to eager'
        return x

    foo(2, use_xla = False)


def test_prefer_xla_runs_compiled():
    @graph_compile(prefer_xla = True)
    def foo(x : TensorSpec(dtype = 'float32')):
        assert not ops.executing_eagerly(), 'prefer_xla should compile (non-eager)'
        return x * 2

    foo(2.)


def test_support_xla_false_warns():
    """ Requesting ``use_xla = True`` on a function declared ``support_xla = False``
        must warn (and silently downgrade) rather than fail. """
    @graph_compile(support_xla = False)
    def foo(x : TensorSpec()):
        return x

    with pytest.warns(UserWarning, match = 'does not support XLA'):
        foo(2, use_xla = True)


# ---------------------------------------------------------------------------
# graph_compile - type-hint casting
# ---------------------------------------------------------------------------

def test_simple_cast():
    @graph_compile
    def foo(x):
        assert isinstance(x, int)

    @graph_compile
    def foo_with_cast(x : TensorSpec()):
        asserts.assert_tensor(x)
        assert ops.dtype_to_str(x.dtype) == 'int32'

    @graph_compile
    def foo_with_spec(x : TensorSpec(dtype = 'float32')):
        asserts.assert_tensor(x)
        assert ops.dtype_to_str(x.dtype) == 'float32'

    foo(2)
    foo_with_cast(2)
    foo_with_cast(x = 2, recompile = True)
    foo_with_spec(2)
    foo_with_spec(x = 2, recompile = True)


def test_cast_default():
    @graph_compile
    def foo(x = 2):
        assert isinstance(x, int)

    @graph_compile
    def foo_with_spec(x : TensorSpec(dtype = 'int32') = 2):
        assert ops.is_tensor(x)
        assert ops.dtype_to_str(x.dtype) == 'int32'

    foo()
    foo_with_spec()


def test_cast_kwargs():
    @graph_compile(kwargs_annots = {
        'y' : TensorSpec(), 'z' : TensorSpec(dtype = 'float32')
    })
    def foo(x, ** kwargs):
        asserts.assert_tensor(kwargs['y'])
        asserts.assert_tensor(kwargs['z'])
        assert ops.is_int(kwargs['y']), str(kwargs['y'].dtype)
        assert ops.is_float(kwargs['z']), str(kwargs['z'].dtype)
        assert isinstance(kwargs['method'], str)
        assert isinstance(kwargs['cond'], bool)
        assert isinstance(kwargs['weight'], float)
        assert kwargs['none'] is None

    foo(2, y = 3, z = 4, method = 'default', cond = True, none = None, weight = 2.)


@pytest.mark.tensorflow
@pytest.mark.skipif(
    keras.backend.backend() == 'tensorflow',
    reason = 'This test requires `tensorflow` available while using another backend'
)
def test_force_tensorflow():
    import tensorflow as tf

    @graph_compile
    def foo(x):
        assert tf.executing_eagerly(), 'This should not detect the XLA execution of another backend'
        assert not ops.executing_eagerly()

    @graph_compile(force_tensorflow = True)
    def foo_tf(x : TensorSpec(dtype = 'int32')):
        assert not tf.executing_eagerly(), 'This should be executed in TF graph'
        assert ops.is_tensorflow_graph()
        assert not ops.executing_eagerly(), \
            'The `ops.executing_eagerly` has to detect `tensorflow-graph` executions in all backends'
        asserts.assert_tf_tensor(x)
        assert ops.dtype_to_str(x.dtype) == 'int32'

    foo(2)
    foo_tf(2)


# ---------------------------------------------------------------------------
# compile_function
# ---------------------------------------------------------------------------

def test_compile_function_no_jit_passthrough():
    """ Without jit and outside tensorflow, ``compile_function`` is a no-op. """
    def f(x):
        return x

    compiled = compile_function(f, jit_compile = False)
    if ops.is_tensorflow_backend():
        assert compiled is not f, 'tensorflow wraps even without jit'
    else:
        assert compiled is f, 'non-tf backend without jit should return `fn` unchanged'


def test_compile_function_jit_correctness():
    compiled = compile_function(lambda x: x * 2, jit_compile = True)
    out = compiled(ops.convert_to_tensor([1, 2, 3], 'float32'))
    asserts.assert_equal([2., 4., 6.], out)


@pytest.mark.tensorflow
def test_compile_function_force_tensorflow():
    import tensorflow as tf

    compiled = compile_function(lambda x: x + 1, jit_compile = False, force_tensorflow = True)
    out = compiled(tf.constant([1, 2, 3]))
    asserts.assert_tf_tensor(out)
    asserts.assert_equal([2, 3, 4], out)


# ---------------------------------------------------------------------------
# replace_kwargs (pure-python signature rewriting)
# ---------------------------------------------------------------------------

def test_replace_kwargs_merges_nested_defaults():
    def nested(a, b = 1, c = 2): pass
    def fn(x, ** kwargs): pass

    replace_kwargs(fn, nested)
    params = inspect.signature(fn).parameters

    assert list(params) == ['x', 'b', 'c']
    assert 'a' not in params, 'params without a default should not be forwarded'
    assert params['b'].default == 1
    assert params['b'].kind == inspect.Parameter.KEYWORD_ONLY
    assert params['c'].default == 2


def test_replace_kwargs_ignore():
    def nested(b = 1, c = 2): pass
    def fn(x, ** kwargs): pass

    replace_kwargs(fn, nested, ignore = ('c', ))
    assert list(inspect.signature(fn).parameters) == ['x', 'b']


def test_replace_kwargs_skips_existing_params():
    def nested(x = 5, b = 1): pass   # `x` already exists on `fn`
    def fn(x, ** kwargs): pass

    replace_kwargs(fn, nested)
    params = inspect.signature(fn).parameters

    assert list(params) == ['x', 'b'], '`x` must not be duplicated'
    assert params['x'].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_replace_kwargs_as_decorator():
    def nested(b = 1, c = 2): pass

    @replace_kwargs(nested)
    def fn(x, ** kwargs): pass

    assert list(inspect.signature(fn).parameters) == ['x', 'b', 'c']


def test_replace_kwargs_updates_annotations():
    def nested(b : int = 1): pass
    def fn(x, ** kwargs): pass

    replace_kwargs(fn, nested)
    assert fn.__annotations__.get('b') is int


# ---------------------------------------------------------------------------
# TensorSpec
# ---------------------------------------------------------------------------

def test_tensorspec_defaults():
    spec = TensorSpec()
    assert spec.shape is None
    assert spec.dtype is None
    assert spec.name is None
    assert spec.static is False


def test_tensorspec_eq_and_hash_consistent():
    a = TensorSpec(shape = (None, ), dtype = 'float32', name = 'x')
    b = TensorSpec(shape = (None, ), dtype = 'float32', name = 'x')

    assert a == b
    assert hash(a) == hash(b), 'equal specs must share their hash'

    # regression : `__eq__` used to ignore `shape` (and `static`)
    assert a != TensorSpec(shape = (5, ), dtype = 'float32', name = 'x')
    assert a != TensorSpec(shape = (None, ), dtype = 'float32', name = 'x', static = True)


def test_tensorspec_eq_type_guard():
    # `__eq__` must not raise on a non-TensorSpec (it used to access `o.name`)
    assert (TensorSpec() == 'not a spec') is False
    assert (TensorSpec() != 42) is True


def test_tensorspec_hashable_in_collections():
    specs = {
        TensorSpec(dtype = 'int32'),
        TensorSpec(dtype = 'int32'),
        TensorSpec(dtype = 'float32'),
    }
    assert len(specs) == 2


# ---------------------------------------------------------------------------
# _infer_execution_mode  (internal mode-selection logic)
# ---------------------------------------------------------------------------
# Returns ``(ExecutionMode, ctx_manager)``. Several branches depend on the active
# backend ("graph without XLA" is tensorflow-only), so those assertions branch on
# ``ops.is_tensorflow_backend()``.

def _infer(run_eagerly = None, use_xla = None, prefer_xla = None,
           support_xla = True, force_tensorflow = False):
    return _infer_execution_mode(
        run_eagerly = run_eagerly,
        use_xla = use_xla,
        prefer_xla = prefer_xla,
        support_xla = support_xla,
        force_tensorflow = force_tensorflow
    )


def test_infer_mode_run_eagerly():
    mode, ctx = _infer(run_eagerly = True)
    assert mode == ExecutionMode.EAGER
    assert isinstance(ctx, ops.EagerExecution)


def test_infer_mode_nested_eager_propagates():
    # inside an EagerExecution scope, should_execute_eagerly() is True, so a nested
    # call (run_eagerly left to None) must also run eagerly — with no ctx manager
    with ops.EagerExecution():
        mode, ctx = _infer(run_eagerly = None)
    assert mode == ExecutionMode.EAGER
    assert ctx is None


def test_infer_mode_nested_eager_overridden_by_explicit_false():
    # an explicit `run_eagerly = False` must override the ambient eager request
    with ops.EagerExecution():
        mode, _ = _infer(run_eagerly = False)
    assert mode != ExecutionMode.EAGER


def test_infer_mode_prefer_xla():
    mode, ctx = _infer(prefer_xla = True)
    assert mode == ExecutionMode.XLA
    assert isinstance(ctx, ops.XLAExecution)


def test_infer_mode_use_xla_explicit_true():
    mode, ctx = _infer(use_xla = True)
    assert mode == ExecutionMode.XLA
    assert isinstance(ctx, ops.XLAExecution)


def test_infer_mode_use_xla_false():
    mode, ctx = _infer(use_xla = False)
    if ops.is_tensorflow_backend():
        assert mode == ExecutionMode.GRAPH
        assert isinstance(ctx, ops.XLAExecution)
    else:
        assert mode == ExecutionMode.EAGER
        assert ctx is None


def test_infer_mode_default_resolution():
    # use_xla=None, no prefer → tensorflow runs a plain graph, others default to XLA
    mode, ctx = _infer()
    assert mode == (ExecutionMode.GRAPH if ops.is_tensorflow_backend() else ExecutionMode.XLA)
    assert isinstance(ctx, ops.XLAExecution)


def test_infer_mode_support_xla_false_warns_and_downgrades():
    with pytest.warns(UserWarning, match = 'does not support XLA'):
        mode, ctx = _infer(use_xla = True, support_xla = False)
    # XLA was requested but unsupported → downgraded to graph (tf) / eager (others)
    if ops.is_tensorflow_backend():
        assert mode == ExecutionMode.GRAPH
    else:
        assert mode == ExecutionMode.EAGER
        assert ctx is None


def test_infer_mode_force_tensorflow_graph():
    # `force_tensorflow` keeps a tf graph reachable even on a non-tf backend
    mode, ctx = _infer(use_xla = False, force_tensorflow = True)
    assert mode == ExecutionMode.GRAPH
    assert isinstance(ctx, ops.XLAExecution)


# ---------------------------------------------------------------------------
# _cast_arg  (type-hint driven casting)
# ---------------------------------------------------------------------------

def test_cast_arg_none_returns_none():
    assert _cast_arg(None, TensorSpec(dtype = 'int32')) is None


def test_cast_arg_non_spec_passthrough():
    # a non-TensorSpec annotation leaves the value untouched
    assert _cast_arg(5, None) == 5
    assert _cast_arg('hello', 'not-a-spec') == 'hello'


def test_cast_arg_scalar_to_tensor():
    out = _cast_arg(2, TensorSpec(dtype = 'int32'))
    asserts.assert_tensor(out)
    assert ops.dtype_to_str(out.dtype) == 'int32'

    out = _cast_arg(2, TensorSpec(dtype = 'float32'))
    assert ops.dtype_to_str(out.dtype) == 'float32'


def test_cast_arg_default_dtype():
    out = _cast_arg(2, TensorSpec())   # dtype None → backend default int
    asserts.assert_tensor(out)
    assert ops.dtype_to_str(out.dtype) == 'int32'


def test_cast_arg_static_skipped_in_xla():
    spec = TensorSpec(dtype = 'int32', static = True)
    # a static arg stays python-side in XLA mode (traced as a constant)
    assert _cast_arg(2, spec, mode = ExecutionMode.XLA) == 2
    # ... but is still cast in graph mode
    asserts.assert_tensor(_cast_arg(2, spec, mode = ExecutionMode.GRAPH))


def test_cast_arg_list_annotation():
    out = _cast_arg([2, 3.], [TensorSpec(dtype = 'int32'), TensorSpec(dtype = 'float32')])
    assert isinstance(out, list) and len(out) == 2
    asserts.assert_tensor(out[0])
    asserts.assert_tensor(out[1])
    assert ops.dtype_to_str(out[0].dtype) == 'int32'
    assert ops.dtype_to_str(out[1].dtype) == 'float32'


def test_cast_arg_dict_annotation_only_annotated_keys():
    out = _cast_arg({'a' : 2, 'b' : 3}, {'a' : TensorSpec(dtype = 'int32')})
    # only keys present in BOTH the annotation and the value are returned
    assert set(out) == {'a'}
    asserts.assert_tensor(out['a'])


def test_cast_arg_tuple_value_broadcasts_spec():
    out = _cast_arg((2, 3), TensorSpec(dtype = 'int32'))
    assert isinstance(out, tuple) and len(out) == 2
    for v in out:
        asserts.assert_tensor(v)
        assert ops.dtype_to_str(v.dtype) == 'int32'


@pytest.mark.tensorflow
def test_cast_arg_force_tensorflow():
    out = _cast_arg(2, TensorSpec(dtype = 'int32'), force_tensorflow = True)
    asserts.assert_tf_tensor(out)
    assert ops.dtype_to_str(out.dtype) == 'int32'
