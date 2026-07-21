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

""" Pure-``pytest`` tests for :mod:`models.core.base_model`.

These cover the **wrapper logic** of `BaseModel` *without loading a real model nor
running any inference* : a `FakeRuntime` stands in for the keras / TensorRT engine and a
`DummyModel` leaf assigns it in `build` (see :mod:`tests.models.core._helpers`).

Covered :
  - signature derivation : `input_shape` / `input_dtype` / `unbatched_*` from a (possibly
    nested) signature, and the `NotImplementedError` signature contract ;
  - the `NotImplementedError` contracts of `infer` / `get_inference_callbacks` /
    `filter_prediction_output`, and `filter_returned_output`'s default delegation ;
  - the directory / file path-properties built by `__init_subclass__` ;
  - `__init__` branching : build (keras runtime) vs inference-only runtime, the `save`
    guard, and config persistence + reload ;
  - `get_config` round-trip / additivity (`build_kwargs`) ;
  - the `ModelInstances` metaclass : memoization, `reload`, `class_name` mismatch ;
  - prediction plumbing : `_normalize_prediction_inputs`, `_finalize_predictions`,
    `_finalize_cached_prediction`, and `predict`'s per-item orchestration (fake `infer`).

Out of scope (need keras) : the real `build` / `call` / `infer`, `save_models_config` /
`_restore_model`, and `describe_model`.
"""

import os
import pytest

from utils.keras import TensorSpec
from models.core.base_model import ModelInstances
from tests.models.core._helpers import (
    DummyModel, OtherDummyModel, NoSignatureModel, FakeRuntime
)


# --- signature derivation ----------------------------------------------------------

def test_shape_and_dtype_derived_from_signature():
    m = DummyModel(
        name = 'sig', save = False,
        input_signature  = TensorSpec(shape = (None, 8), dtype = 'float32'),
        output_signature = TensorSpec(shape = (None, 2), dtype = 'int32'),
    )
    assert m.input_shape  == (None, 8)
    assert m.output_shape == (None, 2)
    assert m.input_dtype  == 'float32'
    assert m.output_dtype == 'int32'

def test_nested_signature_derivation():
    m = DummyModel(
        name = 'nested', save = False,
        input_signature = {
            'a' : TensorSpec(shape = (None, 4), dtype = 'float32'),
            'b' : TensorSpec(shape = (None,),   dtype = 'int32'),
        },
    )
    assert m.input_shape == {'a' : (None, 4), 'b' : (None,)}
    assert m.input_dtype == {'a' : 'float32', 'b' : 'int32'}

def test_unbatched_signature_drops_batch_axis():
    m = DummyModel(
        name = 'unb', save = False,
        input_signature = TensorSpec(shape = (None, 8), dtype = 'float32'),
    )
    assert m.unbatched_input_signature == TensorSpec(shape = (8,), dtype = 'float32')

def test_signature_contract_enforced():
    m = NoSignatureModel(name = 'nosig', save = False)
    with pytest.raises(NotImplementedError):
        _ = m.input_signature
    with pytest.raises(NotImplementedError):
        _ = m.output_signature


# --- inference contracts -----------------------------------------------------------

def test_inference_contracts_raise_by_default():
    m = NoSignatureModel(name = 'contract', save = False)
    with pytest.raises(NotImplementedError):
        m.infer('x')
    with pytest.raises(NotImplementedError):
        m.get_inference_callbacks()
    with pytest.raises(NotImplementedError):
        m.filter_prediction_output({})

def test_filter_returned_output_defaults_to_filter_prediction():
    class _M(DummyModel):
        def filter_prediction_output(self, output):
            return {'id' : output['id']}

    m = _M(name = 'filter_ret', save = False)
    assert m.filter_returned_output({'id' : 1, 'heavy' : [1, 2, 3]}) == {'id' : 1}


# --- path properties ---------------------------------------------------------------

def test_path_properties(isolated_saving_dir):
    root = isolated_saving_dir
    m    = DummyModel(name = 'paths', save = False)
    assert m.directory        == '{}/paths'.format(root)
    assert m.save_dir         == '{}/paths/saving'.format(root)
    assert m.config_file      == '{}/paths/config.json'.format(root)
    assert m.history_file     == '{}/paths/saving/history.json'.format(root)


# --- construction branching --------------------------------------------------------

def test_build_branch_uses_subclass_build():
    # runtime defaults to 'keras' -> `__init__` calls the (overridden) `build`
    m = DummyModel(name = 'build', save = False)
    assert m.runtime == 'keras'
    assert isinstance(m.model, FakeRuntime)
    assert m.base_dtype == 'float32'

def test_inference_only_runtime_uses_build_runtime():
    # runtime != 'keras' -> `__init__` goes through `build_runtime(runtime, ...)`
    m = DummyModel(name = 'inf', save = False, runtime = 'fake')
    assert m.runtime == 'fake'
    assert isinstance(m.model, FakeRuntime)

def test_save_false_writes_nothing(isolated_saving_dir):
    m = DummyModel(name = 'nosave', save = False, runtime = 'fake')
    assert not os.path.exists(m.config_file)
    assert not os.path.exists(m.directory)

def test_save_true_persists_config_and_reloads(isolated_saving_dir):
    m = DummyModel(name = 'persist', save = True, runtime = 'fake')
    assert os.path.exists(m.config_file)

    # fresh construction on the same name -> `__init__` takes the restore branch
    ModelInstances._instances.clear()
    reloaded = DummyModel(name = 'persist')
    assert reloaded.runtime == 'fake'          # runtime came back from the persisted config
    assert reloaded.get_config()['name'] == 'persist'


# --- get_config --------------------------------------------------------------------

def test_get_config_roundtrip():
    m   = DummyModel(name = 'cfg', save = False, runtime = 'fake', pretrained_name = 'foo')
    cfg = m.get_config()
    assert cfg['name'] == 'cfg'
    assert cfg['runtime'] == 'fake'
    assert cfg['pretrained_name'] == 'foo'
    for k in ('run_eagerly', 'support_xla', 'graph_compile_config'):
        assert k in cfg

def test_get_config_includes_build_kwargs():
    # unknown kwargs are stored in `build_kwargs` and surfaced additively by `get_config`
    m = DummyModel(name = 'cfg2', save = False, runtime = 'fake', extra_param = 7)
    assert m.get_config()['extra_param'] == 7


# --- ModelInstances metaclass ------------------------------------------------------

def test_instances_are_memoized_by_name():
    a = DummyModel(name = 'same', save = False)
    b = DummyModel(name = 'same', save = False)
    assert a is b

def test_reload_bypasses_the_cache():
    a = DummyModel(name = 'rl', save = False)
    b = DummyModel(name = 'rl', save = False, reload = True)
    assert a is not b

def test_class_name_mismatch_raises(isolated_saving_dir):
    DummyModel(name = 'clash', save = True, runtime = 'fake')
    ModelInstances._instances.clear()
    # a different class trying to reload the same persisted config must be rejected
    with pytest.raises(ValueError):
        OtherDummyModel(name = 'clash', runtime = 'fake')


# --- prediction plumbing -----------------------------------------------------------

@pytest.mark.parametrize('inputs, expected', [
    pytest.param('a',          ['a'],          id = 'str'),
    pytest.param({'x' : 1},    [{'x' : 1}],    id = 'dict'),
    pytest.param(['a', 'b'],   ['a', 'b'],     id = 'list_kept'),
])
def test_normalize_prediction_inputs(inputs, expected):
    m = DummyModel(name = 'norm', save = False)
    assert m._normalize_prediction_inputs(inputs) == expected


class _FilterModel(DummyModel):
    def filter_prediction_output(self, output):
        return {'id' : output['id']}

def test_finalize_predictions_returns_full_output_when_requested():
    m   = _FilterModel(name = 'fp_full', save = False)
    out = {'id' : 1, 'v' : 2}
    assert m._finalize_predictions(out, return_output = True) is out

def test_finalize_predictions_returns_filtered_when_not_requested():
    m = _FilterModel(name = 'fp_filtered', save = False)
    # no callbacks / no `predicted` -> falls back to `filter_returned_output`
    assert m._finalize_predictions({'id' : 1, 'v' : 2}, return_output = False) == {'id' : 1}

def test_finalize_predictions_resolves_stored_entry():
    class _EntryCallback:
        saves_to_disk  = False
        provides_entry = True
        def __call__(self, infos, output, ** kwargs):
            return 'k1'
        def join(self):
            pass

    m         = _FilterModel(name = 'fp_entry', save = False)
    predicted = {'k1' : {'stored' : True}}
    result    = m._finalize_predictions(
        {'id' : 1}, callbacks = [_EntryCallback()], predicted = predicted, return_output = False
    )
    assert result == {'stored' : True}

def test_finalize_cached_prediction_reapplies_callbacks_without_saving():
    class _RecordingCallback:
        saves_to_disk  = False
        provides_entry = False
        def __init__(self):
            self.seen = []
        def __call__(self, infos, output, ** kwargs):
            self.seen.append((infos, output))
        def join(self):
            pass

    class _M(DummyModel):
        def filter_prediction_output(self, output):
            return output

    m   = _M(name = 'cached', save = False)
    cb  = _RecordingCallback()
    res = m._finalize_cached_prediction('key', callbacks = [cb], predicted = {'key' : {'cached' : 1}})
    assert res == {'cached' : 1}
    # the callback ran on the stored entry, with an empty (non-persisted) `output`
    assert cb.seen == [({'cached' : 1}, {})]


class _PredModel(DummyModel):
    """ Leaf whose `infer` follows the documented contract (finalize a freshly built output). """
    def get_inference_callbacks(self, ** kwargs):
        return {}, []
    def filter_prediction_output(self, output):
        return output
    def infer(self, data, *, callbacks = None, predicted = None, overwrite = False,
              return_output = True, ** kwargs):
        return self._finalize_predictions(
            {'id' : data, 'v' : data * 2},
            callbacks = callbacks, predicted = predicted, return_output = return_output
        )

def test_predict_runs_infer_per_item_in_order():
    m = _PredModel(name = 'pred', save = False)
    assert m.predict([1, 2, 3]) == [
        {'id' : 1, 'v' : 2}, {'id' : 2, 'v' : 4}, {'id' : 3, 'v' : 6}
    ]

def test_predict_normalizes_single_input():
    m = _PredModel(name = 'pred_single', save = False)
    assert m.predict('x') == [{'id' : 'x', 'v' : 'xx'}]
