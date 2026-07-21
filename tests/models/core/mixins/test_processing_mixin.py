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

""" Pure-``pytest`` tests for :mod:`models.core.mixins.processing_mixin`.

`ModelProcessingMixin` is pure-Python wiring : it auto-builds the `{prefix}_data`
functions from the per-side `{prefix}_input` / `{prefix}_output` hooks a leaf model
defines. It imports no keras (only `loggers` for `@timer` and `utils.keras.ops` for the
numpy-dispatching `logical_and`), so these tests run **without a deep-learning backend**
and are left unmarked.

Rather than a full `BaseModel`, we mix `ModelProcessingMixin` into a bare object and
attach the `_input` / `_output` hooks by hand, then call `_init_processing_functions`.
"""

import numpy as np
import pytest

from models.core.mixins.processing_mixin import ModelProcessingMixin


class _Proc(ModelProcessingMixin):
    """ Bare host for the mixin ; hooks are attached per-test before `_init_*`. """


# --- get_input / _get_output sequencing --------------------------------------------

def test_get_input_sequences_prepare_then_process():
    m = _Proc()
    m.prepare_input = lambda inp, ** kw: inp + ['prepared']
    m.process_input = lambda inp, ** kw: inp + ['processed']
    m._init_processing_functions()
    assert m.get_input(['raw']) == ['raw', 'prepared', 'processed']


# --- {prefix}_data auto-wiring -----------------------------------------------------

def test_prepare_data_wired_from_input_and_output():
    m = _Proc()
    m.prepare_input  = lambda inp, ** kw: ('IN', inp)
    m.prepare_output = lambda out, ** kw: ('OUT', out)
    m._init_processing_functions()

    assert hasattr(m, 'prepare_data')
    inputs, output = m.prepare_data({'x' : 1})
    assert inputs == ('IN', {'x' : 1})
    assert output == ('OUT', {'x' : 1})

def test_output_processing_receives_inputs_kwarg():
    """ When `{prefix}_output` declares an `inputs` parameter, the wiring feeds it the
        freshly-processed inputs (e.g. AutoEncoder-style output == inputs). """
    m = _Proc()
    m.process_input = lambda inp, ** kw: inp * 2
    captured = {}
    def process_output(out, inputs = None, ** kw):
        captured['inputs'] = inputs
        return out + 1
    m.process_output = process_output
    m._init_processing_functions()

    inputs, output = m.process_data(10, 5)
    assert inputs == 20
    assert output == 6
    assert captured['inputs'] == 20

def test_get_output_is_bound_to_prepare_output():
    """ A leaf defining a single `get_output` (+ `prepare_input`) has it wired as
        `prepare_output`, then folded into `prepare_data`. """
    m = _Proc()
    m.get_output    = lambda data, ** kw: ('G', data)
    m.prepare_input = lambda inp, ** kw: inp
    m._init_processing_functions()

    assert m.prepare_output is m.get_output
    _, output = m.prepare_data('d')
    assert output == ('G', 'd')


# --- prepare_data argument unpacking -----------------------------------------------

@pytest.mark.parametrize('data, exp_in, exp_out', [
    pytest.param(({'a' : 1},),      {'a' : 1}, {'a' : 1}, id = 'single_dict'),
    pytest.param((('I', 'O'),),     'I',       'O',       id = 'single_2tuple'),
    pytest.param(('I', 'O'),        'I',       'O',       id = 'two_args'),
])
def test_prepare_data_unpacking(data, exp_in, exp_out):
    m = _Proc()
    m.prepare_input  = lambda inp, ** kw: inp
    m.prepare_output = lambda out, ** kw: out
    m._init_processing_functions()

    inputs, output = m.prepare_data(* data)
    assert inputs == exp_in
    assert output == exp_out

def test_prepare_data_rejects_wrong_arity():
    m = _Proc()
    m.prepare_input  = lambda inp, ** kw: inp
    m.prepare_output = lambda out, ** kw: out
    m._init_processing_functions()

    with pytest.raises(RuntimeError):
        m.prepare_data('a', 'b', 'c')


# --- _filter_data ------------------------------------------------------------------

@pytest.mark.parametrize('valid_in, valid_out, expected', [
    (True,  True,  True),
    (True,  False, False),
    (False, True,  False),
])
def test_filter_data_combines_both_sides(valid_in, valid_out, expected):
    m = _Proc()
    m.filter_input  = lambda inp: np.asarray(valid_in)
    m.filter_output = lambda out: np.asarray(valid_out)
    m._init_processing_functions()

    assert hasattr(m, 'filter_data')
    assert bool(m.filter_data('i', 'o')) is expected


# --- get_dataset_config ------------------------------------------------------------

def test_dataset_config_predict_uses_input_hooks():
    m = _Proc()
    m.prepare_input = lambda inp, ** kw: inp
    m.process_input = lambda inp, ** kw: inp
    m._init_processing_functions()

    cfg = m.get_dataset_config('predict')
    assert cfg['prepare_fn'] is m.prepare_input
    assert cfg['process_fn'] is m.process_input
    assert cfg['shuffle'] is False
    assert cfg['cache']   is False

def test_dataset_config_train_uses_data_hooks_and_shuffles():
    m = _Proc()
    m.prepare_input = lambda inp, ** kw: inp
    m._init_processing_functions()

    cfg = m.get_dataset_config('train')
    assert cfg['prepare_fn'] is m.prepare_data
    assert cfg['shuffle'] is True

def test_dataset_config_mode_specific_override():
    m = _Proc()
    cfg = m.get_dataset_config('train', batch_size = 8, train_batch_size = 16)
    # `{mode}_<key>` wins over the plain `<key>`
    assert cfg['batch_size'] == 16
