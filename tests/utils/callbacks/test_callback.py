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

""" Pure-``pytest`` tests for the :class:`utils.callbacks.Callback` core protocol.

Covers the ``__call__`` control-flow : ``cond`` gating, ``initializer`` filling,
lazy one-shot ``build`` and — critically — that ``__call__`` **returns** the value
produced by ``apply`` (regression : the return was previously dropped, which broke
``apply_callbacks``' ``entry`` capture).
"""

from ._helpers import RecordingCallback


# --- return propagation (regression) -----------------------------------------------

def test_call_returns_apply_result():
    cb  = RecordingCallback(return_value = 'entry-key')
    assert cb({}, {'x': 1}) == 'entry-key'
    assert len(cb.calls) == 1

def test_apply_receives_infos_and_output():
    cb      = RecordingCallback()
    infos   = {'a': 1}
    output  = {'b': 2}
    cb(infos, output)
    seen_infos, seen_output, _ = cb.calls[0]
    assert seen_infos is infos
    assert seen_output is output


# --- lazy, one-shot build ----------------------------------------------------------

def test_build_is_lazy_and_runs_once():
    cb = RecordingCallback()
    assert not cb.built and cb.build_count == 0

    cb({}, {})
    assert cb.built and cb.build_count == 1

    cb({}, {})
    assert cb.build_count == 1, 'build() must not run again once built'


# --- cond gating -------------------------------------------------------------------

def test_cond_blocks_apply_and_returns_none():
    cb  = RecordingCallback(cond = lambda ** o: o.get('keep', False))
    assert cb({}, {'keep': False}) is None
    assert cb.calls == []
    assert not cb.built, 'a blocked callback must not even build'

def test_cond_true_lets_apply_run():
    cb = RecordingCallback(cond = lambda ** o: o.get('keep', False))
    assert cb({}, {'keep': True}) == 'applied'
    assert len(cb.calls) == 1


# --- initializer -------------------------------------------------------------------

def test_initializer_fills_only_missing_keys():
    cb      = RecordingCallback(initializer = {'b': lambda ** o: o['a'] + 1})
    output  = {'a': 1}
    cb({}, output)
    assert output['b'] == 2

    output = {'a': 1, 'b': 99}
    cb({}, output)
    assert output['b'] == 99, 'initializer must not overwrite an existing key'
