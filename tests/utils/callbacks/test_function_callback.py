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

""" Pure-``pytest`` tests for :mod:`utils.callbacks.function_callback`.

Covers ``FunctionCallback`` argument merging (``infos`` + optionally ``output``),
return propagation, and the ``QueueCallback`` ``.put`` wiring.
"""

from utils.callbacks import FunctionCallback, QueueCallback
from ._helpers import FakeQueue


def test_function_callback_merges_infos_and_output_and_returns():
    seen = {}
    def fn(** kwargs):
        seen.update(kwargs)
        return 'result'

    cb  = FunctionCallback(fn)
    assert cb({'a': 1}, {'b': 2}) == 'result'
    assert seen == {'a': 1, 'b': 2}

def test_function_callback_include_outputs_false_passes_only_infos():
    seen = {}
    def fn(** kwargs):
        seen.update(kwargs)

    cb = FunctionCallback(fn, include_outputs = False)
    cb({'a': 1}, {'b': 2})
    assert seen == {'a': 1}, 'output must be excluded when include_outputs=False'

def test_queue_callback_puts_infos_only():
    queue = FakeQueue()
    cb    = QueueCallback(queue)
    cb({'a': 1}, {'b': 2})
    assert queue.items == [{'a': 1}], 'QueueCallback forwards infos (not output) to put'
