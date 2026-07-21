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

""" Pure-``pytest`` tests for the :class:`utils.callbacks.Displayer` base logic.

Only the generic ``data_key`` resolution (output-first, then ``infos``) and the
``max_display`` cap are exercised, via a subclass whose ``display`` merely records —
so no plotting / audio / image backend is touched.
"""

from utils.callbacks import Displayer


class RecordingDisplayer(Displayer):
    def __init__(self, data_key = 'x', ** kwargs):
        super().__init__(data_key, ** kwargs)
        self.displayed = []

    def display(self, _data, ** kwargs):
        self.displayed.append(_data)


def test_display_resolves_key_from_output():
    d = RecordingDisplayer('x')
    d({}, {'x': 42})
    assert d.displayed == [42]

def test_display_falls_back_to_infos():
    d = RecordingDisplayer('x')
    d({'x': 7}, {'y': 1})  # 'x' absent from output -> resolved from infos
    assert d.displayed == [7]

def test_max_display_caps_number_of_displays():
    d = RecordingDisplayer('x', max_display = 1)
    d({}, {'x': 1})
    d({}, {'x': 2})
    assert d.displayed == [1], 'display must stop once max_display is reached'
