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

""" Pure-``pytest`` tests for :mod:`models.core.mixins.classification_mixin`.

`ClassificationModelMixin` is label bookkeeping : `_init_labels` (padding to `nb_class`,
`label_to_idx`), `get_label_id` and `get_config_labels`. Only the `tf.lookup`-based graph
branch of `get_label_id` needs tensorflow ; the `str` / `int` / `list` branches exercised
here are pure-python (numpy only), so these tests stay unmarked.

The mixin is tested standalone (no `BaseModel`) : `_init_labels` is the only setup needed.
"""

import numpy as np
import pytest

from models.core.mixins.classification_mixin import ClassificationModelMixin


class _Clf(ClassificationModelMixin):
    """ Bare host for the mixin. """


def test_init_labels_basic():
    m = _Clf()
    m._init_labels(['cat', 'dog'])
    assert m.labels == ['cat', 'dog']
    assert m.nb_class == 2
    assert m.label_to_idx == {'cat' : 0, 'dog' : 1}

def test_init_labels_single_is_wrapped():
    m = _Clf()
    m._init_labels('cat')
    assert m.labels == ['cat']
    assert m.nb_class == 1

def test_init_labels_pads_to_nb_class():
    m = _Clf()
    m._init_labels(['cat'], nb_class = 3)
    assert m.labels == ['cat', '', '']
    assert m.nb_class == 3


@pytest.mark.parametrize('data, expected', [
    pytest.param('dog',                 1,            id = 'known_str'),
    pytest.param('unknown',            -1,            id = 'unknown_str'),
    pytest.param(2,                    -1,            id = 'unknown_int'),
    pytest.param(['cat', 'dog', 'x'],  [0, 1, -1],   id = 'list'),
    pytest.param(np.array(['cat']),    [0],          id = 'ndarray'),
])
def test_get_label_id(data, expected):
    m = _Clf()
    m._init_labels(['cat', 'dog'])
    assert m.get_label_id(data) == expected

def test_get_label_id_from_dict():
    m = _Clf()
    m._init_labels(['cat', 'dog'])
    assert m.get_label_id({'label' : 'dog'}) == 1


def test_get_config_labels():
    m = _Clf()
    m._init_labels(['cat', 'dog'])
    assert m.get_config_labels() == {'labels' : ['cat', 'dog'], 'nb_class' : 2}
