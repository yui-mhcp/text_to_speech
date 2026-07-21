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

""" Pure-``pytest`` tests for :mod:`models.core.mixins.checkpoint_mixin`.

Only the **keras-free** persistence surface is covered : everything guarded by
`self.runtime == 'keras'` (weights / keras-config (de)serialization, `_restore_model`)
is out of scope here. We therefore drive the mixin through a `DummyModel` on the
inference-only `'fake'` runtime, for which `save` only writes `config.json`.
"""

import os

from utils import load_json
from tests.models.core._helpers import DummyModel


def test_init_persistence_defaults(isolated_saving_dir):
    m = DummyModel(name = 'persist_defaults', save = False, runtime = 'fake')
    assert m.serialized_compile_config is None
    assert m._history is None
    assert m._checkpoint_manager is None


def test_save_disabled_is_a_noop(isolated_saving_dir):
    m = DummyModel(name = 'save_off', save = False, runtime = 'fake')
    m.save()   # `_save` is False -> returns immediately
    assert not os.path.exists(m.config_file)


def test_fake_runtime_persists_only_config(isolated_saving_dir):
    # `save = True` triggers persistence during `__init__` ; the keras-only artefacts
    # (weights config, history) must NOT be written for an inference-only runtime.
    m = DummyModel(name = 'cfg_only', save = True, runtime = 'fake')
    assert os.path.exists(m.config_file)
    assert not os.path.exists(m.config_models_file)
    assert not os.path.exists(m.history_file)


def test_save_config_roundtrip(isolated_saving_dir):
    m   = DummyModel(name = 'rt', save = True, runtime = 'fake', pretrained_name = 'base')
    cfg = load_json(m.config_file)
    assert cfg['class_name'] == 'DummyModel'
    assert cfg['config']['name'] == 'rt'
    assert cfg['config']['runtime'] == 'fake'
    assert cfg['config']['pretrained_name'] == 'base'
