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

""" Pure-``pytest`` tests for :mod:`models.core.utils.saving`.

The module is a thin, keras-free layer over the model registry root (`get_saving_dir`)
and small path / json helpers. This covers the new `set_saving_dir` / `saving_dir`
override helpers plus the registry-path accessors.

Note : the package `conftest` already redirects `get_saving_dir` to a temp dir (autouse
`isolated_saving_dir`), so the tests below nest their own overrides on top and always
restore what they changed.
"""

import os
import pytest

from models.core.utils.saving import (
    get_saving_dir, set_saving_dir, saving_dir,
    get_model_dir, is_model_name, get_model_infos, get_model_class, get_model_config,
)


# --- set_saving_dir / saving_dir ---------------------------------------------------

def test_set_saving_dir_returns_previous():
    previous = get_saving_dir()
    old = set_saving_dir('some/dir')
    try:
        assert old == previous
        assert get_saving_dir() == 'some/dir'
    finally:
        set_saving_dir(previous)

def test_saving_dir_context_manager_scopes_and_restores():
    before = get_saving_dir()
    with saving_dir('scoped/dir') as d:
        assert d == 'scoped/dir'
        assert get_saving_dir() == 'scoped/dir'
    assert get_saving_dir() == before

def test_saving_dir_restores_on_error():
    before = get_saving_dir()
    with pytest.raises(RuntimeError):
        with saving_dir('boom/dir'):
            raise RuntimeError('boom')
    assert get_saving_dir() == before


# --- registry path accessors -------------------------------------------------------

def test_get_model_dir(tmp_path):
    with saving_dir(str(tmp_path)):
        assert get_model_dir('foo', 'config.json') == os.path.join(str(tmp_path), 'foo', 'config.json')

def test_is_model_name(tmp_path):
    with saving_dir(str(tmp_path)):
        assert not is_model_name('nope')

        os.makedirs(os.path.join(str(tmp_path), 'model'))
        assert not is_model_name('model')          # directory without config.json

        open(os.path.join(str(tmp_path), 'model', 'config.json'), 'w').close()
        assert is_model_name('model')              # config.json present


# --- get_model_infos / class / config from an object -------------------------------

def test_get_model_infos_from_object():
    class _Obj:
        def get_config(self):
            return {'a' : 1}

    obj = _Obj()
    assert get_model_infos(obj)  == {'class_name' : '_Obj', 'config' : {'a' : 1}}
    assert get_model_class(obj)  == '_Obj'
    assert get_model_config(obj) == {'a' : 1}

def test_get_model_infos_none_is_empty():
    assert get_model_infos(None) == {}
