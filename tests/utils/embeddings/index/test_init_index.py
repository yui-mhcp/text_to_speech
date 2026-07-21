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

""" Pure-``pytest`` tests for :func:`utils.embeddings.index.init_index` (the
factory that resolves a backend name / instance into a :class:`VectorIndex`). """

import numpy as np
import pytest

from utils.embeddings.index import init_index, NumpyIndex


def test_init_index_from_string():
    assert isinstance(init_index('numpy'), NumpyIndex)

def test_init_index_default_is_numpy():
    assert isinstance(init_index(), NumpyIndex)

def test_init_index_passthrough_instance():
    index = NumpyIndex(metric = 'euclidian')
    assert init_index(index) is index

def test_init_index_unknown_raises():
    with pytest.raises(ValueError):
        init_index('does_not_exist')

def test_init_index_filename_and_embeddings_raises():
    with pytest.raises(ValueError):
        init_index('numpy', filename = 'foo.npy', embeddings = np.zeros((2, 3), 'float32'))


# --- `dict` form : config-only, default backend, optional `index` key --------------

def test_init_index_from_dict_defaults_to_numpy():
    index = init_index({'metric' : 'cosine'})
    assert isinstance(index, NumpyIndex)
    assert index.metric == 'cosine'

def test_init_index_from_dict_with_explicit_backend():
    index = init_index({'index' : 'numpy', 'metric' : 'euclidian'})
    assert isinstance(index, NumpyIndex)
    assert index.metric == 'euclidian'

def test_init_index_from_dict_does_not_mutate_caller():
    config = {'index' : 'numpy', 'metric' : 'cosine'}
    init_index(config)
    assert config == {'index' : 'numpy', 'metric' : 'cosine'}

def test_init_index_from_dict_filename_and_embeddings_raises():
    with pytest.raises(ValueError):
        init_index({'filename' : 'foo.npy', 'embeddings' : np.zeros((2, 3), 'float32')})
