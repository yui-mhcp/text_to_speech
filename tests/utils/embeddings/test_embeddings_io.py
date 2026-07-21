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

""" Pure-``pytest`` tests for :mod:`utils.embeddings.embeddings_io`.

Covers the ``embeddings_to_np`` conversions (numpy / tensor / dataframe / string),
a ``save_embeddings`` -> ``load_embeddings`` round-trip in ``.npy`` and the
extension helpers. Tolerant comparisons go through the ``asserts`` fixture.

The ``tensor`` conversion case is marked ``keras`` (auto-skipped when keras is
missing) ; every other case is keras-free.
"""

import os
import numpy as np
import pytest

from utils.embeddings import embeddings_to_np, save_embeddings, load_embeddings
from utils.embeddings.embeddings_io import _get_embeddings_file_ext

MAX_ERR = 1e-5


# --- `embeddings_to_np` ------------------------------------------------------------
# Each case is a *builder* (called inside the test) so the keras-only cases never
# touch keras at collection time.

def _case_numpy():
    x = np.reshape(np.arange(64), [4, 16])
    return x, x

def _case_tensor():
    import keras.ops as K
    x = np.reshape(np.arange(64), [4, 16]).astype('float32')
    return K.convert_to_tensor(x), x

def _case_dataframe():
    import pandas as pd
    x  = np.reshape(np.arange(64), [4, 16])
    df = pd.DataFrame([{'id' : i, 'embedding' : emb} for i, emb in enumerate(x)])
    return df, x

def _case_string_1d():
    v = np.arange(64) / 3.
    return str(v), v.astype(np.float32)

_TO_NP_CASES = [
    pytest.param(_case_numpy,     id = 'numpy'),
    pytest.param(_case_tensor,    id = 'tensor', marks = pytest.mark.keras),
    pytest.param(_case_dataframe, id = 'dataframe'),
    pytest.param(_case_string_1d, id = 'string_1d'),
]

@pytest.mark.parametrize('case', _TO_NP_CASES)
def test_embeddings_to_np(case, asserts):
    inputs, target = case()
    result = embeddings_to_np(inputs)
    asserts.assert_array(result)
    asserts.assert_equal(target, result, max_err = MAX_ERR)


# --- `save_embeddings` / `load_embeddings` round-trip ------------------------------

def test_save_load_roundtrip_npy(temp_dir, asserts):
    emb  = np.arange(40).reshape(8, 5).astype(np.float32)
    path = os.path.join(temp_dir, 'roundtrip.npy')
    try:
        saved = save_embeddings(path, emb)
        assert os.path.exists(saved)

        loaded = load_embeddings(path)
        asserts.assert_equal(emb, loaded, max_err = MAX_ERR)
    finally:
        if os.path.exists(path): os.remove(path)

def test_save_embeddings_unsupported_extension_raises(temp_dir):
    with pytest.raises(ValueError):
        save_embeddings(os.path.join(temp_dir, 'bad.txt'), np.zeros((2, 3), 'float32'))


# --- `_get_embeddings_file_ext` ----------------------------------------------------

def test_get_embeddings_file_ext(temp_dir):
    base = os.path.join(temp_dir, 'ext_probe')
    path = base + '.npy'
    np.save(path, np.zeros((2, 3), 'float32'))
    try:
        assert _get_embeddings_file_ext(base) == '.npy'
        assert _get_embeddings_file_ext(os.path.join(temp_dir, 'does_not_exist')) is None
    finally:
        if os.path.exists(path): os.remove(path)
