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

""" Pure-``pytest`` tests for :mod:`custom_train_objects.generators`.

Focuses on ``GE2EGenerator`` : the speaker-verification sampler that, from a
``(id, filename)`` dataframe, builds *rounds* of ``n_samples`` files per id. The
tests assert on the grouping invariants (each group is single-id, right shape /
length / dtype), the ``output_signature`` wrapping and the ``set_batch_size``
validation — none of which require really loading a file (``load_fn`` is identity).

``GE2EGenerator`` subclasses keras' ``PyDataset`` -> marked ``keras`` and skipped
when the backend is missing.
"""

import numpy as np
import pandas as pd
import pytest

keras = pytest.importorskip('keras')

from utils.keras import TensorSpec
from custom_train_objects.generators.ge2e_generator import GE2EGenerator

pytestmark = pytest.mark.keras

N_SAMPLES   = 2
N_IDS       = 3
N_FILES     = 4


@pytest.fixture
def dataset():
    rows = [
        {'id' : id_name, 'filename' : '{}_{}.wav'.format(id_name, j)}
        for id_name in ('a', 'b', 'c')
        for j in range(N_FILES)
    ]
    return pd.DataFrame(rows)

@pytest.fixture
def generator(dataset):
    """ A fresh generator (``set_batch_size`` mutates state, so never share). """
    return GE2EGenerator(
        dataset, N_SAMPLES,
        id_column   = 'id',
        file_column = 'filename',
        load_fn     = lambda data: data,
        output_signature = TensorSpec(shape = (None, ), dtype = 'float32'),
        random_state = 0,
        cache_size   = 0,
    )


# --- grouping invariants -----------------------------------------------------------

def test_groups_shape_and_dtype(generator):
    assert generator.groups.ndim == 2
    assert generator.groups.shape[1] == N_SAMPLES
    assert generator.groups.shape[0] == len(generator.group_ids)
    assert generator.groups.dtype == np.int32

def test_len_is_groups_times_samples(generator):
    assert len(generator) == len(generator.groups) * N_SAMPLES

def test_each_group_is_single_id(generator, dataset):
    """ Every sampled group must contain files of one and only one id. """
    id_of = {idx : name for name, idx in generator.ids.items()}
    for group, gid in zip(generator.groups, generator.group_ids):
        rows = dataset.iloc[group]
        assert len(rows) == N_SAMPLES
        assert set(rows['id']) == {id_of[gid]}

def test_group_ids_within_range(generator):
    assert set(np.unique(generator.group_ids)).issubset(set(range(N_IDS)))

def test_all_files_belong_to_dataset(generator, dataset):
    assert set(generator.unique_files).issubset(set(dataset['filename']))


# --- output signature --------------------------------------------------------------

def test_output_signature_appends_id_spec(generator):
    sign = generator.output_signature
    assert isinstance(sign, tuple) and len(sign) == 2
    assert tuple(sign[1].shape) == (1, )

def test_missing_output_signature_raises(dataset):
    gen = GE2EGenerator(
        dataset, N_SAMPLES, id_column = 'id', file_column = 'filename',
        load_fn = lambda data: data, random_state = 0, cache_size = 0,
    )
    with pytest.raises(NotImplementedError):
        gen.output_signature


# --- __getitem__ -------------------------------------------------------------------

def test_getitem_returns_data_and_id(generator):
    out = generator[0]
    assert isinstance(out, tuple) and len(out) == 2
    assert out[1][0] == generator.group_ids[0]


# --- set_batch_size validation -----------------------------------------------------

@pytest.mark.parametrize('batch_size', [
    pytest.param(3, id = 'not_multiple'),       # 3 % 2 != 0
    pytest.param(N_SAMPLES, id = 'too_small'),  # batch_size <= n_samples
])
def test_set_batch_size_invalid(generator, batch_size):
    with pytest.raises(ValueError):
        generator.set_batch_size(batch_size)

def test_set_batch_size_valid(generator):
    generator.set_batch_size(2 * N_SAMPLES)         # 4 : multiple of 2 and > 2
    assert generator.batch_size == 2 * N_SAMPLES
    assert generator.shuffle_groups is not None
