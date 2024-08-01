# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base_vectors_db import BaseVectorsDB
from .dense_vectors import DenseVectors

from utils.keras_utils import ops

def build_vectors_db(vectors, data = None, *, mode = None, ** kwargs):
    assert ops.is_array(vectors) and ops.rank(vectors) in (1, 2, 3)
    if ops.ndim(vectors) == 1: vectors = ops.expand_dims(vectors, axis = 0)
    
    if mode is None:
        if len(vectors.shape) == 2: mode = 'dense'
        elif vectors.shape[2] == 1: mode = 'sparse'
        else:                       mode = 'colbert'
    
    if mode == 'dense':
        return DenseVectors(vectors, data = data, ** kwargs)
    elif mode == 'colbert':
        raise NotImplementedError()
    elif mode == 'sparse':
        raise NotImplementedError()
    else:
        raise ValueError('Unsupported vectors mode : {}'.format(mode))