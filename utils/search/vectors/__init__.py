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

import os
import numpy as np

from .base_vectors_db import BaseVectorsDB
from .dense_vectors import DenseVectors

from utils.file_utils import load_data
from utils.keras_utils import ops

def build_vectors_db(data, vectors = None, primary_key = None, *, mode = None, ** kwargs):
    if isinstance(data, str):
        return load_vectors_db(data)
    
    if mode is None:    mode = 'dense'
    
    if mode == 'dense':
        return DenseVectors(data, vectors = vectors, primary_key = primary_key, ** kwargs)
    elif mode == 'colbert':
        raise NotImplementedError()
    elif mode == 'sparse':
        raise NotImplementedError()
    else:
        raise ValueError('Unsupported vectors mode : {}'.format(mode))
        
def load_vectors_db(filename):
    if not os.path.exists(filename): return None
    
    config = load_data(filename)
    if isinstance(config, np.ndarray):  config = {'vectors' : config}
    elif hasattr(config, 'to_dict'):    config = config.to_dict('list')
    elif not config or 'vectors' not in config: return None
    
    return DenseVectors(** config)
