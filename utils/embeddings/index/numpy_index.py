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

import numpy as np

from ...keras import ops
from .vector_index import VectorIndex

class NumpyIndex(VectorIndex):
    def concat_embeddings(self, embeddings):
        """ Concatenate a list of embeddings """
        return np.concatenate(embeddings, axis = 0)
    
    def prepare_embeddings(self, embeddings):
        """ Converts `embeddings` to a 2D array of the correct format """
        embeddings = ops.convert_to_numpy(embeddings)
        if embeddings.ndim == 1: embeddings = np.expand_dims(embeddings, 0)
        return embeddings