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

from ...keras import ops
from .vector_index import VectorIndex

class KerasIndex(VectorIndex):
    def concat_embeddings(self, embeddings):
        """ Concatenate a list of embeddings """
        return ops.concatenate(embeddings, axis = 0)

    def prepare_embeddings(self, embeddings):
        """ Converts `embeddings` to a 2D array of the correct format """
        embeddings = ops.convert_to_tensor(embeddings)
        if len(embeddings.shape) == 1: embeddings = ops.expand_dims(embeddings, 0)
        return embeddings
    
    def __getitem__(self, index):
        """ Return the embeddings at the given `index`(es) """
        if isinstance(index, (int, slice)) or ops.is_bool(index):
            return self.embeddings[index]
        else:
            return ops.gather(self.embeddings, index, axis = 0)