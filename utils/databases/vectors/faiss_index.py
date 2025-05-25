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

class FaissIndex(VectorIndex):
    def __init__(self,
                 embedding_dim,
                 * args,
                 
                 metric = 'cosine',
                 
                 index_type = None,
                 retrain_fraction   = 0.1,
                 
                 use_gpu    = True,
                 gpu_memory = None,
                 
                 ** kwargs
                ):
        import faiss
        
        super().__init__(embedding_dim, metric = metric, ** kwargs)
        
        self.use_gpu    = use_gpu
        self.index_type = index_type
        
        self.last_training  = 0
        self.retrain_fraction   = retrain_fraction
        
        if self.metric == 'cosine':
            index   = faiss.IndexFlatIP(self.embedding_dim)
        elif self.metric == 'euclidian':
            index   = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError('FaissIndex only supports euclidian or cosine distance')
        
        if index_type:
            index = getattr(faiss, index_type)(index, self.embedding_dim, * args, ** kwargs)
        
        if use_gpu:
            gpu = faiss.StandardGpuResources()
            if gpu_memory: gpu.setTempMemory(gpu_memory * (1024 ** 2))

            index = faiss.index_cpu_to_gpu(gpu, 0, index)

        self.vectors = index

    def _normalize(self, vectors, normalized = False, ** _):
        if self.metric == 'cosine' and not normalized:
            vectors = ops.normalize(vectors)
        return ops.convert_to_numpy(vectors)
    
    def add_data(self, vectors, ** kwargs):
        if len(vectors.shape) == 1: vectors = vectors[None]
        
        vectors = self._normalize(vectors, ** kwargs)
        
        n_added = len(vectors) + len(self) - self.last_training
        if self.index_type and len(self) and n_added / self.last_training >= self.retrain_fraction:
            vectors = np.concatenate([self.vectors.reconstruct_n(0), vectors], axis = 0)
            self.vectors.reset()
            self.vectors.is_trained = False

        if not self.vectors.is_trained:
            self.vectors.train(vectors)
            self.last_training = len(vectors)
        
        self.vectors.add(vectors)
    
    def get_data(self, index):
        raise NotImplementedError()
    
    def remove_data(self, index):
        raise NotImplementedError()

    def save_data(self, filename):
        import faiss
        
        faiss.write_index(self.vectors, filename)
    
    def load_data(self, filename):
        import faiss
        
        return faiss.load_index(filename)

    def top_k(self, query, k = 10, ** kwargs):
        query = self._normalize(query, ** kwargs)
        return self.vectors.search(query, k = k, ** kwargs)
