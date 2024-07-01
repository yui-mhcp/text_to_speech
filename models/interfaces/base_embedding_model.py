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
import logging
import numpy as np
import pandas as pd

from .base_model import BaseModel
from utils.hparams import HParams
from utils.keras_utils import TensorSpec, ops, execute_eagerly
from utils import convert_to_str, load_embedding, save_embeddings, select_embedding, sample_df

logger  = logging.getLogger(__name__)

_default_embeddings_filename = 'default_embeddings'

@execute_eagerly(numpy = True, signature = [TensorSpec(shape = (None, ), dtype = 'float32')])
def _load_np_embedding(filename):
    filename = convert_to_str(filename)
    return np.load(filename)

class BaseEmbeddingModel(BaseModel):
    def _init_embedding(self,
                        encoder_name,
                        embedding_dim,
                        use_label_embedding = True,
                       ):
        """
            Initializes the embedding-related variables
            Arguments :
                - encoder_name  : the model's name that produces embeddings (typically a `Siamese Network`)
                - embedding_dim : the embeddings' dimension
                - use_label_embedding   : whether to use label-based or instance-based embedding
        """
        self.__encoder      = None
        self.__embeddings   = None

        self.encoder_name   = encoder_name
        self.embedding_dim  = embedding_dim
        self.use_label_embedding    = use_label_embedding
    
    @property
    def embedding_dir(self):
        return os.path.join(self.directory, 'embeddings')
    
    @property
    def has_default_embedding(self):
        return os.path.exists(self.embedding_dir) and len(os.listdir(self.embedding_dir)) > 0
    
    @property
    def default_embedding_file(self):
        return os.path.join(self.embedding_dir, _default_embeddings_filename)
    
    @property
    def embedding_signature(self):
        return TensorSpec(shape = (None, self.embedding_dim), dtype = 'float32')
    
    @property
    def training_hparams_embedding(self):
        return HParams(
            augment_embedding   = False,
            use_label_embedding = None
        )
                
    @property
    def embeddings(self):
        return self.__embeddings
    
    @property
    def encoder(self):
        if self.__encoder is None:
            from models import get_pretrained
            self.__encoder = get_pretrained(self.encoder_name)
            self.__encoder.model.trainable = False
        return self.__encoder
    
    def _str_embedding(self):
        des = "- Embedding's dim : {}\n".format(self.embedding_dim)
        if self.encoder_name is not None:
            des += "- Encoder name : {}\n".format(self.encoder_name)
        return des
    
    def pred_similarity(self, y_true, y_pred):
        score = self.encoder([y_true, y_pred])
        return score if not self.encoder.embed_distance else 1. - score
    
    def load_encoder(self, name = None):
        if self.__encoder is not None:
            if name is None or self.__encoder.nom == name:
                return
        
        if name is None and self.encoder_name is None:
            raise ValueError("You must provide the encoder's name !")
        
        if self.encoder_name is None:
            self.encoder_name = name
        else:
            name = self.encoder_name
        
        from models import get_pretrained
        self.__encoder = get_pretrained(name)
    
    def set_default_embeddings(self, embeddings, filename = None):
        self.add_embeddings(embeddings, _default_embeddings_filename)
    
    def add_embeddings(self, embeddings, name):
        save_embeddings(self.embedding_dir, embeddings, embedding_name = name)
    
    def set_embeddings(self, embeddings):
        self.__embeddings = embeddings
        if not self.has_default_embedding:
            self.set_default_embeddings(embeddings)
    
    def load_embeddings(self, directory = None, filename = None, ** kwargs):
        if not self.has_default_embedding and directory is None:
            raise ValueError("No default embeddings available !\n  Use the 'set_default_embeddings()' or 'set_embeddings()' method")
        
        if directory is None:
            directory = self.embedding_dir
            if len(os.listdir(directory)) == 1:
                filename = os.listdir(directory)[0]
        if filename is None:
            filename = _default_embeddings_filename
        
        embeddings = load_embedding(
            directory, embedding_dim = directory, embedding_name = filename, ** kwargs
        )
        
        self.set_embeddings(embeddings)
        
    def embed(self, data, ** kwargs):
        return self.encoder.embed(data, ** kwargs)
    
    def embed_dataset(self, * args, ** kwargs):
        return self.encoder.embed_dataset(* args, ** kwargs)

    def get_embedding(self, data, label_embedding_key = 'label_embedding', key = 'embedding',
                      embed_if_not_exist = True, ** kwargs):
        """ This function is used in `encode_data` and must return a single embedding """
        if isinstance(data, list):
            return ops.stack([self.get_embedding(d) for d in data], axis = 0)
        elif isinstance(data, pd.DataFrame):
            return ops.stack([self.get_embedding(row) for _, row in data.iterrows()], axis = 0)
        
        embedding = data
        if isinstance(data, (dict, pd.Series)):
            embedding_key = label_embedding_key
            if not self.use_label_embedding and key in data:
                embedding_key = key
            if embedding_key in data:
                embedding = data[embedding_key]
            elif embed_if_not_exist:
                logger.info('Embedding key {} is not in data, embedding it !'.format(embedding_key))
                embedding = self.embed(data)
            else:
                logger.error('Embedding key {} is not present in data and `embed_if_not_exist = False`'.format(key))
                return None
        
        elif ops.is_string(embedding):
            embedding = _load_np_embedding(embedding, shape = [self.embedding_dim])
        elif not ops.is_array(embedding):
            if embed_if_not_exist:
                logger.info('Embedding key {} is not in data, embedding it !'.format(key))
                embedding = self.embed(data)
            else:
                logger.error('Unknown embedding type and `embed_if_not_exist = False` (type {}) : {}'.format(type(data), data))
                return None
        
        return embedding
        
    def maybe_augment_embedding(self, embedding):
        if self.augment_embedding:
            return ops.cond(
                ops.random.uniform(()) < self.augment_prct,
                lambda: embedding + ops.random.normal(ops.shape(embedding), stddev = 0.025),
                lambda: embedding
            )
        return embedding
    
    def get_config_embedding(self):
        return {
            'encoder_name'  : self.encoder_name,
            'embedding_dim' : self.embedding_dim,
            'use_label_embedding'   : self.use_label_embedding
        }
