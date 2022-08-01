
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
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
import pandas as pd
import tensorflow as tf

from models.interfaces.base_model import BaseModel
from utils import load_embedding, save_embeddings, select_embedding, sample_df

_default_embeddings_filename = 'default_embeddings'

class BaseEmbeddingModel(BaseModel):
    def _init_embedding(self,
                        encoder_name,
                        embedding_dim,
                        * args,
                        use_label_embedding = True,
                        ** kwargs
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
    
    def _init_folders(self):
        super(BaseEmbeddingModel, self)._init_folders()
        os.makedirs(self.embedding_dir, exist_ok = True)
    
    @property
    def embedding_dir(self):
        return os.path.join(self.folder, 'embeddings')
    
    @property
    def has_default_embedding(self):
        return len(os.listdir(self.embedding_dir)) > 0
    
    @property
    def default_embedding_file(self):
        return os.path.join(self.embedding_dir, _default_embeddings_filename)
    
    @property
    def embedding_signature(self):
        return tf.TensorSpec(shape = (None, self.embedding_dim), dtype = tf.float32)
    
    @property
    def training_hparams(self):
        return super().training_hparams(
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
            self.__encoder.get_model().trainable = False
        return self.__encoder
    
    def _str_embedding(self):
        des = "- Embedding's dim : {}\n".format(self.embedding_dim)
        if self.encoder_name is not None:
            des += "- Encoder's name : {}\n".format(self.encoder_name)
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
        
    def embed(self, audios, ** kwargs):
        return self.encoder.embed(audios, ** kwargs)
    
    def get_embedding(self, data, label_embedding_key = 'label_embedding'):
        """ This function is used in `encode_data` and must return a single embedding """
        def load_np(filename):
            if hasattr(filename, 'numpy'): filename = filename.numpy().decode('utf-8')
            return np.load(filename)
        
        embedding = data
        if isinstance(data, (dict, pd.Series)):
            embedding_key = label_embedding_key
            if not self.use_label_embedding and 'embedding' in data:
                embedding_key = 'embedding'
            embedding = data[embedding_key]
        
        if isinstance(embedding, tf.Tensor) and embedding.dtype == tf.string:
            embedding = tf.py_function(load_np, [embedding], Tout = tf.float32)
            embedding.set_shape([self.embedding_dim])
        elif isinstance(embedding, str):
            embedding = np.load(embedding)
        
        return embedding
        
    def maybe_augment_embedding(self, embedding):
        if self.augment_embedding:
            embedding = tf.cond(
                self.augment_embedding and tf.random.uniform(()) < self.augment_prct,
                lambda: embedding + tf.random.normal(tf.shape(embedding), stddev = 0.025),
                lambda: embedding
            )
        return embedding
        
    def train(self, x, * args, ** kwargs):
        if isinstance(x, pd.DataFrame) and not self.has_default_embedding:
            self.set_default_embeddings(sample_df(x, n = 50, n_sample = 10))
        
        return super().train(x, * args, ** kwargs)
    
    def get_config_embedding(self):
        return {
            'encoder_name'  : self.encoder_name,
            'embedding_dim' : self.embedding_dim,
            'use_label_embedding'   : self.use_label_embedding
        }
