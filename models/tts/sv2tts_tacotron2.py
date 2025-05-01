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

import os

from .tacotron2 import Tacotron2
from utils.keras import TensorSpec
from utils import load_embeddings, save_embeddings, select_embedding, pad_batch, is_dataframe

class SV2TTSTacotron2(Tacotron2):
    """
        This model is an extension of `Tacotron2` with an additional inputs : the speaker embedding.
        A `speaker embedding` is an abstract representation of a speaker prosody (typically produced by a model, such as a `Siamese Network`). 
        Adding this embedding as inputs aims to force the model to produce a mel-spectrogram with the same prosody as the speaker. 
    """
    _directories    = {
        ** Tacotron2._directories, 'embeddings_dir' : '{root}/{self.name}/embeddings'
    }

    def __init__(self,
                 lang,
                 *,
                 
                 encoder_name   = None,
                 embedding_dim  = None,
                 use_label_embedding    = None,
                 
                 ** kwargs
                ):
        self.__encoder      = None
        self.__embeddings   = None

        self.encoder_name   = encoder_name
        self.embedding_dim  = embedding_dim
        self.use_label_embedding    = use_label_embedding

        super().__init__(lang = lang, ** kwargs)
    
    def build(self, ** kwargs):
        super().build(
            encoder_speaker_embedding_dim   = self.embedding_dim, ** kwargs
        )
    
    @property
    def has_default_embedding(self):
        return os.path.exists(self.default_embedding_file)
    
    @property
    def default_embedding_file(self):
        candidates = os.listdir(self.embeddings_dir) if os.path.exists(self.embeddings_dir) else []
        candidates = [f for f in candidates if not f.startswith(('.', '_'))]
        if len(candidates) == 1:
            return os.path.join(self.embeddings_dir, candidates[0])
        return os.path.join(self.embeddings_dir, 'embeddings')
                
    @property
    def embeddings(self):
        if self.__embeddings is None and self.has_default_embedding:
            self.__embeddings = load_embeddings(self.default_embedding_file)
        return self.__embeddings
    
    @embeddings.setter
    def embeddings(self, value):
        self.__embeddings = value
        if not self.has_default_embedding:
            self.set_default_embeddings(value)

    @property
    def encoder(self):
        if self.__encoder is None:
            from models import get_pretrained
            self.__encoder = get_pretrained(self.encoder_name)
            self.__encoder.model.trainable = False
        return self.__encoder

    @property
    def input_signature(self):
        sign = super().input_signature
        return sign[:-2] + (
            TensorSpec(shape = (None, self.embedding_dim), dtype = 'float32'),
        ) + sign[-2:]
    
    def __str__(self):
        des = super().__str__()
        des += "- Encoder : {}\n".format(self.encoder_name)
        des += "- Embedding dim : {}\n".format(self.embedding_dim)
        return des
    
    def compile(self, *, loss_config = {}, ** kwargs):
        if 'mel_loss' in loss_config and 'similarity' in loss_config['mel_loss']:
            raise NotImplementedError()
        
        super().compile(loss_config = loss_config, ** kwargs)
    
    def infer(self, text, *, embeddings = 0, ** kwargs):
        if embeddings is None or isinstance(embeddings, (int, str, dict)):
            embeddings = self.select_embedding(embeddings)
        
        return super().infer(text, embeddings = embeddings, ** kwargs)
    
    def set_default_embeddings(self, embeddings, filename = None):
        self.add_embeddings(embeddings, 'embeddings')
    
    def add_embeddings(self, embeddings, name):
        save_embeddings(
            filename    = name,
            embeddings  = embeddings,
            directory   = self.embeddings_dir
        )
    
    def select_embedding(self, embeddings = None, mode = None):
        if not hasattr(embeddings, 'shape'):
            if mode is None: mode = embeddings
            embeddings = self.embeddings
        
        if mode is None: mode = {'mode' : 'mean' if self.use_label_embedding else 'random'}
        elif not isinstance(mode, dict): mode = {'mode' : mode}
        
        return select_embedding(embeddings, ** mode)

    def embed(self, data, ** kwargs):
        return self.encoder.embed(data, ** kwargs)
    
    def embed_dataset(self, * args, ** kwargs):
        return self.encoder.embed_dataset(* args, ** kwargs)

    
    def get_speaker_embedding(self, data):
        """ This function is used in `encode_data` and must return a single embedding """
        if isinstance(data, list):
            return pad_batch([self.get_speaker_embedding(d) for d in data])
        elif is_dataframe(data):
            return pad_batch([self.get_speaker_embedding(row) for _, row in data.iterrows()])
        
        if isinstance(data, dict):
            if not self.use_label_embedding:
                key = 'embedding'
            elif isinstance(self.use_label_embedding, str):
                key = self.use_label_embedding
            else:
                key = 'speaker_embedding'
            return data[key]
        
        elif hasattr(data, 'shape'):
            return data
        else:
            raise ValueError('Unsupported embedding : {}'.format(data))
    
    def prepare_data(self, data):
        inputs, outputs = super().prepare_data(data)
        
        embedded_speaker = self.get_speaker_embedding(data)
        
        return inputs[:-2] + (embedded_speaker, ) + inputs[-2:], outputs
    
    def get_dataset_config(self, * args, ** kwargs):
        config = super().get_dataset_config(* args, ** kwargs)
        inp, out = config['pad_kwargs']['padding_values']
        config['pad_kwargs']['padding_values'] = (
            inp[:-2] + (0., ) + inp[-2:], out
        )
        
        return config
    
    def predict(self, * args, embeddings = None, mode = None, overwrite = True, ** kwargs):
        embeddings = self.select_embedding(embeddings, mode)
        
        return super().predict(
            * args, embeddings = embeddings, overwrite = overwrite, ** kwargs
        )
    
    def get_config(self):
        return {
            ** super().get_config(),
            'encoder_name'  : self.encoder_name,
            'embedding_dim' : self.embedding_dim,
            'use_label_embedding'   : self.use_label_embedding
        }
