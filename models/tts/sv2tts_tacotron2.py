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

from .tacotron2 import Tacotron2
from utils.keras_utils import ops
from utils import select_embedding, is_dataframe
from models.interfaces.base_embedding_model import BaseEmbeddingModel

class SV2TTSTacotron2(BaseEmbeddingModel, Tacotron2):
    """
        This model is an extension of `Tacotron2` with an additional inputs : the speaker embedding.
        A `speaker embedding` is an abstract representation of a speaker prosody (typically produced by a model, such as a `Siamese Network`). 
        Adding this embedding as inputs aims to force the model to produce a mel-spectrogram with the same prosody as the speaker. 
    """
    def __init__(self,
                 lang,
                 encoder_name   = None,
                 embedding_dim  = None,
                 use_label_embedding    = None,
                 
                 speaker_encoder_name       = None,
                 speaker_embedding_dim      = None,
                 use_utterance_embedding    = None,
                 
                 ** kwargs
                ):
        """
            `speaker_encoder_name`, `speaker_embedding_dim` and `use_utterance_embedding` are deprecated, please use their associated argument (respectively `encoder_name`, `embedding_dim` and `use_label_embedding`)
        """
        assert encoder_name or speaker_encoder_name
        assert embedding_dim or speaker_embedding_dim
        
        should_update   = False
        if encoder_name is None:
            should_update, encoder_name         = True, speaker_encoder_name
        if embedding_dim is None:
            should_update, embedding_dim        = True, speaker_embedding_dim
        if use_label_embedding is None:
            should_update, use_label_embedding  = True, not use_utterance_embedding
        
        self._init_embedding(
            encoder_name    = encoder_name,
            embedding_dim   = embedding_dim,
            use_label_embedding = use_label_embedding
        )
        super().__init__(lang = lang, ** kwargs)
        
        if should_update: self.save_config()
    
    def build(self, ** kwargs):
        super().build(
            encoder_speaker_embedding_dim   = self.embedding_dim, ** kwargs
        )
    
    @property
    def input_signature(self):
        sign = super().input_signature
        return sign[:-2] + (self.embedding_signature, ) + sign[-2:]
    
    @property
    def training_hparams(self):
        return super().training_hparams(** self.training_hparams_embedding)
    
    def __str__(self):
        return super().__str__() + self._str_embedding()
    
    def compile(self, loss_config = {}, ** kwargs):
        if 'mel_loss' in loss_config and 'similarity' in loss_config['mel_loss']:
            self.load_encoder()
            loss_config.setdefault('similarity_function', self.pred_similarity)
        
        super().compile(loss_config = loss_config, ** kwargs)
    
    def infer(self, inputs, spk_embedding = None, ** kwargs):
        if spk_embedding is None or isinstance(spk_embedding, (int, str, dict)):
            spk_embedding = self.select_embedding(spk_embedding)
        
        if not isinstance(inputs, (list, tuple)):
            if isinstance(inputs, str):
                inputs = ops.expand_dims(self.get_input(inputs), axis = 0)
            elif ops.ndim(inputs) == 1:
                inputs = ops.expand_dims(inputs, axis = 0)
        
        if ops.rank(spk_embedding) == 1:
            spk_embedding = ops.expand_dims(spk_embedding, axis = 0)
        if ops.shape(spk_embedding)[0] < ops.shape(inputs)[0]:
            spk_embedding = ops.tile(spk_embedding, [ops.shape(inputs)[0], 1])
        
        return super().infer([inputs, spk_embedding], ** kwargs)
    
    def select_embedding(self, embeddings = None, mode = None):
        # load embeddings if needed
        if ops.is_array(embeddings) or is_dataframe(embeddings):
            self.set_embeddings(embeddings)
        elif self.embeddings is None:
            if embeddings is not None: embeddings, mode = None, embeddings
            self.load_embeddings()
        
        if mode is None:                mode = 'mean' if self.use_label_embedding else 'random'
        if not isinstance(mode, dict):  mode = {'mode' : mode}
        
        selected_embedding = select_embedding(self.embeddings, ** mode)
        selected_embedding = ops.expand_dims(
            ops.cast(selected_embedding, 'float32'), axis = 0
        )
        return selected_embedding
    
    def get_speaker_embedding(self, data):
        """ This function is used in `prepare_data` and returns a single embedding """
        return self.get_embedding(data, label_embedding_key = 'speaker_embedding')
        
    def prepare_data(self, data):
        inputs, outputs = super().prepare_data(data)
        
        embedded_speaker = self.get_speaker_embedding(data)
        
        return inputs[:-2] + (embedded_speaker, ) + inputs[-2:], outputs
    
    def augment_data(self, inputs, outputs):
        inputs, outputs = super().augment_data(inputs, outputs)

        embedded_speaker    = self.maybe_augment_embedding(inputs[-3])
        
        return inputs[:-3] + (embedded_speaker, ) + inputs[-2:], outputs
        
    def get_dataset_config(self, * args, ** kwargs):
        config = super().get_dataset_config(* args, ** kwargs)
        inp, out = config['pad_kwargs']['padding_values']
        config['pad_kwargs']['padding_values'] = (
            inp[:-2] + (0., ) + inp[-2:], out
        )
        
        return config
            
    def embed_and_predict(self, audios, sentences, ** kwargs):
        embeddings = self.embed(audios)
        
        return self.predict(sentences, embeddings = embeddings, ** kwargs)
    
    def predict(self, * args, embeddings = None, mode = None, overwrite = True, ** kwargs):
        selected_embedding = self.select_embedding(embeddings, mode)
        
        return super().predict(
            * args, spk_embedding = selected_embedding, overwrite = overwrite, ** kwargs
        )
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update(self.get_config_embedding())
        
        return config
    