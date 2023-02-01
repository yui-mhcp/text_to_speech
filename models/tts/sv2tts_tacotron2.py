
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

import tensorflow as tf

from utils import select_embedding
from models.tts.tacotron2 import Tacotron2
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
    
    def _build_model(self, **kwargs):
        super()._build_model(
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
    
    def infer(self, text, spk_embedding, * args, ** kwargs):
        if isinstance(text, str):
            text    = tf.expand_dims(self.encode_text(text), axis = 0)
        elif len(tf.shape(text)) == 1:
            text    = tf.expand_dims(text, axis = 0)
        
        if len(tf.shape(spk_embedding)) == 1:
            spk_embedding = tf.expand_dims(spk_embedding, axis = 0)
        if tf.shape(spk_embedding)[0] < tf.shape(text)[0]:
            spk_embedding = tf.tile(spk_embedding, [tf.shape(text)[0], 1])
        
        return super().infer([text, spk_embedding], * args, ** kwargs)
    
    def get_speaker_embedding(self, data):
        """ This function is used in `encode_data` and returns a single embedding """
        return self.get_embedding(data, label_embedding_key = 'speaker_embedding')
        
    def encode_data(self, data):
        inputs, outputs = super().encode_data(data)
        
        embedded_speaker = self.get_speaker_embedding(data)
        
        return inputs[:-2] + (embedded_speaker, ) + inputs[-2:], outputs
    
    def augment_data(self, inputs, outputs):
        inputs, outputs = super().augment_data(inputs, outputs)

        embedded_speaker    = self.maybe_augment_embedding(inputs[-3])
        
        return inputs[:-3] + (embedded_speaker, ) + inputs[-2:], outputs
        
    def get_dataset_config(self, ** kwargs):
        config = super().get_dataset_config(** kwargs)
        config['pad_kwargs']    = {
            'padding_values'    : (
                (self.blank_token_idx, 0., self.pad_mel_value, 0), (self.pad_mel_value, 1.)
            )
        }
        
        return config
            
    def embed_and_predict(self, audios, sentences, ** kwargs):
        embeddings = self.embed(audios)
        
        return self.predict(sentences, embeddings = embeddings, ** kwargs)
    
    def get_pipeline(self, * args, embeddings = None, embedding_mode = {}, overwrite = True,
                     ** kwargs):
        """
            See `Tacotron2.get_pipeline` for all the information
            
            Arguments :
                - args / kwargs : args passed to super().get_pipeline()
                - embeddings    : the embeddings to use as input (only 1 is selected from this set and effectively used)
                - embedding_mode    : kwargs passed to `select_embedding()`
            Return : result of super().predict()
            
            Note : currently we just save the resulting audio for a given sentence but not the speaker / embedding used to generate it. 
            So it can be more interesting to put `overwrite = True` for this model as it is basically used to generate audio with multiple voices (it is the reason why this argument is overriden to `True` in this function)
        """
        # load embeddings if needed
        if embeddings is not None:
            self.set_embeddings(embeddings)
        elif self.embeddings is None:
            self.load_embeddings()
        
        if self.use_label_embedding:
            embedding_mode.setdefault('mode', 'mean')
        
        selected_embedding = select_embedding(self.embeddings, ** embedding_mode)
        selected_embedding = tf.expand_dims(
            tf.cast(selected_embedding, tf.float32), axis = 0
        )
        
        return super().get_pipeline(
            * args, spk_embedding = selected_embedding, overwrite = overwrite, ** kwargs
        )
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update(self.get_config_embedding())
        
        return config
    