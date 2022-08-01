
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

from loggers import timer
from utils import select_embedding
from models.tts.tacotron2 import Tacotron2
from models.interfaces.base_embedding_model import BaseEmbeddingModel
from models.weights_converter import partial_transfer_learning

class SV2TTSTacotron2(BaseEmbeddingModel, Tacotron2):
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
        return self.text_signature + (
            self.embedding_signature,
            self.audio_signature,
            tf.TensorSpec(shape = (None,), dtype = tf.int32),
        )
    
    def __str__(self):
        des = super().__str__()
        des += self._str_embedding()
        return des
    
    def compile(self, loss_config = {}, ** kwargs):
        if 'mel_loss' in loss_config and 'similarity' in loss_config['mel_loss']:
            self.load_encoder()
            loss_config.setdefault('similarity_function', self.pred_similarity)
        
        super().compile(loss_config = loss_config, ** kwargs)
    
    @timer(name = 'inference')
    def infer(self, text, text_length, spk_embedding, * args, ** kwargs):
        if tf.rank(spk_embedding) == 1:
            spk_embedding = tf.expand_dims(spk_embedding, axis = 0)
        if not isinstance(text, str) and tf.shape(spk_embedding)[0] < tf.shape(text)[0]:
            spk_embedding = tf.tile(spk_embedding, [tf.shape(text)[0], 1])
        
        return super().infer([text, spk_embedding], text_length, * args, ** kwargs)
    
    def get_speaker_embedding(self, data):
        """ This function is used in `encode_data` and returns a single embedding """
        return self.get_embedding(data, label_embedding_key = 'speaker_embedding')
        
    def encode_data(self, data):
        text, text_length, mel_input, mel_length, mel_output, gate = super().encode_data(data)
        
        embedded_speaker = self.get_speaker_embedding(data)
        
        return text, text_length, embedded_speaker, mel_input, mel_length, mel_output, gate
        
    def filter_data(self, text, text_length, embedded_speaker, mel_input, 
                    mel_length, mel_output, gate):
        return super().filter_data(
            text, text_length, mel_input, mel_length, mel_output, gate
        )
    
    def augment_embedding(self, embedding):
        return tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: embedding + tf.random.normal(tf.shape(embedding), stddev = 0.025),
            lambda: embedding
        )
        
    def augment_data(self, text, text_length, embedded_speaker, mel_input, 
                     mel_length, mel_output, gate):
        mel_input = self.augment_audio(mel_input)
        embedded_speaker    = self.maybe_augment_embedding(embedded_speaker)
        
        return text, text_length, embedded_speaker, mel_input, mel_length, mel_output, gate
        
    def preprocess_data(self, text, text_length, embedded_speaker, mel_input, 
                        mel_length, mel_output, gate):
        (text, text_length, mel_input, mel_length), target = super().preprocess_data(
            text, text_length, mel_input, mel_length, mel_output, gate
        )
        
        return (text, text_length, embedded_speaker, mel_input, mel_length), target
                
    def get_dataset_config(self, ** kwargs):
        config = super().get_dataset_config(** kwargs)
        config['pad_kwargs']    = {
            'padding_values'    : (self.blank_token_idx, 0, 0., 0., 0, 0., 1.)
        }
        
        return config
            
    def embed_and_predict(self, audios, sentences, ** kwargs):
        embeddings = self.embed(audios)
        
        return self.predict(sentences, embeddings = embeddings, ** kwargs)
    
    def get_pipeline(self, * args, embeddings = None, embedding_mode = {}, overwrite = True,
                     ** kwargs):
        """
            See `Tacotron2.get_pipeline` for full information
            
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
    
