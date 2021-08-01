import os
import numpy as np
import pandas as pd
import tensorflow as tf

from models.tts.tacotron2 import Tacotron2
from models.siamese.audio_siamese import AudioSiamese
from models.weights_converter import partial_transfer_learning
from utils import load_embedding, save_embeddings, select_embedding, sample_df

_default_embeddings_filename = 'default_embeddings'

class SV2TTSTacotron2(Tacotron2):
    def __init__(self,
                 lang,
                 speaker_encoder_name,
                 speaker_embedding_dim, 
                 use_utterance_embedding    = False,
                 
                 **kwargs
                ):
        self.__embeddings   = None
        self.__speaker_encoder  = None

        self.speaker_embedding_dim  = speaker_embedding_dim
        self.use_utterance_embedding    = use_utterance_embedding
        self.speaker_encoder_name       = speaker_encoder_name
        
        super().__init__(lang = lang, **kwargs)
    
    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.embedding_dir, exist_ok=True)
    
    def init_train_config(self,
                          augment_speaker_embedding = False,
                          use_utterance_embedding = False,
                          ** kwargs
                          ):
        self.augment_speaker_embedding  = augment_speaker_embedding
        self.use_utterance_embedding    = use_utterance_embedding
        
        super().init_train_config(** kwargs)
    
    def _build_model(self, **kwargs):
        super()._build_model(
            encoder_speaker_embedding_dim   = self.speaker_embedding_dim,
            ** kwargs
        )
    
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
    def input_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),
            tf.TensorSpec(shape = (None,), dtype = tf.int32),
            tf.TensorSpec(shape = (None, self.speaker_embedding_dim), dtype = tf.float32),
            tf.TensorSpec(shape = (None, None, self.n_mel_channels), dtype = tf.float32),
            tf.TensorSpec(shape = (None,), dtype = tf.int32),
        )
    
    @property
    def training_hparams(self):
        return super().training_hparams(
            augment_speaker_embedding  = False,
            use_utterance_embedding    = False
        )
    
    @property
    def embedding_dim(self):
        return self.speaker_embedding_dim
                
    @property
    def embeddings(self):
        return self.__embeddings
    
    @property
    def speaker_encoder(self):
        if self.__speaker_encoder is None:
            self.__speaker_encoder = AudioSiamese(nom = self.speaker_encoder_name)
            self.__speaker_encoder.get_model().trainable = False
        return self.__speaker_encoder
    
    def __str__(self):
        des = super().__str__()
        des += "Speaker embedding dim : {}\n".format(self.speaker_embedding_dim)
        if self.speaker_encoder_name is not None:
            des += "Speaker encoder : {}\n".format(self.speaker_encoder_name)
        return des
    
    def compile(self, loss_config = {}, ** kwargs):
        if 'mel_loss' in loss_config and 'similarity' in loss_config['mel_loss']:
            self.speaker_encoder
            loss_config.setdefault('similarity_function', self.pred_similarity)
        
        super().compile(loss_config = loss_config, ** kwargs)
    
    def pred_similarity(self, y_true, y_pred):
        score = self.speaker_encoder([y_true, y_pred])
        return score if not self.speaker_encoder.embed_distance else 1. - score
    
    def load_speaker_encoder(self, name = None):
        if self.__speaker_encoder is not None:
            if name is None or self.__speaker_encoder.nom == name:
                return
        
        if name is None and self.speaker_encoder_name is None:
            raise ValueError("You must provide the name for the speaker encoder !")
        
        if self.speaker_encoder_name is None:
            self.speaker_encoder_name = name
        else:
            name = self.speaker_encoder_name
        
        self.__speaker_encoder = AudioSiamese(nom = name)
    
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
            if len(os.listdir(self.embedding_dir)) == 1:
                filename = os.listdir(self.embedding_dir)[0]
        if filename is None:
            filename = _default_embeddings_filename
        
        embeddings = load_embedding(
            directory,
            embedding_dim   = self.embedding_dim, 
            embedding_name  = filename,
            ** kwargs
        )
        
        self.set_embeddings(embeddings)
        
    def infer(self, text, text_length, spk_embedding, * args, ** kwargs):
        if tf.rank(spk_embedding) == 1:
            spk_embedding = tf.expand_dims(spk_embedding, axis = 0)
        if not isinstance(text, str) and tf.shape(spk_embedding)[0] < tf.shape(text)[0]:
            spk_embedding = tf.tile(spk_embedding, [tf.shape(text)[0], 1])
        
        return super().infer([text, spk_embedding], text_length, * args, ** kwargs)
    
    def embed(self, audios, ** kwargs):
        self.load_speaker_encoder()
        return self.__speaker_encoder.embed(audios, ** kwargs)
    
    def get_speaker_embedding(self, data):
        """ This function is used in `encode_data` and must return a single embedding """
        def load_np(filename):
            return np.load(filename.numpy().decode('utf-8'))
        
        embedding = data
        if isinstance(data, (dict, pd.Series)):
            embedding_key = 'speaker_embedding'
            if self.use_utterance_embedding and 'embedding' in data:
                embedding_key = 'embedding'
            embedding = data[embedding_key]
        
        if isinstance(embedding, tf.Tensor) and embedding.dtype == tf.string:
            embedding = tf.py_function(load_np, [embedding], Tout = tf.float32)
            embedding.set_shape([self.speaker_embedding_dim])
        elif isinstance(embedding, str):
            embedding = np.load(embedding)
        
        return embedding
        
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
        mel_input = self.augment_mel(mel_input)
        if self.augment_speaker_embedding:
            embedded_speaker    = self.augment_embedding(embedded_speaker)
        
        return text, text_length, embedded_speaker, mel_input, mel_length, mel_output, gate
        
    def preprocess_data(self, text, text_length, embedded_speaker, mel_input, 
                        mel_length, mel_output, gate):
        (text, text_length, mel_input, mel_length), target = super().preprocess_data(
            text, text_length, mel_input, mel_length, mel_output, gate
        )
        
        return (text, text_length, embedded_speaker, mel_input, mel_length), target
                
    def get_dataset_config(self, **kwargs):
        config = super().get_dataset_config(**kwargs)
        config['pad_kwargs']    = {
            'padded_shapes'     : (
                (None,), (), (self.speaker_embedding_dim,), (None, self.n_mel_channels), (),
                (None, self.n_mel_channels), (None,)
            ),
            'padding_values'    : (self.blank_token_idx, 0, 0., 0., 0, 0., 1.)
        }
        
        return config
            
    def train(self, x, * args, ** kwargs):
        if isinstance(x, pd.DataFrame) and not self.has_default_embedding:
            self.set_default_embeddings(sample_df(x, n = 50, n_sample = 10))
        
        return super().train(x, * args, ** kwargs)
    
    def embed_and_predict(self,
                          audios,
                          sentences,                
                          ** kwargs
                         ):
        embeddings = self.embed(audios)
        
        return self.predict(sentences, embeddings = embeddings, ** kwargs)
    
    def predict(self,
                * args,
                embeddings  = None,
                embedding_mode  = {},
                overwrite   = True,
                ** kwargs
               ):
        """
            Perform Tacotron-2 inference on all phrases
            Arguments :
                - args / kwargs : args passed to super().predict()
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
        
        if not self.use_utterance_embedding:
            embedding_mode.setdefault('mode', 'mean')
        
        selected_embedding = select_embedding(self.embeddings, ** embedding_mode)
        selected_embedding = tf.expand_dims(
            tf.cast(selected_embedding, tf.float32), axis = 0
        )
        
        return super().predict(
            * args, spk_embedding = selected_embedding, overwrite = overwrite, ** kwargs
        )
    
    def get_config(self, *args, **kwargs):
        config = super().get_config(*args, **kwargs)
        config['speaker_embedding_dim'] = self.speaker_embedding_dim
        config['use_utterance_embedding']   = self.use_utterance_embedding
        config['speaker_encoder_name']      = self.speaker_encoder_name
        
        return config
    
    @classmethod
    def build_from_sv2tts_pretrained(cls, 
                                     nom,
                                     pretrained_name   = 'sv2tts_tacotron2',
                                     ** kwargs
                                    ):
        with tf.device('cpu') as device:        
            pretrained_model = SV2TTSTacotron2(nom = pretrained_name)
        
        kwargs.setdefault('lang', pretrained_model.lang)
        kwargs.setdefault('text_encoder', pretrained_model.text_encoder)
        kwargs.setdefault('speaker_encoder_name', pretrained_model.speaker_encoder_name)
        kwargs.setdefault('speaker_embedding_dim', pretrained_model.speaker_embedding_dim)
        
        instance = cls(nom = nom, max_to_keep = 1, pretrained_name = pretrained_name, ** kwargs)

        partial_transfer_learning(instance.tts_model, pretrained_model.tts_model)
        
        instance.save()
        
        return instance
    
