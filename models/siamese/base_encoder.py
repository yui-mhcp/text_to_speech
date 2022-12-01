
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
import time
import random
import logging
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from models.interfaces import BaseModel
from utils.distance import distance, KNN
from utils.thread_utils import Pipeline
from utils.embeddings import _embedding_filename, _default_embedding_ext, load_embedding, save_embeddings, embeddings_to_np
from utils import normalize_filename, plot_embedding, pad_batch, sample_df

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

def l2_normalize(x):
    import tensorflow as tf
    return tf.math.l2_normalize(x, axis = -1)

class BaseEncoderModel(BaseModel):
    """
        Base class for Siamese Network architecture 
        
        The concept of Siamese Network is to have a unique model that "encodes" 2 inputs in an embedding space, compare them according to a distance function and based on the distance, decides a score of similarity (or disimilarity).
        
        The output of a Siamese is the output of a Dense with 1 neuron and a sigmoid function. 
        - It means that if embed_distance == True, an output of 0.75 can be interpreted as "75% that the 2 inputs are different"
        - Otherwise if embed_distance == False, an output of 0.75 can beinterpreted as "75% that the 2 inputs are similar (of the same class)"
        
        You must define functions : 
            - build_encoder(** kwargs)  : return a `tf.keras.Sequential` model which will be the encoder of the siamese network
            - get_input(data)       : load a single input data (for the dataset pipeline)
            - augment_input(data)   : augment a single input data
            - preprocess_input(inputs)  : apply a preprocessing on a batch of inputs
        
        The dataset pipeline is as follow : 
            1) encode_data receives a 2-tuple (same, not_same)
                Where both are dict with suffixes _x and _y for data belonging to the 1st / 2nd input
                The get_input(data) receives all key-value pairs with _x and _y suffixes (one at a time) and remove the suffixe so that you can treat it the same way
                `sare` inputs where _x and _y belong to the same class
                `not_same` are inputs where _x and _y belong to different classes
            2) augment_data(same, not_same) : receives input x / y (non-batched) once at a time so that you can augment them in independant way
                Each (from same and not_same) will be passed to `augment_input`
            
            3) preprocess_data(same, not_same)  : receives batched datas for same and not same. They are then concatenated to form a single batch of same and not_same
                Note that if the 1st dimension mismatch (variable length data), they are padded to match the longest one
    """
    def __init__(self,
                 distance_metric    = 'cosine',
                 embed_distance = True,
                 threshold  = 0.5,
                 ** kwargs
                ):
        self.__friends  = None
        
        self.threshold      = threshold
        self.embed_distance     = embed_distance
        self.distance_metric    = distance_metric
        
        super(BaseEncoderModel, self).__init__(** kwargs)
    
    def init_train_config(self, ** kwargs):
        super().init_train_config(** kwargs)
        
        if hasattr(self.get_loss(), 'variables') and hasattr(self.get_metric(), 'set_variables'):
            self.get_metric().set_variables(self.get_loss().variables)

    def build_encoder(self, normalize = True, ** kwargs):
        """ Creates the `tf.keras.Model` that encodes the input on an embedding vector """
        raise NotImplementedError("You must define the `build_encoder` method !")
    
    def get_input(self, data):
        """ Process `data` to return a single input for the encoder """
        raise NotImplementedError("You must define the `get_input(data)` method !")

    def get_output(self, data):
        if isinstance(data, (dict, pd.Series)):
            return data['label'] if 'label' in data else data['id']
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            return data[1]
        raise ValueError('Unknown data : {}'.format(data))

    def filter_input(self, inp):
        """ Filter a single processed input (from get_input()) """
        return True
        
    def augment_input(self, inp):
        """ Augment a single processed input (from get_input()) """
        return inp
    
    def preprocess_input(self, inputs):
        """ Preprocess a batch of inputs (if you need to do processing on the whole batch) """
        return inputs
    
    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.embeddings_dir, exist_ok = True)
        
    def _build_model(self, normalize = True, ** kwargs):
        """ Builds the main model with the `SimilarityLayer` """
        encoder = self.build_encoder(normalize = normalize, ** kwargs)
        
        if normalize and isinstance(encoder, tf.keras.Sequential):
            encoder.add(tf.keras.layers.Lambda(l2_normalize, name = 'l2_normalization'))
        
        super(BaseEncoderModel, self)._build_model(model = encoder)
    
    @property
    def embeddings_dir(self):
        """ Special directory for saving `friends` embeddings """
        return os.path.join(self.folder, 'embeddings')
    
    @property
    def embeddings_file(self):
        return os.path.join(self.embeddings_dir, 'embeddings_{}'.format(self.embedding_dim))
    
    @property
    def friends(self):
        if self.__friends is None:
            self.__friends = self.load_friends()
        return self.__friends

    @property
    def encoder(self):
        return self.model
    
    @property
    def decoder(self):
        raise NotImplementedError()
    
    @property
    def encoder_input_signature(self):
        return tf.TensorSpec(shape = self.encoder_input_shape, dtype = tf.float32)
    
    @property
    def embedding_dim(self):
        return self.encoder.output_shape[-1]
    
    @property
    def input_signature(self):
        return self.encoder_input_signature
    
    @property
    def output_signature(self):
        return tf.TensorSpec(shape = (None, ), dtype = tf.int32)
    
    def __str__(self):
        des = super().__str__()
        des += "- Embedding dim : {}\n".format(self.embedding_dim)
        des += "- Distance metric : {}\n".format(self.distance_metric)
        return des

    def compile(self,
                loss    = 'ge2e_loss', loss_config      = {'mode' : 'softmax'},
                metrics = ['ge2e_metric'], metrics_config = {},
                ** kwargs
               ):
        loss_config.setdefault('distance_metric', self.distance_metric)
        metrics_config.setdefault('distance_metric', self.distance_metric)
        metrics_config.setdefault('mode', loss_config.get('mode', 'softmax'))
        super().compile(
            loss = loss, loss_config = loss_config,
            metrics = metrics, metrics_config = metrics_config, ** kwargs
        )
    
    def decode_output(self, output):
        """
            Return whether the 2 inputs can be considered as the same class based on the output score 
            
            This function returns whether the inputs can be considered as same independently of `self.embed_distance`
        """
        return output < self.threshold if self.embed_distance else output > self.threshold
    
    def distance(self, embedded_1, embedded_2, ** kwargs):
        """ Return distance between embeddings (based on self.distance_metric) """
        return distance(embedded_1, embedded_2, method = self.distance_metric, ** kwargs)

    def encode_data(self, data):
        """ Apply `self.encode()` on same and not_same separately """
        return self.get_input(data), self.get_output(data)
    
    def filter_data(self, inputs, output):
        return self.filter_input(inputs)
    
    def augment_data(self, inputs, output):
        return self.augment_input(inputs), output
    
    def preprocess_data(self, inputs, output):
        return self.preprocess_input(inputs), output
        
    def get_dataset_config(self, **kwargs):
        """ Add default configuration for siamese dataset """
        kwargs['batch_before_map']  = True
        
        return super().get_dataset_config(**kwargs)
        
    def _get_train_config(self, * args, test_size = 1, test_batch_size = 1., ** kwargs):
        """ Set new default test_batch_size to embed 128 data (32 same + 32 not-same pairs)  """
        return super()._get_train_config(
            * args, test_size = test_size, test_batch_size = test_batch_size, ** kwargs
        )
        

    def predict_with_target(self, batch, step, prefix, directory = None, **kwargs):
        """ Embeds the batch (inputs, ids) and plot the results with the given ids """
        if directory is None: directory = self.train_test_dir
        else: os.makedirs(directory, exist_ok = True)
        kwargs.setdefault('show', False)
        
        inputs, ids = batch
        
        embeddings  = self.encoder(inputs, training = False)
        
        title       = 'Embedding space (step {})'.format(step)
        filename    = os.path.join(directory, prefix + '.png')
        plot_embedding(
            embeddings, ids = ids, filename = filename, title = title, ** kwargs
        )
    
    @timer
    def embed(self, data, batch_size = 128, tqdm = lambda x: x, ** kwargs):
        """
            Embed a list of data
            
            Pipeline : 
                1) Call self.get_input(data) to have encoded data
                2) Take a batch of `batch_size` inputs
                3) Call pad_batch(batch) to have pad it (if necessary)
                4) Call self.preprocess_input(batch) to apply a preprocessing (if needed)
                5) Pass the processed batch to self.encoder
                6) Concat all produced embeddings to return [len(data), self.embedding_dim] matrix
            
            This function is the core of the `siamese networks` as embeddings are used for everything (predict similarity / distance), label predictions, clustering, make funny colored plots, ...
        """
        if tqdm is None: tqdm = lambda x: x
        
        time_logger.start_timer('processing')
        if not isinstance(data, (list, tuple, pd.DataFrame)): data = [data]
        
        inputs = self.get_input(data, ** kwargs)

        time_logger.stop_timer('processing')
        
        encoder = self.encoder
        
        embedded = []
        for idx in tqdm(range(0, len(inputs), batch_size)):
            time_logger.start_timer('processing')

            batch = inputs[idx : idx + batch_size]
            batch = pad_batch(batch) if not isinstance(batch[0], (list, tuple)) else [pad_batch(b) for b in zip(* batch)]
            batch = self.preprocess_input(batch)
            
            time_logger.stop_timer('processing')
            time_logger.start_timer('encoding')

            embedded_batch = encoder(batch, training = False)
            
            time_logger.stop_timer('encoding')

            embedded.append(embedded_batch)

        return tf.concat(embedded, axis = 0)
    
    @timer
    def plot_embedding(self, data, ids = None, batch_size = 128, ** kwargs):
        """
            Call self.embed() on `data` and plot the result
            Arguments :
                - data  : the data to embed
                    If pd.DataFrame : define `ids` as the 'id' column values
                    If dict         : keys are used for ids and values are `data`
                - ids   : the ids to use for plot
                - batch_size    : batch_size for embedding
                - kwargs        : plot kwargs
        """
        if isinstance(data, pd.DataFrame) and ids is None:
            col_id = ids if isinstance(ids, str) else 'id'
            ids = data[col_id].values if col_id in data.columns else None
        elif isinstance(data, dict):
            data, ids = list(data.keys()), list(data.values())
        
        embedded = self.embed(data, batch_size = batch_size)
        
        time_logger.start_timer('showing')
        plot_embedding(embedded, ids = ids, ** kwargs)
        time_logger.stop_timer('showing')

    def embed_dataset(self, directory, dataset, embedding_name = None, ** kwargs):
        """
            Calls `self.predict` and save the result to `{directory}/embeddings/{embedding_name}` (`embedding_name = self.nom` by default)
        """
        if not directory.endswith('embeddings'): directory = os.path.join(directory, 'embeddings')
        
        return self.predict(
            dataset,
            save    = True,
            directory   = directory,
            embedding_name  = embedding_name if embedding_name else self.nom,
            ** kwargs
        )
    
    def get_pipeline(self,
                     id_key = 'filename',
                     batch_size = 1,
                     
                     save   = True,
                     directory  = None,
                     embedding_name = _embedding_filename,
                     ** kwargs
                    ):
        @timer
        def preprocess(row, ** kw):
            inputs = self.get_input(row)
            if not isinstance(row, (dict, pd.Series)): row = {}
            row['processed'] = inputs
            return row
        
        @timer
        def inference(inputs, ** kw):
            batch_inputs = inputs if isinstance(inputs, list) else [inputs]
            
            batch = [inp.pop('processed') for inp in batch_inputs]
            batch = pad_batch(batch) if not isinstance(batch[0], (list, tuple)) else [
                pad_batch(b) for b in zip(* batch)
            ]
            batch = self.preprocess_input(batch)
            
            embeddings = encoder(batch, training = False)

            for row, embedding in zip(batch_inputs, embeddings): row['embedding'] = embedding
            
            return inputs
        
        if save:
            embedding_file = embedding_name
            if '{}' in embedding_file: embedding_file = embedding_file.format(self.embedding_dim)
            if not os.path.splitext(embedding_file)[1]: embedding_file += _default_embedding_ext
            if directory is None: directory = self.pred_dir
            filename = os.path.join(directory, embedding_file)
        
        encoder = self.encoder

        pipeline = Pipeline(** {
            ** kwargs,
            'filename'  : None if not save else filename,
            'id_key'    : id_key,
            'save_keys' : [id_key, 'embedding'],
            'as_list'   : True,
            
            'tasks'     : [
                preprocess,
                {'consumer' : inference, 'batch_size' : batch_size, 'allow_multithread' : False}
            ]
        })
        pipeline.start()
        return pipeline
    
    @timer
    def predict(self, data, ** kwargs):
        pipeline = self.get_pipeline(** kwargs)
        
        return pipeline.extend_and_wait(data, ** kwargs)
    
    def get_config(self, * args, ** kwargs):
        """ Return base configuration for a `siamese network` """
        config = super().get_config(*args, **kwargs)
        config['threshold']     = self.threshold
        config['embed_distance']    = self.embed_distance
        config['distance_metric']   = self.distance_metric
        
        return config