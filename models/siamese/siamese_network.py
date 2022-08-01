
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
from utils.embeddings import load_embedding, save_embeddings, embed_dataset, embeddings_to_np
from utils import normalize_filename, plot_embedding, pad_batch, sample_df

time_logger = logging.getLogger('timer')

def l2_normalize(x):
    import tensorflow as tf
    return tf.math.l2_normalize(x, axis = -1)

class SiameseNetwork(BaseModel):
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
                 distance_metric    = 'euclidian',
                 embed_distance     = True,
                 threshold          = 0.5,
                 ** kwargs
                ):
        """
            Constructor for a base siamese network

            Arguments :
                - distance_metric   : the distance function to compute between the 2 embeddings
                - embed_distance    : whether to embed distance or similarity
                    If True : the bigger the output, the bigger the distance
                    If False : the bigger the output, the bigger the similarity
                - threshold     : thereshold to decide de decision boundary (currently only 0.5 is correctly handled in loss / metrics)
        """
        self.__friends  = None
        
        self.threshold      = threshold
        self.embed_distance     = embed_distance
        self.distance_metric    = distance_metric
        
        super(SiameseNetwork, self).__init__(**kwargs)
    
    def build_encoder(self, normalize = True, ** kwargs):
        """
            Return a `tf.keras.Sequential` model which is the encoder of the siamese network 
        """
        raise NotImplementedError("You must define the `build_encoder` method !")
    
    def get_input(self, data):
        """
            Process `data` to return a single output for the encoder
            
            `data` is basically a dict or a pd.Series but can take every type of value you want
        """
        raise NotImplementedError("You must define the `get_input(data)` method !")

    def filter_input(self, inp):
        """ Filter a single processed input (from get_input()) """
        return True
        
    def augment_input(self, inp):
        """ Augment a single processed input (from get_input()) """
        return inp
    
    def preprocess_input(self, inputs):
        """
        Preprocess a batch of inputs (if you need to do processing on the whole batch)
        """
        return inputs
    
    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.embeddings_dir, exist_ok = True)
        
    def _build_model(self, normalize = True, ** kwargs):
        """ Build the `siamese` architecture with self.build_encoder() as encoder part """
        encoder = self.build_encoder(normalize = normalize, ** kwargs)
        
        if isinstance(encoder, tf.keras.Sequential):
            if normalize:
                encoder.add(tf.keras.layers.Lambda(
                    l2_normalize, name = 'normalization_layer'
                ))
            input_kwargs = {'input_shape' : encoder.input_shape[1:]}
        else:
            if normalize:
                logging.warning("Encoder is not a `tf.keras.Sequential` so you have to handle `normalize` internally !")
            input_kwargs = {'input_signature' : self.encoder_input_signature}
        
        siamese_config = {
            'architecture_name' : 'siamese',
            'model'             : encoder,
            'distance_metric'   : self.distance_metric,
            ** input_kwargs
        }
                
        super()._build_model(siamese = siamese_config)
    
    @property
    def embeddings_dir(self):
        """ Special directory for saving `friends` embeddings """
        return os.path.join(self.folder, 'embeddings')
    
    @property
    def embeddings_file(self):
        return os.path.join(
            self.embeddings_dir, 'embeddings_{}.csv'.format(self.embedding_dim)
        )
    
    @property
    def encoder_input_signature(self):
        return tf.TensorSpec(shape = self.encoder_input_shape, dtype = tf.float32)
    
    @property
    def embedding_dim(self):
        return self.encoder.output_shape[-1]
    
    @property
    def input_signature(self):
        return (self.encoder_input_signature, self.encoder_input_signature)
    
    @property
    def output_signature(self):
        return tf.TensorSpec(shape = (None, 1), dtype = tf.int32)
    
    @property
    def encoder(self):
        signature = self.encoder_input_signature
        n = 1 if not isinstance(signature, (list, tuple)) else len(signature)
        return self.siamese.layers[n * 2]
    
    @property
    def decoder(self):
        signature = self.encoder_input_signature
        n = 1 if not isinstance(signature, (list, tuple)) else len(signature)
        
        inputs = [
            tf.keras.layers.Input(shape = (None, self.embedding_dim)),
            tf.keras.layers.Input(shape = (None, self.embedding_dim))
        ]
        out = inputs
        for l in self.siamese.layers[n * 2 + 1:]:
            out = l(out)
        return tf.keras.Model(inputs, out, name = 'siamese_decoder')
    
    @property
    def friends(self):
        if self.__friends is None:
            self.__friends = self.load_friends()
        return self.__friends
    
    def __str__(self):
        des = super().__str__()
        des += "Embedding dim : {}\n".format(self.embedding_dim)
        des += "Distance metric : {}\n".format(self.distance_metric)
        return des
                
    def compile(self,
                loss        = 'binary_crossentropy',
                metrics     = ['binary_accuracy', 'eer'],
                ** kwargs
               ):
        super().compile(loss = loss, metrics = metrics, ** kwargs)
        
    def decode_output(self, output):
        """
            Return whether the 2 inputs can be considered as the same class based on the output score 
            
            This function returns whether the inputs can be considered as same independently of `self.embed_distance`
        """
        return output < self.threshold if self.embed_distance else output > self.threshold
    
    def distance(self, embedded_1, embedded_2):
        """ Return distance between embeddings (based on self.distance_metric) """
        return distance(embedded_1, embedded_2, method = self.distance_metric)
        
    def encode(self, data):
        """
            Call self.get_input() on normalized version of data (by removing the _x / _y suffix)
            This function process separately `same` and `not_same`
        """
        inp_x = self.get_input({k[:-2] : v for k, v in data.items() if '_x' in k})
        inp_y = self.get_input({k[:-2] : v for k, v in data.items() if '_y' in k})
        
        if 'same' in data: same = int(data['same'])
        elif 'id' in data: same = 1
        else: same = int(data['id_x'] == data['id_y'])
        
        return (inp_x, inp_y), [same]
    
    def filter(self, data):
        (inp_x, inp_y), target = data
        return tf.logical_and(self.filter_input(inp_x), self.filter_input(inp_y))
    
    def augment(self, data):
        """ Augment `same` or `not_same` separately """
        (inp_x, inp_y), target = data
        
        inp_x = self.augment_input(inp_x)
        inp_y = self.augment_input(inp_y)
        
        return (inp_x, inp_y), target
    
    def concat(self, x_same, x_not_same):
        """
            Concat both batched `x_same` and `x_not_same` together (same function called for the y) 
        """
        return tf.concat([x_same, x_not_same], axis = 0)

    def encode_data(self, same, not_same):
        """ Apply `self.encode()` on same and not_same separately """
        return self.encode(same), self.encode(not_same)
    
    def filter_data(self, same, not_same):
        return tf.logical_and(self.filter(same), self.filter(not_same))
    
    def augment_data(self, same, not_same):
        """ Apply `self.augment()` on same and not_same separately """
        return self.augment(same), self.augment(not_same)
    
    def preprocess_data(self, same, not_same):
        """ 
            oncat `x` and `y` from same and not_same and call `self.preprocess_input` on the batched result
            
            In theory it should also pad inputs but it is not working (quite strange...)
            As a solution, I created the `self.concat(x, y)` method so that you can pad as you want (can be useful if you have special padding_values)
        """
        (same_x, same_y), target_same = same
        (not_same_x, not_same_y), target_not_same = not_same
        
        inp_x = self.preprocess_input(self.concat(same_x, not_same_x))
        inp_y = self.preprocess_input(self.concat(same_y, not_same_y))
        
        target = tf.concat([target_same, target_not_same], axis = 0)
        if self.embed_distance: target = 1 - target
        
        return (inp_x, inp_y), target
        
    def get_dataset_config(self, **kwargs):
        """ Add default configuration for siamese dataset """
        kwargs['siamese'] = True
        kwargs['batch_before_map']  = True
        
        return super().get_dataset_config(**kwargs)
        
    def _get_train_config(self, * args, test_size = 1,
                          test_batch_size = 32, ** kwargs):
        """
        Set new default test_batch_size to embed 128 data (32 same + 32 not-same pairs) 
        """
        return super()._get_train_config(
            * args, test_size = test_size, test_batch_size = test_batch_size, ** kwargs
        )
        
    def predict_with_target(self, batch, step, prefix, directory = None, **kwargs):
        """
            Embed the x / y in batch and plot their embeddings 
            This function should be improved to add labels information but as ids are not in batch, I do not know how to add information of similarity in the plots...
        """
        if directory is None: directory = self.train_test_dir
        else: os.makedirs(directory, exist_ok = True)
        kwargs.setdefault('show', False)
        
        (inp_x, inp_y), _ = batch
        
        encoder = self.encoder
        embedded_x = encoder(inp_x, training = False)
        embedded_y = encoder(inp_y, training = False)
        embedded = tf.concat([embedded_x, embedded_y], axis = 0)
        
        title       = 'embedding space (step {})'.format(step)
        filename    = os.path.join(directory, prefix + '.png')
        plot_embedding(
            embedded, filename = filename, title = title, ** kwargs
        )
    
    @timer
    def evaluate(self,
                 dataset,
                 ids    = None, 
                 batch_size = 128,
                 
                 mode   = 'classification',
                 ** kwargs
                ):
        """ Evaluate the model on a specified evaluation method """
        if isinstance(dataset, pd.DataFrame) and ids is None:
            ids = dataset['id'].values
        embedded_dataset = self.embed(dataset, batch_size = batch_size)
        
        kwargs['batch_size'] = batch_size
        if mode == 'classification':
            return self.evaluate_classification(embedded_dataset, ids, ** kwargs)
        elif mode == 'similarity':
            return self.evaluate_similarity(embedded_dataset, ids, ** kwargs)
        elif mode == 'clustering':
            return self.evaluate_clustering(embedded_dataset, ids, ** kwargs)
        
    def evaluate_classification(self,
                                dataset,
                                ids,
                                
                                samples     = None,
                                samples_ids = None,
                                
                                batch_size  = 128,
                                sample_size = None, 
                                ** kwargs
                               ):
        """
            Evaluate the `label prediction` performance of the model
            
            This function essentially embed `samples` if necessary, sample `sample_size` for each label in `samples` and call `self.recognize` to have predicted labels
            Then it computes the `accuracy` to get the average number of well-predicted labels
        """
        if samples is None:
            samples = self.friends
        elif isinstance(samples, str):
            samples = load_embedding(samples)
        
        assert isinstance(samples, (np.ndarray, tf.Tensor, pd.DataFrame)), "Unknown samples type : {}".format(type(samples))
        assert len(samples) > 0, "You must provide samples to recognize new datas !"
        
        if not isinstance(samples, pd.DataFrame):
            samples = pd.DataFrame([
                {'id' : samples_ids[i], 'embedding' : samples[i]}
                for i in range(len(ssamples))
            ])
            
        if sample_size is not None:
            samples = sample_df(samples, n = None, n_sample = sample_size)

        if 'embedding' not in samples.columns:
            samples['embedding'] = list(self.embed(samples, batch_size = batch_size).numpy())

        pred = self.recognize(embedded = dataset, samples = samples, ** kwargs).numpy()
        
        return np.mean(pred == ids)
    
    def evaluate_similarity(self, * args, ** kwargs):
        raise NotImplementedError()
    
    def evaluate_clustering(self, * args, ** kwargs):
        raise NotImplementedError()
    
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

            embedded_batch = encoder(batch)
            
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

    @timer
    def pred_similarity(self, x, y, decoder = None):
        """
            Return a score of similarity between x and y
            Arguments : 
                - x : a single embedding vector
                - y : a single / matrix of embedding vector(s)
                - decoder   : decoder to use (basically `self.decoder` but used to not build it for every `x`)
        """
        if decoder is None: decoder = self.decoder
        
        if len(tf.shape(y)) == 1: y = tf.expand_dims(y, axis = 0)
        if len(tf.shape(x)) == 1: x = tf.expand_dims(x, axis = 0)
        if tf.shape(x)[0] != tf.shape(y)[0]:
            x = tf.tile(x, [tf.shape(y)[0], 1])
        
        scores = tf.reshape(decoder([x, y]), [-1])
        if self.embed_distance: scores = 1. - scores
        
        return scores
    
    def pred_distance(self, x, y, decoder = None):
        """
            Return a score of distance for all pairs
            The result is symetric matrix [n,n] where the element [i,j] is the disimilarity probability between i-th and j-th embeddings
        """
        return 1. - self.pred_similarity(x, y, decoder)
    
    def pred_similarity_matrix(self, embeddings):
        """
            Return a score of distance for all pairs
            The result is symetric matrix [n,n] where the element [i,j] is the similarity probability between i-th and j-th embeddings
        """
        decoder = self.decoder
        
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n), dtype = np.float32)
        
        decoder = self.decoder
        
        pred_pairs = []
        for i in range(1, len(embeddings)):
            scores = self.pred_similarity(embeddings[i], embeddings[:i], decoder)

            similarity_matrix[i, :i] = scores.numpy()
        
        similarity_matrix = similarity_matrix + similarity_matrix.T + np.eye(n)
        
        return similarity_matrix
    
    def pred_distance_matrix(self, embeddings):
        """
            Return a score of distance for all pairs
            The result is symetric matrix [n,n] where the element [i,j] is the disimilarity probability between i-th and j-th embeddings
        """
        return 1. - self.pred_similarity_matrix(embeddings)
    
    @timer
    def embed_dataset(self, directory, dataset, ** kwargs):
        """ Call the `embed_dataset` function with `self.embed` as embedding function """
        return embed_dataset(
            directory   = directory, 
            dataset     = dataset, 
            embed_fn    = self.embed, 
            embedding_dim   = self.embedding_dim, 
            ** kwargs
        )
            
    def load_friends(self):
        if not os.path.exists(self.embeddings_file):
            return pd.DataFrame([], columns = ['id', 'embedding'])
        return load_embeddings(self.embeddings_file)
    
    def save_friends(self):
        save_embeddings(self.friends_file, self.friends)
    
    def show_friends(self, n = 25, n_sample = 25, ** kwargs):
        friends = self.friends

        logging.info("Number of friends : {} ({} embeddings)".format(
            len(friends['id'].unique()), len(friends)
        ))
        
        to_plot = sample_df(friends, n = n, n_sample = n_sample)
        plot_embedding(to_plot, ** kwargs)
    
    def set_friends(self, embeddings):
        assert 'id' in embeddings.columns and 'embedding' in embeddings.columns
        
        self.__friends = embeddings
        self.save_friends()
    
    def add_friend(self, name, embeddings = None, ** kwargs):
        """
            Add new friend(s) to the model
            A 'friend' is a label the model knowh and can identify ! :D
            It means if you add new friends, you can call the "identify" method and the model will tell you which of its friends your data is
            
            Arguments :
                - name  : the name for this friend
                    Can be a dict of information (with at least 'id' key)
                    if pd.DataFrame, is used as name + embeddings
                - embeddings    : np.ndarray of embeddings
                    if name is a pd.DataFrame, embeddings is not used
        """
        if isinstance(name, str):
            if os.path.exists(name):
                name = load_embedding(name, self.embedding_dim)
            else:
                assert embeddings is not None, "You must provide embeddings !"
        
        if not isinstance(name, pd.DataFrame):
            embeddings = embeddings_to_np(embeddings)
            if embeddings.ndim == 1: embeddings = np.expand_dims(embeddings, axis = 0)
            
            if not isinstance(name, dict): name = {'id' : str(name)}
            name = pd.DataFrame([
                {** name, 'embedding' : e} for e in embeddings
            ])
        
        embeddings = name
        
        assert isinstance(embeddings, pd.DataFrame), "Wrong type for name : {}".format(type(embeddings))
        assert 'id' in embeddings.columns, "The embedding dataframe must have an 'id' column\n{}".format(embeddings)
        
        friends = self.friends
        
        friends_embeddings      = embeddings_to_np(friends)
        new_friends_embeddings  = embeddings_to_np(embeddings)
        
        already_know = np.array([
            (e in friends_embeddings).all(axis = 1).any() for e in new_friends_embeddings
        ])
        
        embeddings  = embeddings[~ already_know]
        
        self.__friends = pd.concat([friends, embeddings])
        
        self.save_friends()
        
    def remove_friend(self, name):
        friends = self.friends
        
        mask = friends['id'] == name
        
        dropped = friends[~mask]
        
        self.__friends = friends[mask]
        self.save_friends()
        
        return dropped
    
    def predict(self, * args, ** kwargs):
        return self.recognize(* args, ** kwargs)
    
    @timer
    def recognize(self,
                  datas     = None,
                  embedded  = None,
                  
                  samples   = None,
                  ids       = None, 
                  
                  batch_size = 128,
                  tqdm  = lambda x: x,
                  
                  ** kwargs
                 ):
        """
            Predict labels for each `datas` based on the `K-NN` decision rule
            
            Arguments :
                - datas     : non-embedded datas
                - embedded  : embedded data (if None, call self.embed() on `datas`)
                
                - samples   : embedded samples to use as labelled points
                - ids       : the ids for each labelled sample
                
                - batch_size    : batch_size for the self.embed() calls
                - tqdm      : progress bar for the `knn.predict()` call
                - kwargs    : kwargs passed to `knn.predict`
        """
        assert datas is not None or embedded is not None
        
        if samples is None: samples = self.friends
        
        if isinstance(samples, str):
            samples = load_embedding(samples)
        
        if embedded is None:
            embedded = self.embed(datas, batch_size = batch_size, ** kwargs)

        # Apply K-NN to find best id for each embedded audio
        knn = KNN(samples, ids)

        return knn.predict(embedded, ** kwargs)
    
    
    def get_config(self, *args, ** kwargs):
        """ Return base configuration for a `siamese network` """
        config = super().get_config(*args, **kwargs)
        config['threshold']     = self.threshold
        config['embed_distance']    = self.embed_distance
        config['distance_metric']   = self.distance_metric
        
        return config