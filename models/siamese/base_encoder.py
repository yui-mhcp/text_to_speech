
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
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from datasets import prepare_dataset
from models.interfaces import BaseModel
from utils.distance import distance, KNN
from utils.embeddings import load_embedding, save_embeddings, embeddings_to_np
from utils import plot_embedding, pad_batch, sample_df, convert_to_str

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

DEPRECATED_CONFIG   = ('threshold', 'embed_distance')

def l2_normalize(x):
    import tensorflow as tf
    return tf.math.l2_normalize(x, axis = -1)

class BaseEncoderModel(BaseModel):
    """
        Base class for Encoder Network architecture 
        
        The concept of Encoder Network is to have a unique model that "encodes" inputs in an embedding space (named the `embedding`), and compares them based on a distance (or similarity) metric function.
        
        You must define functions : 
            - build_encoder(** kwargs)  : return a `tf.keras.Sequential` model which will be the encoder of the siamese network
            - get_input(data)       : load a single input data (for the dataset pipeline)
            - augment_input(data)   : (optional) augment a single input data
            - preprocess_input(inputs)  : (optional) apply a preprocessing on a batch of inputs
        
        The evaluation of such models is mainly based on clustering evaluation : all inputs are embedded in the embedding space, and then clustered to group the similar embeddings into clusters. These clusters can then be evaluated according to the expected clusters (labels). 
    """
    def __init__(self, distance_metric = 'cosine', ** kwargs):
        """
            Constructor for the base encoder configuration
            
            Arguments :
                - distance_metric   : the distance (or similarity) metric to use for the comparison of embeddings
        """
        for k in DEPRECATED_CONFIG: kwargs.pop(k, None)
        
        self.__friends  = None
        
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
        """ Returns the ID or label of `data` """
        if isinstance(data, (dict, pd.Series)):
            return data['label'] if 'label' in data else data['id']
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            return data[1]
        
        raise ValueError('Unhandled data (type {}) : {}'.format(type(data), data))

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
    def embedding_file(self):
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
        return None
    
    @property
    def encoder_input_shape(self):
        return self.encoder.input_shape
    
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
        des += "- Embedding dim   : {}\n".format(self.embedding_dim)
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
    
    def distance(self, embedded_1, embedded_2, ** kwargs):
        """ Return distance between embeddings (based on `self.distance_metric`) """
        return distance(embedded_1, embedded_2, method = self.distance_metric, ** kwargs)

    def encode_data(self, data):
        """ Calls `self.get_input(data)` and `self.get_output(data)` """
        return self.get_input(data), self.get_output(data)
    
    def filter_data(self, inputs, output):
        """ Calls `self.filter_input(inputs) `"""
        return self.filter_input(inputs)
    
    def augment_data(self, inputs, output):
        """ Calls `self.augment_input(inputs)` """
        return self.augment_input(inputs), output
    
    def preprocess_data(self, inputs, output):
        """ Calls `self.preprocess_input(inputs)` """
        return self.preprocess_input(inputs), output
        
    def get_dataset_config(self, ** kwargs):
        kwargs['batch_before_map']  = True
        
        return super().get_dataset_config(** kwargs)
        
    def _get_train_config(self, * args, test_size = 1, test_batch_size = 1., ** kwargs):
        """ Set new default test_batch_size to embed 128 data (32 same + 32 not-same pairs)  """
        return super()._get_train_config(
            * args, test_size = test_size, test_batch_size = test_batch_size, ** kwargs
        )
        

    def predict_with_target(self, batch, step, prefix, directory = None, ** kwargs):
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
    def embed(self, data, batch_size = 128, pad_value = 0., tqdm = lambda x: x, ** kwargs):
        """
            Embed a list of data
            
            Arguments :
                - data  : the data to embed, any type supported by `self.get_input`
                - batch_size    : the number of data to encode in parallel
                - pad_value     : the padding value to use (if padding is required)
                - tqdm      : progress bar
                - kwargs    : ignored
            Return :
                - embeddings    : `tf.Tensor` of shape `(len(data), self.embedding_dim)`
            
            Pipeline : 
                1) Calls `self.get_input(data)` to have encoded data
                2) Take a batch of `batch_size` inputs
                3) Calls `pad_batch(batch, pad_value)` to get a valid batch (if necessary)
                4) Calls `self.preprocess_input(batch)` to apply a preprocessing (if any)
                5) Pass the processed batch to `self.encoder`
                6) Concat all produced embeddings to return [len(data), self.embedding_dim] matrix
            
            This function is the core of the `encoder networks` as embeddings are used for everything (predict similarity / distance), label predictions, clustering, make funny colored plots, ...
            
            Note : if batching is used (i.e. the data is sequential of variable lengths), make sure that your encoder supports masking correctly, otherwise embeddings may differ between `batch_size = 1` and `batch_size > 1` due to padding
            
            Simple code to test the batching support :
            ```
            from utils import is_equal
            
            emb1 = model.embed(data, batch_size = 1)
            emb2 = model.embed(data, batch_size = 16)
            
            print(is_equal(emb1, emb2)[1])
            ```
            If the data is not of variable shape (e.g. images), batching is well supported by default
        """
        if tqdm is None: tqdm = lambda x: x
        
        time_logger.start_timer('encoding data')
        if not isinstance(data, (list, tuple, pd.DataFrame)): data = [data]
        
        inputs = self.get_input(data, ** kwargs)

        time_logger.stop_timer('encoding data')
        
        encoder = self.encoder
        
        embedded = []
        for idx in tqdm(range(0, len(inputs), batch_size)):
            time_logger.start_timer('processing')

            batch = inputs[idx : idx + batch_size]
            if not isinstance(batch[0], (list, tuple)):
                batch = pad_batch(batch, pad_value = pad_value)
            else:
                batch = [
                    pad_batch(b, pad_value = v) for b, v in zip(zip(* batch), pad_value)
                ]
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
            Calls self.embed(data) and plots the result
            Arguments :
                - data  : the data to embed
                    If pd.DataFrame : define `ids` as the 'id' column values
                    If dict         : keys are used for ids and values are `data`
                - ids   : the ids to use for plot
                - batch_size    : batch_size for embedding
                - kwargs        : plot kwargs
            
            Nearly equivalent to `plot_embedding(model.embed(data), ids)`
        """
        if isinstance(data, pd.DataFrame):
            col_id  = ids if isinstance(ids, str) else 'id'
            ids     = data[col_id].values if col_id in data.columns else None
        elif isinstance(data, dict):
            data, ids = list(data.keys()), list(data.values())
        
        embedded = self.embed(data, batch_size = batch_size)
        
        time_logger.start_timer('showing')
        plot_embedding(embedded, ids = ids, ** kwargs)
        time_logger.stop_timer('showing')

    @timer
    def predict_distance(self, x, y,
                         
                         as_matrix  = False,
                         decoder    = None,
                         embed_similarity = False,
                         ** kwargs
                        ):
        """
            Return a score of distance between x and y
            
            Arguments :
                - x : embedding vector(s) or a valid value for `self.embed`
                - y : embedding vector(s) or a valid value for `self.embed`
                - as_matrix : whether to return a matrix of distances between each pair or a 1-1 distance (if `True`, prefer using `self.predict_distance_matrix`)
                - decoder   : callable that takes a list [x, y] as argument and returns a score
                - embed_similarity : whether the decoder returns a similarity or distance score
            Return :
                - distance : `tf.Tensor` of distance scores
                    If `as_matrix = False`:
                        - a 1-D vector representing the distance between `x[i]` and `y[i]`
                    else:
                        - a 2-D matrix representing the distance between `x[i]` and `y[j]`
            
            **Important Note** : even if `self.distance_metric` is a similarity score, the result is a *distance score*, which differs from `self.distance` that simply returns the distance metric value
            
            Note :
            - a *distance score* means that the higher the score is, the more the data is different
            - a *similarity score* means that the higher the score is, the more the data is similar
        """
        if decoder is None: decoder = self.decoder
        
        if not isinstance(x, (np.ndarray, tf.Tensor)) or x.shape[-1] != self.embedding_dim:
            x = self.embed(x)
        if not isinstance(y, (np.ndarray, tf.Tensor)) or y.shape[-1] != self.embedding_dim:
            y = self.embed(y)
        
        if len(x.shape) == 1: x = tf.expand_dims(x, axis = 0)
        if len(y.shape) == 1: y = tf.expand_dims(y, axis = 0)
        
        if decoder is None:
            scores = self.distance(x, y, as_matrix = as_matrix, force_distance = True, ** kwargs)
        else:
            if x.shape[0] == 1 and y.shape[0] > 1:
                x = tf.tile(x, [y.shape[0], 1])
            
            scores = decoder([x, y], as_matrix = as_matrix)
            if embed_similarity: scores = 1. - scores
        
        return scores
    
    def predict_similarity(self, x, y, * args, ** kwargs):
        """ returns `1. - self.predict_distance(...)` """
        return 1. - self.predict_distance(x, y, * args, ** kwargs)
    
    def predict_distance_matrix(self, x, y = None, ** kwargs):
        if y is None: y = x
        return self.predict_distance(x, y, as_matrix = True, ** kwargs)
    
    def predict_similarity_matrix(self, x, y = None, ** kwargs):
        if y is None: y = x
        return self.predict_similarity(x, y, as_matrix = True, ** kwargs)

    def predict(self,
                datas,
                batch_size  = 128,
                main_key    = 'filename',
                embedding_key   = 'embedding',

                cache_size  = -1,

                save    = True,
                directory   = None,
                overwrite   = False,
                save_every  = 10000,
                embedding_name  = 'embeddings',

                tqdm    = lambda x, ** kwargs: x,
                verbose = False,

                ** kwargs
               ):
        """
            Embeds then save a list of data
            
            Arguments :
                - datas : the data to embed
                    - pd.DataFrame  : must contain `main_key`
                    - other type    : must be convertible to a `pd.DataFrame` containing `main_key`
                - batch_size    : the number of data to embed in parallel
                - main_key      : the unique key used to identify a data
                - embedding_key : the column name for the embeddings
                
                - cache_size    : the number of data to pre-load using the `tf.data.Dataset` API (this allows multi-threaded data loading).
                
                - save  : whether to save result or not
                - directory : where to save the result
                - overwrite : whether to overwrite already predicted data (identified by `main_key`)
                - save_every    : performs a checkpoint every `save_every` data embedded
                - embedding_name    : the embedding filename
                
                - tqdm  : progress bar
                - verbose   : whether to log each iteration or not
                
                - kwargs    : forwarded to `load_embedding`
            Return :
                - embeddings    : `pd.DataFrame` with an `embedding` key
            
            Note : if the saving feature is not required, it is recommanded to use `self.embed` to avoid overhead / complexity. The input type is also less restrictive in the `self.embed` method.
        """
        if directory is None: directory = self.pred_dir
        if not isinstance(datas, pd.DataFrame): datas = pd.DataFrame(datas)

        round_tqdm  = tqdm
        
        cache_size  = max(cache_size, batch_size)

        embeddings = load_embedding(
            directory       = directory,
            embedding_name  = embedding_name,
            aggregate_on    = None,
            ** kwargs
        )
        
        processed, to_process = [], datas
        if embeddings is not None:
            if 'filename' in main_key:
                datas[main_key]      = datas[main_key].apply(lambda f: f.replace(os.path.sep, '/'))
                embeddings[main_key] = embeddings[main_key].apply(lambda f: f.replace(os.path.sep, '/'))

            if not overwrite:
                mask        = datas[main_key].isin(embeddings[main_key])
                to_process  = datas[~mask]
                processed   = embeddings[embeddings[main_key].isin(datas[main_key])]
            else:
                embeddings  = embeddings[~embeddings[main_key].isin(datas[main_key])]

        logger.info("Processing datas...\n  {} items already processed\n  {} items to process".format(len(datas) - len(to_process), len(to_process)))

        if len(to_process) == 0: return processed

        if isinstance(processed, pd.DataFrame): processed = processed.to_dict('records')
        
        data_loader     = None
        if cache_size > batch_size and len(to_process) > batch_size:
            data_loader = iter(prepare_dataset(
                datas, map_fn = self.get_input, cache = False, batch_size = 0
            ))

        if verbose or data_loader is not None: round_tqdm  = lambda x: x
        else: tqdm = lambda x: x

        to_process = to_process.to_dict('records')

        n, n_round = 0 if embeddings is None else len(embeddings), len(to_process) // cache_size + 1
        for i in round_tqdm(range(n_round)):
            if verbose: logger.info("Round {} / {}".format(i + 1, n_round))

            batch = to_process[i * cache_size : (i + 1) * cache_size]
            if len(batch) == 0: continue

            inputs = batch
            if data_loader is not None:
                inputs = [next(data_loader) for _ in tqdm(range(len(batch)))]

            embedded = self.embed(inputs, batch_size = batch_size, tqdm = tqdm, ** kwargs)
            if hasattr(embedded, 'numpy'): embedded = embedded.numpy()

            embedded = [
                {main_key : infos[main_key], 'id' : infos['id'], embedding_key : emb} 
                for infos, emb in zip(batch, embedded)
            ]
            processed.extend(embedded)
            embedded    = pd.DataFrame(embedded)
            embeddings  = pd.concat([embeddings, embedded], ignore_index = True) if embeddings is not None else embedded

            if save and len(embeddings) - n >= save_every:
                n = len(embeddings)
                if verbose: logger.info("Saving at utterance {}".format(len(embeddings)))
                save_embeddings(
                    directory, embeddings, embedding_name, remove_file_prefix = True
                )

        if save and n < len(embeddings):
            save_embeddings(
                directory, embeddings, embedding_name, remove_file_prefix = True
            )

        return pd.DataFrame(processed)

    def embed_dataset(self, directory, dataset, embedding_name = None, ** kwargs):
        if not directory.endswith('embeddings'): directory = os.path.join(directory, 'embeddings')
        
        return self.predict(
            dataset,
            save    = True,
            directory   = directory,
            embedding_name  = embedding_name if embedding_name else self.nom,
            ** kwargs
        )
    
    @timer
    def recognize(self, queries = None, samples = None, ids = None, embedded = None, ** kwargs):
        """
            Predict labels for each `queries` based on the `K-NN` classifier (cf `utils.distance.knn`)
            
            Arguments :
                - queries   : non-embedded data to classify (any value supported by `self.embed`)
                - embedded  : embedded queries
                
                - samples   : embedded samples to use as labelled points (any type supported by the `KNN` class). If `None`, `self.friends` is used
                - ids       : the ids for each labelled sample (may be string or integer)
                
                - kwargs    : forwarded to `self.embed` and `knn.predict
            Return :
                - labels    : the result of `knn.predict`, a 1-D `tf.Tensor` of labels
        """
        assert queries is not None or embedded is not None
        
        if samples is None:  samples = self.friends
        if embedded is None: embedded = self.embed(queries, ** kwargs)

        # Apply K-NN to find best id for each embedded audio
        knn = KNN(samples, ids, distance_metric = self.distance_metric, ** kwargs)

        return knn.predict(embedded, ** kwargs)

    @timer
    def evaluate(self, dataset, ids = None, mode = 'classification', ** kwargs):
        """
            Evaluate the model based on a specified evaluation method
            
            Arguments :
                - dataset   : the data to embed then evaluate on
                - ids       : the expected labels (`y_true`)
                - mode      : the evaluation methodology
                - kwargs    : forwarded to `self.embed` and `self.evaluate_{mode}`
            Return :
                - metrics   : the resulting metrics (may vary according to the evaluation methodology)
        """
        if isinstance(dataset, pd.DataFrame) and ids is None: ids = dataset['id'].values
        embedded_dataset = self.embed(dataset, ** kwargs)
        
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
                                sample_ids = None,
                                
                                sample_size = None,
                                
                                ** kwargs
                               ):
        from sklearn.metrics import confusion_matrix
        
        if samples is None:             samples = self.friends
        elif isinstance(samples, str):  samples = load_embedding(samples)
        
        assert isinstance(samples, (np.ndarray, tf.Tensor, pd.DataFrame)), "Unknown samples type : {}".format(type(samples))
        assert len(samples) > 0, "You must provide samples to classify new data !"
        
        if sample_size is not None:
            if not isinstance(samples, pd.DataFrame):
                samples = pd.DataFrame([
                    {'id' : sample_ids[i], 'embedding' : samples[i]}
                    for i in range(len(ssamples))
                ])
            samples = sample_df(samples, n = None, n_sample = sample_size)

        if isinstance(samples, pd.DataFrame):
            if sample_ids is None:
                sample_ids = samples['id'].values
            if 'embedding' not in samples.columns:
                samples['embedding'] = list(self.embed(samples, ** kwargs).numpy())

        pred = self.recognize(embedded = dataset, samples = samples, ids = sample_ids, ** kwargs)
        
        if isinstance(ids[0], str):  pred = convert_to_str(pred)
        elif hasattr(pred, 'numpy'): pred = pred.numpy()
        
        cm = confusion_matrix(ids, pred)
        
        return {
            'pred'  : pred,
            'true'  : ids,
            'confusion_matrix'  : cm,
            'metrics'   : {
                'accuracy'  : np.sum(np.diag(cm)) / len(ids),
                'balanced_accuracy' : np.mean(np.diag(cm) / np.sum(cm, axis = -1))
            }
        }
    
    def evaluate_similarity(self, * args, ** kwargs):
        raise NotImplementedError()
    
    def evaluate_clustering(self, * args, ** kwargs):
        raise NotImplementedError()

    def load_friends(self, embedding_file = None):
        """ Loads the embedding file containing the friends """
        if embedding_file is None: embedding_file = self.embedding_file
            
        if not os.path.exists(embedding_file):
            return pd.DataFrame([], columns = ['id', 'embedding'])
        return load_embeddings(embedding_file)
    
    def save_friends(self, embedding_file = None):
        if embedding_file is None: embedding_file = self.embedding_file
        save_embeddings(embedding_file, self.friends)
    
    def show_friends(self, n = 25, n_sample = 25, ** kwargs):
        friends = self.friends

        logger.info("Number of friends : {} ({} embeddings)".format(
            len(friends['id'].unique()), len(friends)
        ))
        
        samples = sample_df(friends, n = n, n_sample = n_sample)
        plot_embedding(samples, ** kwargs)
    
    def set_friends(self, embeddings):
        """ Overwrites `self.friends` by `embeddings` """
        assert 'id' in embeddings.columns and 'embedding' in embeddings.columns
        
        self.__friends = embeddings
        self.save_friends()
    
    def add_friend(self, name, embeddings = None, ** kwargs):
        """
            Add new friend(s) to the model
            A 'friend' is a label the model knows and can identify ! :D
            It means that, if you add new friends, you can call the "identify" method, and the model will tell you which of its friends your data is
            
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
        
        assert isinstance(embeddings, pd.DataFrame), "Wrong type for name : {}\n{}".format(type(embeddings), embeddings)
        assert 'id' in embeddings.columns, "The embedding dataframe must have an 'id' column\n{}".format(embeddings)
        assert 'embedding' in embeddings.columns, "The embedding dataframe must have an 'embedding' column\n{}".format(embeddings)
        
        friends = self.friends
        
        friends_embeddings      = embeddings_to_np(friends)
        new_friends_embeddings  = embeddings_to_np(embeddings)
        
        already_know = np.any(np.all(
            np.expand_dims(a, 0) == np.expand_dims(b, 1), axis = -1
        ), axis = -1)
        
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

    def get_config(self, * args, ** kwargs):
        """ Return base configuration for the `encoder network` """
        config = super().get_config(* args, ** kwargs)
        config.update({
            'distance_metric'   : self.distance_metric
        })
        
        return config
