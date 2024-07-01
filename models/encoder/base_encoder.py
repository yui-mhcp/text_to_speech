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
import keras
import logging
import numpy as np
import pandas as pd

from loggers import timer, time_logger
from utils.datasets import prepare_dataset
from utils.keras_utils import TensorSpec, ops
from models.interfaces.base_model import BaseModel
from utils import plot_embedding, pad_batch, distance, load_embedding, save_embeddings, path_to_unix
from custom_train_objects import GE2EGenerator, get_loss

logger = logging.getLogger(__name__)

DEPRECATED_CONFIG = ('threshold', 'embed_distance')

class BaseEncoderModel(BaseModel):
    """
        Base class for Encoder architecture 
        
        The concept of Encoder is to have a unique model that "encodes" the inputs in an embedding space (named the `embedding`), and compares them based on a distance (or similarity) metric.
        
        The evaluation of such models is mainly based on clustering evaluation : all inputs are embedded in the embedding space, and then clustered to group the similar embeddings into clusters. These clusters can be evaluated according to the expected (labels). 
    """
    
    _directories    = {
        ** BaseModel._directories, 'embeddings_dir' : '{root}/{self.name}/embeddings'
    }

    _default_loss   = 'GE2ELoss'
    _default_metrics    = ['GE2EMetric']
    
    def __init__(self, distance_metric = 'cosine', ** kwargs):
        """
            Constructor for the base encoder configuration
            
            Arguments :
                - distance_metric   : the distance (or similarity) metric to use for the comparison of embeddings
        """
        for k in DEPRECATED_CONFIG: kwargs.pop(k, None)
        
        self.distance_metric    = distance_metric
        
        super(BaseEncoderModel, self).__init__(** kwargs)
        
        self.__embeddings = None
    
    @property
    def embeddings(self):
        if self.__embeddings is None: self.__embeddings = self.load_embeddings()
        return self.__embeddings

    @property
    def encoder(self):
        return self.model
    
    @property
    def embedding_dim(self):
        return self.encoder.output_shape[-1]
    
    @property
    def output_signature(self):
        return TensorSpec(shape = (None, ), dtype = 'int32')
    
    @property
    def default_loss_config(self):
        return {'distance_metric' : self.distance_metric}
    
    @property
    def default_metrics_config(self):
        return self.default_loss_config

    def __str__(self):
        des = super().__str__()
        des += "- Embedding dim   : {}\n".format(self.embedding_dim)
        des += "- Distance metric : {}\n".format(self.distance_metric)
        return des

    def _add_tracked_variable(self, tracked_type, name, value):
        if isinstance(value, keras.Model) and hasattr(value.loss, 'init_variables'):
            value.loss.init_variables(value)
            if not hasattr(self, 'process_batch_output'):
                self.process_batch_output = _reshape_labels_for_ge2e
                self._init_processing_functions()
        
        return super()._add_tracked_variable(tracked_type, name, value)
    
    def compile(self, *, loss = None, loss_config = {}, ** kwargs):
        if loss is None: loss = getattr(self, '_default_loss', None)
        
        loss_config = {** self.default_loss_config, ** loss_config}
        
        loss   = get_loss(loss, ** loss_config)
        if hasattr(loss, 'init_variables'):
            loss.init_variables(self.model)
        
            if not hasattr(self, 'process_batch_output'):
                self.process_batch_output = _reshape_labels_for_ge2e
                self._init_processing_functions()

        super().compile(loss = loss, ** kwargs)
            
    
    def prepare_dataset(self, dataset, mode, ** kwargs):
        config = self.get_dataset_config(mode, ** kwargs)
        
        if isinstance(dataset, pd.DataFrame):
            load_fn = config.pop('prepare_fn', None)
            if load_fn is not None and not hasattr(self, 'prepare_output'):
                load_fn     = self.prepare_input
                signature   = self.unbatched_input_signature
            else:
                signature   = [self.unbatched_input_signature, self.unbatched_output_signature]
            
            dataset = GE2EGenerator(
                dataset,
                load_fn = load_fn,
                output_signature    = signature,
                
                ** kwargs
            )
            logger.info('{} generator created :\n{}'.format(mode.capitalize(), dataset))
        elif isinstance(dataset, GE2EGenerator):
            config.pop('prepare_fn', None)
            if not dataset.batch_size:
                dataset.set_batch_size(config['batch_size'])
            else:
                config['batch_size'] = dataset.batch_size
        
        for k in ('cache', 'shuffle'): config[k] = False
        
        return prepare_dataset(dataset, ** config)
    
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
    def embed(self, data, batch_size = 128, pad_value = 0., processed = False, tqdm = lambda x: x, ** kwargs):
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
                3) Encodes the batch with `self.encoder`
            
            This function is the core of the `Encoder` model, as embeddings are used for everything (predict similarity / distance), label predictions, clustering, make funny colored plots, ...
            
            Note : if batching is used (i.e. the data is sequential of variable lengths), make sure that `self.encoder` correctly supports masking. Otherwise, embeddings may differ between `batch_size = 1` and `batch_size > 1` due to padding (which is unexpected !)
            
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
        
        if not processed:
            with time_logger.timer('processing'):
                if not isinstance(data, (list, tuple, pd.DataFrame)): data = [data]

                inputs = self.get_input(data, ** kwargs)
        else:
            inputs = data
        
        encoder     = self.encoder
        should_pad  = any(s is None for s in self.input_shape[1:])
        
        embeddings = []
        for idx in tqdm(range(0, len(inputs), batch_size)):
            with time_logger.timer('batching'):
                batch = inputs[idx : idx + batch_size]
                batch = pad_batch(batch, pad_value = pad_value) if should_pad else ops.stack(batch, axis = 0)

            with time_logger.timer('encoding'):
                embeddings.append(encoder(batch, training = False))

        return ops.concatenate(embeddings, axis = 0) if len(embeddings) > 1 else embeddings[0]
    
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
        
        plot_embedding(embedded, ids = ids, ** kwargs)

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
                data,
                *,
                
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
        if not isinstance(data, pd.DataFrame): data = pd.DataFrame(data)

        round_tqdm  = tqdm
        
        cache_size  = max(cache_size, batch_size)

        embeddings = load_embedding(
            directory       = directory,
            embedding_name  = embedding_name,
            aggregate_on    = None,
            ** kwargs
        )
        
        processed, to_process = [], data
        if embeddings is not None:
            if 'filename' in main_key:
                data[main_key]       = data[main_key].apply(path_to_unix)
                embeddings[main_key] = embeddings[main_key].apply(path_to_unix)

            if not overwrite:
                mask        = data[main_key].isin(embeddings[main_key])
                to_process  = data[~mask]
                processed   = embeddings[embeddings[main_key].isin(data[main_key])]
            else:
                embeddings  = embeddings[~embeddings[main_key].isin(data[main_key])]

        logger.info("Processing data...\n  {} items already processed\n  {} items to process".format(len(data) - len(to_process), len(to_process)))

        if len(to_process) == 0: return processed

        if isinstance(processed, pd.DataFrame): processed = processed.to_dict('records')
        
        data_loader     = None
        if cache_size > batch_size and len(to_process) > batch_size:
            data_loader = iter(prepare_dataset(
                data, process_fn = self.get_input, cache = False, batch_size = 0
            ))

        if verbose or data_loader is not None: round_tqdm  = lambda x: x
        else: tqdm = lambda x: x

        to_process = to_process.to_dict('records')

        n, n_round = 0 if embeddings is None else len(embeddings), len(to_process) // cache_size + 1
        for i in round_tqdm(range(n_round)):
            if verbose: logger.info("Round {} / {}".format(i + 1, n_round))

            batch = to_process[i * cache_size : (i + 1) * cache_size]
            if len(batch) == 0: continue

            if data_loader is not None:
                inputs = [next(data_loader) for _ in tqdm(range(len(batch)))]
            else:
                inputs = [self.get_input(b) for b in batch]

            embedded = self.embed(
                inputs, batch_size = batch_size, tqdm = tqdm, processed = True, ** kwargs
            )
            if ops.is_tensor(embedded): embedded = ops.convert_to_numpy(embedded)

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
            embedding_name  = embedding_name if embedding_name else self.name,
            ** kwargs
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({'distance_metric' : self.distance_metric})
        
        return config

def _reshape_labels_for_ge2e(output, ** _):
    uniques, indexes = ops.unique(ops.reshape(output, [-1]), return_inverse = True)
    return ops.reshape(indexes, [ops.shape(uniques)[0], -1])
