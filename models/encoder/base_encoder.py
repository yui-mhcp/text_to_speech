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

from functools import wraps

from utils import *
from loggers import timer, time_logger
from utils.datasets import prepare_dataset
from utils.search.vectors import build_vectors_db
from utils.keras_utils import TensorSpec, ops, tree
from models.interfaces.base_model import BaseModel
from custom_train_objects import GE2EGenerator, get_loss

logger = logging.getLogger(__name__)

DEPRECATED_CONFIG = ('threshold', 'embed_distance')

def _sort_by_length(data):
    if isinstance(data, dict):
        return len(data.get('text'))
    elif isinstance(data, str) or ops.is_array(data):
        return len(data)
    return 0

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
    
    def prepare_for_xla(self, *, inputs, pad_multiple = 256, ** kwargs):
        if self.pad_value is not None and ops.is_array(inputs) and ops.rank(inputs) in (2, 3):
            inputs = pad_to_multiple(
                inputs, pad_multiple, axis = 1, constant_values = self.pad_value
            )
        kwargs['inputs'] = inputs
        return kwargs
    
    @property
    def pad_value(self):
        return None
    
    @property
    def embeddings(self):
        if self.__embeddings is None: self.__embeddings = self.load_embeddings()
        return self.__embeddings

    @property
    def encoder(self):
        return self.model
    
    @property
    def embedding_dim(self):
        return self.model.embedding_dim if hasattr(self.model, 'embedding_dim') else self.encoder.output_shape[-1]
        
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

    def _maybe_init_loss_weights(self, model, loss):
        if hasattr(loss, 'init_variables'):
            loss.init_variables(model)
            if not hasattr(self, 'process_batch_output'):
                self.process_batch_output = _reshape_labels_for_ge2e
                self._init_processing_functions()

    def _add_tracked_variable(self, tracked_type, name, value):
        if isinstance(value, keras.Model):
            self._maybe_init_loss_weights(value, value.loss)
        
        return super()._add_tracked_variable(tracked_type, name, value)
    
    def compile(self, *, loss = None, loss_config = {}, ** kwargs):
        if loss is None: loss = getattr(self, '_default_loss', None)
        
        loss_config = {** self.default_loss_config, ** loss_config}
        
        loss = get_loss(loss, ** loss_config)
        self._maybe_init_loss_weights(self.model, loss)

        super().compile(loss = loss, ** kwargs)
    
    def get_dataset_config(self, mode, ** kwargs):
        if self.pad_value is not None:
            kwargs.setdefault('pad_kwargs', {}).update({'padding_values' : (self.pad_value, 0)})
        
        return super().get_dataset_config(mode, ** kwargs)

    def prepare_dataset(self, dataset, mode, ** kwargs):
        config = self.get_dataset_config(mode, ** kwargs)
        
        if is_dataframe(dataset):
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
    
    @apply_on_batch(default_batch_size = 8, sort_key = _sort_by_length)
    @timer
    def embed(self,
              data  = None,
              *,
              
              processed = False,
              
              keys  = 'dense',
              to_numpy  = False,
              return_raw    = None,
              
              encoder   = None,
              
              ** kwargs
             ):
        """
            Embed a list of data
            
            Arguments :
                - data  : the data to embed, any type supported by `self.get_input`
                - processed : whether `data` is already processed or not
                - to_numpy  : whether to convert embeddings (`Tensor`) to `ndarray`
                - return_raw    : whether the output should be raw `Tensor / ndarray` or `BaseEmbedding` class
            Return :
                - embeddings    : the embedded data
                    If `return_raw` : `ndarray` (if `to_numpy = True`) else `Tensor`
                    Else            : `BaseEmbedding` class(es)
        """
        if encoder is None: encoder = self
        
        with time_logger.timer('processing'):
            inputs = data if processed else self.get_input(data, ** kwargs)
            
            inputs = stack_batch(
                inputs,
                dtype   = ops.dtype_to_str(getattr(inputs[0], 'dtype', type(inputs[0]))),
                pad_value   = self.pad_value,
                maybe_pad   = self.pad_value is not None
            )
        
        with time_logger.timer('embedding'):
            out = encoder(inputs, return_mask = True, as_dict = True)
            if to_numpy:
                if not isinstance(out, (tuple, dict)):
                    out = ops.convert_to_numpy(out)
                else:
                    out = tree.map_structure(ops.convert_to_numpy, out)
            
            embeddings = getattr(out, 'output', out)
        
        if keys and isinstance(embeddings, dict):
            if return_raw is None: return_raw = False
            
            embeddings = {k : v for k, v in embeddings.items() if k in keys}
            if len(embeddings) == 1:
                embeddings = list(embeddings.values())[0]
        
        if isinstance(embeddings, dict) and not return_raw:
            for k, v in embeddings.items():
                try:
                    embeddings[k] = build_vectors_db(
                        vectors = embeddings[k],
                        mode    = k,
                        data    = data,
                        mask    = out.mask,
                        inputs  = inputs,
                        model   = self,
                        ** kwargs
                    )
                except Exception as e:
                    embeddings[k] = v
        elif return_raw is False:
            embeddings = build_vectors_db(
                vectors = embeddings,
                data    = data,
                mask    = out.mask,
                inputs  = inputs,
                model   = self,
                ** kwargs
            )
        
        return embeddings
    
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
        if is_dataframe(data):
            col_id  = ids if isinstance(ids, str) else 'id'
            ids     = data[col_id].values if col_id in data.columns else None
        elif isinstance(data, dict):
            data, ids = list(data.keys()), list(data.values())
        
        embedded = self.embed(data, batch_size = batch_size, to_numpy = True)
        
        plot_embedding(embedded, ids = ids, ** kwargs)

    @timer
    def predict_distance(self, x, y, as_matrix = True, ** kwargs):
        """
            Return a score of distance between x and y
            
            Arguments :
                - x : embedding vector(s), or any valid value for `self.embed`
                - y : embedding vector(s), or any valid value for `self.embed`
                - as_matrix : whether to return a matrix of distances between each pair, or a 1-1 distance (if `True`, prefer using `self.predict_distance_matrix`)
            Return :
                - distance : `Tensor` of distance scores
                    If `as_matrix = False`:
                        - 1-D vector representing the distance between `x[i]` and `y[i]`
                    else:
                        - 2-D matrix representing the distance between `x[i]` and `y[j]`
            
            Note :
            - a *distance score* means that the higher the score is, the more the data is different
            - a *similarity score* means that the higher the score is, the more the data is similar
        """
        if not ops.is_array(x) or x.shape[-1] != self.embedding_dim: x = self.embed(x)
        if not ops.is_array(y) or y.shape[-1] != self.embedding_dim: y = self.embed(y)
        
        if len(x.shape) == 1: x = ops.expand_dims(x, axis = 0)
        if len(y.shape) == 1: y = ops.expand_dims(y, axis = 0)
        
        return distance(
            x, y, self.distance_metric, as_matrix = as_matrix, force_distance = True, ** kwargs
        )
    
    @wraps(predict_distance)
    def predict_similarity(self, * args, ** kwargs):
        """ returns `1. - self.predict_distance(...)` """
        return - self.predict_distance(* args, ** kwargs)
    
    predict_distance_matrix     = partial(predict_distance, as_matrix = True)
    predict_similarity_matrix   = partial(predict_similarity, as_matrix = True)
    
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
        import pandas as pd
        
        if directory is None: directory = self.pred_dir
        if not isinstance(data, pd.DataFrame): data = pd.DataFrame(data)

        round_tqdm  = tqdm
        
        cache_size  = max(cache_size, batch_size)

        embeddings = load_embeddings(
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

