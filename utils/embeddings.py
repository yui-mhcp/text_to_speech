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
import random
import logging
import numpy as np

from tqdm import tqdm
from multiprocessing import cpu_count

from .pandas_utils import is_dataframe
from .sequence_utils import pad_batch
from .file_utils import load_data, dump_data, path_to_unix, remove_path
from .keras_utils import TensorSpec, ops, graph_compile

logger = logging.getLogger(__name__)

_embeddings_file_ext    = {'.csv', '.npy', '.pkl', 'pdpkl', '.embeddings.h5', '.h5'}
_default_embeddings_ext = '.h5'

def get_embeddings_file_ext(filename):
    """ Returns a valid extension for `filename` such that `filename + ext` exists """
    for ext in _embeddings_file_ext:
        if os.path.exists(filename + ext): return ext
    return None

def save_embeddings(filename, embeddings, *, directory = None, remove_file_prefix = True):
    """
        Save `embeddings` to the given `filename`
        
        Arguments :
            - filename  : the file in which to store the embeddings. Must have one of the supported file format for embeddings.
            - embeddings    : `pd.DataFrame` or raw embeddings to store
            
            - directory : directory in which to put the file (optional)
            - remove_file_prefix    : whether to remove a specific file prefix to `filename*` columns (only relevant if `embeddings` is a `pd.DataFrame`)
                If `True`, removes the `utils.datasets.get_dataset_dir` prefix
                It is useful when saving datasets, as the dataset directory may differ between hosts, meaning that filenames also differ.
                Note that it is removed only at the start of filenames such that if your filename is not in the dataset directory, it will have no effect ;)
    """
    if directory: filename = os.path.join(directory, filename)
    if not os.path.splitext(filename)[1]:
        filename += _default_embeddings_ext
    elif not any(filename.endswith(ext) for ext in _allowed_embeddings_ext):
        raise ValueError('Unsupported embeddings file extension !\n  Accepted : {}\n  Got : {}'.format(_allowed_embeddings_ext, filename))
    
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    
    if '{}' in filename:
        embedding_dim   = embeddings_to_np(
            embeddings.iloc[:1] if is_dataframe(embeddings) else embeddings
        ).shape[-1]
        filename        = filename.format(embedding_dim)
    
    if filename.endswith('.npy'):
        embeddings = embeddings_to_np(embeddings)
    elif remove_file_prefix:
        embeddings = remove_path(embeddings, remove_file_prefix)
    
    logger.debug('Saving embeddings to {}'.format(filename))
    
    dump_data(filename, embeddings)
    
    return filename

def load_embeddings(filename,
                    *,
                    
                    dataset     = None,
                    filename_prefix  = True,
                   
                    aggregate_on    = 'id',
                    aggregate_mode  = 0,
                    aggregate_name  = 'speaker_embedding',
                    
                    ** kwargs
                   ):
    """
        Load embeddings from file (csv / npy / pkl) and create an aggregation version (if expected)
        
        Arguments :
            - filename  : the file containing the embeddings
            
            - dataset   : the dataset on which to merge embeddings
            - filename_prefix   : a path to add at all filenames' start (i.e. each value in result['filename'])
                if `True`, it adds the `get_dataset_dir` as prefix
                Note that if the filename exists as is, it will have no effect
            
            - aggregate_on  : the column to aggregate on
            - aggregate_mode    : the mode for the aggregation
            - aggregate_name    : the name for the aggregated embeddings' column (default to `speaker_embedding` for retro-compatibility)
        Return :
            - embeddings or dataset merged with embeddings (merge is done on columns that are both in `dataset` and `embeddings`)
    """
    if not os.path.exists(filename):
        ext = get_embeddings_file_ext(filename)
        if not ext:
            logger.warning('Embeddings file {} does not exist !'.format(filename))
            return dataset
        
        filename += ext
    
    embeddings  = load_data(filename)
    if not is_dataframe(embeddings): return embeddings
    
    if any('Unnamed:' in col for col in embeddings.columns):
        embeddings = embeddings.drop(
            columns = [col for col in embeddings.columns if 'Unnamed:' in col]
        )

    if aggregate_on:
        embeddings = aggregate_embeddings(
            embeddings, aggregate_on, aggregation_name = aggregate_name, mode = aggregate_mode
        )

    for col in embeddings.columns:
        if 'embedding' not in col or isinstance(embeddings.loc[0, col], np.ndarray): continue
        embeddings[col] = embeddings[col].apply(embeddings_to_np)

    if filename_prefix:
        if filename_prefix is True:
            try:
                from .datasets import get_dataset_dir
                filename_prefix = get_dataset_dir()
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Unable to import `get_dataset_dir`. Explicitely provide `prefix`')

        for col in embeddings.columns:
            if 'filename' not in col: continue
            embeddings[col] = embeddings[col].apply(
                lambda f: '{}/{}'.format(filename_prefix, f)
            )
    
    if dataset is None: return embeddings
    
    intersect = list(set(embeddings.columns).intersection(set(dataset.columns)))
    
    for col in intersect:
        if embeddings[col].dtype != dataset[col].dtype:
            embeddings[col] = embeddings[col].apply(dataset[col].dtype)
        
        if 'filename' in col:
            embeddings[col] = embeddings[col].apply(path_to_unix)
            dataset[col]    = dataset[col].apply(path_to_unix)
    
    logger.debug('Merging embeddings with dataset on columns {}'.format(intersect))
    dataset = dataset.merge(embeddings, on = intersect)

    if len(dataset) == 0:
        raise ValueError('Merge resulted in an empty dataframe !\n  Columns : {}\n  Embeddings : {}'.format(intersect, embeddings))
    
    dataset = dataset.dropna(
        axis = 'index', subset = [c for c in dataset.columns if 'embedding' in c]
    )
    
    return dataset

def embeddings_to_np(embeddings, col = 'embedding', dtype = float, force_np = True):
    """
        Return a numpy matrix of embeddings from a dataframe column
        
        Arguments :
            - embeddings    : the embeddings to load / convert
            - col   : the column to use if `embeddings` is a `pd.DataFrame`
            - dtype : the embeddings dtype (if string representation)
            - force_np  : whether to convert `Tensor` to `np.ndarray` or not
        Return :
            - embedding : `np.ndarray` of dtype `float32`, the embeddings
                If `force_np == False`, the result may be a `Tensor`
    """
    # if it is a string representation of a numpy matrix
    if isinstance(embeddings, str):
        embeddings = embeddings.strip()
        if embeddings.startswith('['):
            # if it is a 2D-matrix
            if embeddings.startswith('[['):
                return pad_batch([
                    embeddings_to_np(xi + ']', dtype = dtype)
                    for xi in embeddings[1:-1].split(']')
                ])
            # if it is a 1D-vector
            sep = '\t' if ', ' not in embeddings else ', '
            return np.fromstring(embeddings[1:-1], dtype = dtype, sep = sep).astype(np.float32)
        elif os.path.isfile(embeddings):
            from .embeddings_io import load_embeddingss
            
            return embeddings_to_np(load_embeddingss(embeddings), col = col, dtype = dtype)
        else:
            raise ValueError("The file {} does not exist !".format(embeddings))
    
    elif isinstance(embeddings, np.ndarray):
        return embeddings
    elif ops.is_tensor(embeddings):
        return ops.convert_to_numpy(embeddings) if force_np else embeddings
    elif is_dataframe(embeddings):
        embeddings = [embeddings_to_np(e) for e in embeddings[col].values]
        if len(embeddings[0].shape) == 1: return np.array(embeddings)
        
        return pad_batch(embeddings)
    else:
        raise ValueError("Invalid type of embeddings : {}\n{}".format(
            type(embeddings), embeddings
        ))

def aggregate_embeddings(dataset,
                         column = 'id',
                         embedding_col  = 'embedding',
                         aggregation_name   = 'speaker_embedding',
                         mode = 0
                        ):
    """ Aggregates the `embedding_col` column by grouping on `column` """
    if embedding_col not in dataset.columns:
        raise ValueError('The embedding column {} is not available in {}'.format(
            embedding_col, dataset.columns
        ))
    
    from utils.pandas_utils import aggregate_df
    
    if aggregation_name in dataset.columns: dataset.pop(aggregation_name)
    
    if column not in dataset.columns:
        if 'id' not in dataset.columns:
            raise RuntimeError('The column {} is not available in {} !'.format(
                column, dataset.columns
            ))
        logger.warning('The column {} is not available in {}. Using by default `id`'.format(
            column, dataset.columns
        ))
        column = 'id'
    
    if column != 'id' and 'id' in dataset.columns and np.any(dataset[column].isnan()):
        dataset = dataset.fillna({
            col : dataset['id'] for col in (column if isinstance(column, list) else [column])
        })
    
    return aggregate_df(
        dataset, group_by = column, columns = embedding_col, merge = True, ** {
            aggregation_name : mode
        }
    )

def select_embedding(embeddings, mode = 'random', ** kwargs):
    """
        Returns a single embedding (`np.ndarray` with rank 1) from a collection of embeddings 
        
        Arguments : 
            - embeddings    : `pd.DataFrame` with 'embedding' col or `np.ndarray` (2D matrix)
            - mode      : selection mode (int / 'avg' / 'mean' / 'average' / 'random' / callable)
            - kwargs    : filtering criteria (if `embeddings` is a `pd.DataFrame`) (see `filter_df`)
        Return :
            - embedding : 1D `np.ndarray`
    """
    if is_dataframe(embeddings):
        filtered_embeddings = embeddings
        if any(k in embeddings.columns for k in kwargs.keys()):
            from utils.pandas_utils import filter_df
            
            filtered_embeddings = filter_df(embeddings, ** kwargs)

            if len(filtered_embeddings) == 0:
                logger.warning('No embedding respects filters {}'.format(kwargs))
                filtered_embeddings = embeddings
        np_embeddings = embeddings_to_np(filtered_embeddings)
    else:
        np_embeddings = embeddings_to_np(embeddings, force_np = False)
        if len(np_embeddings.shape) == 1: np_embeddings = np.expand_dims(np_embeddings, axis = 0)
    
    if isinstance(mode, int):
        return np_embeddings[mode]
    elif mode in ('mean', 'avg', 'average'):
        return np.mean(np_embeddings, axis = 0)
    elif mode == 'random':
        idx = random.randrange(len(np_embeddings))
        if logger.isEnabledFor(logging.DEBUG): logger.debug('Selected embedding : {}'.format(idx))
        return np_embeddings[idx]
    elif callable(mode):
        return mode(np_embeddings)
    else:
        raise ValueError("Unknown embedding selection mode !\n  Accepted : {}\n  Got : {}".format(
            _accepted_modes, mode
        ))

def embed_dataset(directory,
                  dataset,
                  embed_fn,
                  batch_size    = 1,
                  
                  load_fn   = None,
                  cache_size    = 10000,
                  max_workers   = cpu_count(),
                  
                  save_every    = 10000,
                  overwrite     = False,
                  embedding_name    = None,
                  
                  tqdm = tqdm,
                  verbose   = True,
                  
                  ** kwargs
                 ):
    import pandas as pd
    
    if load_fn is None:
        cache_size, round_tqdm, tqdm = batch_size, tqdm, None
    else:
        from .threading import Consumer
        
        round_tqdm      = lambda x: x
        cache_size      = max(cache_size, batch_size)
        load_consumer   = Consumer(load_fn, max_workers = max_workers, keep_result = False)
        load_consumer.start()
    
    embeddings = None
    if not overwrite:
        embeddings = load_embeddings(
            directory = directory, embedding_name = embedding_name, aggregate_on = None, ** kwargs
        )
    
    processed, to_process = [], dataset
    if embeddings is not None:
        processed   = dataset['filename'].isin(set(embeddings['filename'].values))
        to_process  = dataset[~processed]

    if len(to_process) == 0:
        logger.info("Dataset already processed !")
        return embeddings
    
    to_process = to_process.to_dict('records')

    logger.info("Processing dataset...\n  {} utterances already processed\n  {} utterances to process".format(len(dataset) - len(to_process), len(to_process)))

    n, n_round = 0 if embeddings is None else len(embeddings), len(to_process) // cache_size + 1
    for i in round_tqdm(range(n_round)):
        if load_fn is not None: logger.info("Round {} / {}".format(i + 1, n_round))
        
        batch = to_process[i * cache_size : (i + 1) * cache_size]
        if len(batch) == 0: continue
        
        if load_fn is not None:
            batch = load_consumer.extend_and_wait(batch, tqdm = tqdm, stop = False, ** kwargs)

        embedded = embed_fn(batch, batch_size = batch_size, tqdm = tqdm, ** kwargs)
        if hasattr(embedded, 'numpy'): embedded = embedded.numpy()
        
        embedded = pd.DataFrame([
            {'filename' : infos['filename'], 'id' : infos['id'], 'embedding' : emb} 
            for infos, emb in zip(to_process[i * cache_size : (i + 1) * cache_size], embedded)
        ])
        embeddings = pd.concat([embeddings, embedded], ignore_index = True) if embeddings is not None else embedded
        
        if len(embeddings) - n >= save_every:
            n = len(embeddings)
            if load_fn is not None: logger.info("Saving at utterance {}".format(len(embeddings)))
            save_embeddings(
                directory, embeddings, embedding_name, remove_file_prefix = True
            )
    
    if n < len(embeddings):
        save_embeddings(
            directory, embeddings, embedding_name, remove_file_prefix = True
        )
    
    return embeddings


@graph_compile
def compute_centroids(embeddings    : TensorSpec(shape = (None, None), dtype = 'float'),
                      ids       : TensorSpec(shape = (None, )),
                      num_ids   : TensorSpec(shape = (), dtype = 'int32', static = True) = None,
                      sorted    = False
                     ):
    """
        Compute the mean embeddings (named the `centroids`) for each id
        Arguments :
            - embeddings    : 2D matrix of embeddings
            - ids   : array of ids where `embeddings[i]` has `ids[i]`
        Return :
            - (unique_ids, centroids)
                - unique_ids    : vector of unique ids
                - centroids     : centroids[i] is the centroid associated to embeddings of ids[i]
    """
    if not sorted and ops.is_numeric(ids):
        sorted_indexes  = ops.argsort(ids)
        embeddings  = ops.take(embeddings, sorted_indexes, axis = 0)
        ids = ops.take(ids, sorted_indexes, axis = 0)
        sorted = True
    
    if num_ids is None or not ops.is_int(ids):
        uniques, indices = ops.unique(ids, return_inverse = True)
        num_ids = ops.shape(uniques)[0]
    else:
        indices = ids
        uniques = ops.arange(num_ids, dtype = 'int32') if ops.is_tensor(embeddings) else np.arange(num_ids, dtype = 'int32')
    
    if not sorted:
        sorted_indexes  = ops.argsort(indices)
        embeddings  = ops.take(embeddings, sorted_indexes, axis = 0)
        indices     = ops.take(indices, sorted_indexes, axis = 0)
    
    return uniques, ops.segment_mean(embeddings, indices, num_segments = num_ids, sorted = True)

@graph_compile
def get_embeddings_with_ids(embeddings  : TensorSpec(shape = (None, None), dtype = 'float'),
                            assignment  : TensorSpec(shape = (None, )),
                            ids         : TensorSpec(shape = (None, ))
                           ):
    """
        Returns a subset of `embeddings` and `assignment` with the expected `ids`
        
        It is a graph-compatible version of regular numpy masking :
        ```python
            sub_embeddings, sub_ids = get_embeddings_with_ids(embeddings, assignment, ids)
            
            # equivalent to (where `embeddings` and `assignment` are `np.ndarray`'s)
            mask = np.isin(assigment, ids)
            sub_embeddings, sub_ids = embeddings[mask], assignment[mask]
        ```

        Arguments :
            - embeddings    : `Tensor` with shape `(n_embeddings, embedding_dim)`
            - assignment    : `Tensor` with shape `(n_embeddings, )`, the embeddings ids
            - ids       : `Tensor`, the expected ids to keep
        Return :
            - (embeddings, assignment)  : subset of `embeddings` and `assignment` with valid id
    """
    mask = ops.isin(assignment, ids)
    return embeddings[mask], assignment[mask]

def visualize_embeddings(embeddings,
                         metadata,
                         log_dir    = 'embedding_visualization',
                         images     = None,
                         image_size = 128,
                         label_col  = 'id',
                         metadata_filename = 'metadata.tsv'
                        ):
    import tensorflow as tf
    
    from tensorboard.plugins import projector
    
    os.makedirs(log_dir, exist_ok = True)
    
    embeddings_var = embeddings
    if not isinstance(embeddings, tf.Variable):
        embeddings_var = tf.Variable(embeddings)
    
    ckpt = tf.train.Checkpoint(embedding = embeddings_var)
    ckpt.save(os.path.join(log_dir, 'embedding.ckpt'))
    
    if is_dataframe(metadata):
        metadata = metadata[[label_col] + [c for c in metadata.columns if c != label_col]]

        metadata.to_csv(os.path.join(log_dir, metadata_filename), sep = '\t', index = False)
    else:
        with open(os.path.join(log_dir, metadata_filename), 'w', encoding = 'utf-8') as file:
            file.write(''.join([str(v) + '\n' for v in metadata]))
    
    config      = projector.ProjectorConfig()
    embedding   = config.embeddings.add()
    embedding.tensor_name   =  'embedding/.ATTRIBUTES/VARIABLE_VALUE'
    
    embedding.metadata_path = metadata_filename
    
    if images is not None:
        from utils.image import build_sprite
        
        build_sprite(images, directory = log_dir, image_size = image_size, filename = 'sprite.jpg')
        embedding.sprite.image_path = 'sprite.jpg'
        embedding.sprite.single_image_dim.extend([image_size, image_size])
    
    projector.visualize_embeddings(log_dir, config)
    