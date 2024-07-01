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
import random
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from multiprocessing import cpu_count

from .file_utils import load_data, dump_data, path_to_unix
from utils.keras_utils import TensorSpec, graph_compile, ops

logger = logging.getLogger(__name__)

_allowed_embeddings_ext = ('.csv', '.npy', '.pkl', '.pdpkl', '.embeddings.h5')
_default_embedding_ext  = '.embeddings.h5'
_embedding_filename = 'embeddings_{}'

_accepted_modes     = ('random', 'mean', 'average', 'avg', 'int', 'callable')

def get_embedding_file_ext(filename):
    """ Returns a valid extension for `filename` such that `filename + ext` exists """
    for ext in _allowed_embeddings_ext:
        if os.path.exists(filename + ext): return ext
    return None

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
        if embeddings.startswith('['):
            # if it is a 2D-matrix
            if embeddings.startswith('[['):
                from utils.sequence_utils import pad_batch
                
                return pad_batch([
                    embeddings_to_np(xi + ']', dtype = dtype)
                    for xi in embeddings[1:-1].split(']')
                ])
            # if it is a 1D-vector
            sep = '\t' if ', ' not in embeddings else ', '
            return np.fromstring(embeddings[1:-1], dtype = dtype, sep = sep).astype(np.float32)
        elif not os.path.isfile(embeddings):
            raise ValueError("The file {} does not exist !".format(embeddings))
            
        return embeddings_to_np(load_embeddings(embeddings), col = col, dtype = dtype)
    
    elif isinstance(embeddings, np.ndarray):    return embeddings
    elif hasattr(embeddings, 'numpy'):          return ops.convert_to_numpy(embeddings) if force_np else embeddings
    elif isinstance(embeddings, pd.DataFrame):
        embeddings = [embeddings_to_np(e) for e in embeddings[col].values]
        if len(embeddings[0].shape) == 1: return np.array(embeddings)
        
        from utils.sequence_utils import pad_batch
        
        return pad_batch(embeddings)
    else:
        raise ValueError("Invalid type of embeddings : {}\n{}".format(
            type(embeddings), embeddings
        ))

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
    if isinstance(embeddings, pd.DataFrame):
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

def load_embedding(directory,
                   embedding_name   = _embedding_filename,
                   embedding_dim    = None,
                   dataset          = None,
                   filename_prefix  = True,
                   
                   aggregate_on     = 'id',
                   aggregate_mode   = 0,
                   aggregate_name   = 'speaker_embedding',
                   ** kwargs
                  ):
    """
        Load embeddings from file (csv / npy / pkl) and create an aggregation version (if expected)
        
        Arguments :
            - directory     : directory in which embeddings are stored (must be 'embeddings' or have a sub-directory named 'embeddings')
                It can also be the embeddings' filename
            - embedding_name    : the embeddings' filename (can contains '{}' which will be formatted by `embedding_dim`)
            - embedding_dim : dimension of the embedding (used to format `filename` if required)
            
            - filename_prefix   : a path to add at all filenames' start (i.e. each value in result['filename'])
                if `True`, it adds the `get_dataset_dir` as prefix
                Note that if the filename exists as is, it will have no effect
            
            - dataset       : the dataset on which to merge embeddings
            
            - aggregate_on  : the column to aggregate on
            - aggregate_mode    : the mode for the aggregation
            - aggregate_name    : the name for the aggregated embeddings' column (default to `speaker_embedding` for retro-compatibility)
        Return :
            - embeddings or dataset merged with embeddings (merge is done on columns that are both in `dataset` and `embeddings`)
        
        Note : the effective loading filename is
            - directory : if `os.path.isfile(directory)`
            - embedding_name    : if `os.path.exists(embedding_name)`
            - else  : `os.path.join(directory, embedding_name)`
        
    """
    if os.path.isfile(directory):
        emb_file = directory
    else:
        if '{}' in embedding_name:
            assert embedding_dim is not None
            embedding_name = embedding_name.format(embedding_dim)
        
        if os.path.exists(embedding_name):
            emb_file = embedding_name
        else:
            if not directory.endswith('embeddings'):
                directory = os.path.join(directory, 'embeddings')
            emb_file = os.path.join(directory, embedding_name)
    
    if not os.path.splitext(emb_file)[1]:
        ext = get_embedding_file_ext(emb_file)
        if ext: emb_file += ext
    
    if not os.path.exists(emb_file):
        logger.warning('Embedding file {} does not exist !'.format(emb_file))
        return None
    
    embeddings  = load_data(emb_file)
    if isinstance(embeddings, dict):
        embeddings = pd.DataFrame({k : list(v) for k, v in embeddings.items()})
    elif not isinstance(embeddings, pd.DataFrame):
        return embeddings
    elif any('Unnamed:' in col for col in embeddings.columns):
        embeddings = embeddings.drop(
            columns = [col for col in embeddings.columns if 'Unnamed:' in col]
        )
    
    for col in embeddings.columns:
        if 'embedding' not in col or isinstance(embeddings.loc[0, col], np.ndarray): continue
        embeddings[col] = embeddings[col].apply(embeddings_to_np)
    
    if aggregate_on is not None and aggregate_on in embeddings.columns:
        embeddings = aggregate_embeddings(
            embeddings, aggregate_on, aggregation_name = aggregate_name, mode = aggregate_mode
        )
    
    if filename_prefix is not None and 'filename' in embeddings.columns:
        if filename_prefix is True:
            try:
                from utils.datasets import get_dataset_dir
                filename_prefix = get_dataset_dir()
            except (ImportError, ModuleNotFoundError):
                pass
        
        if isinstance(filename_prefix, str):
            embeddings['filename'] = embeddings['filename'].apply(
                lambda f: os.path.join(filename_prefix, f) if not f.startswith(filename_prefix) else f
            )
    
    if dataset is None: return embeddings
    
    intersect = list(set(embeddings.columns).intersection(set(dataset.columns)))
    
    for col in intersect:
        if embeddings[col].dtype != dataset[col].dtype:
            embeddings[col] = embeddings[col].apply(str)
        
        if 'filename' in col:
            embeddings[col] = embeddings[col].apply(path_to_unix)
            dataset[col]    = dataset[col].apply(path_to_unix)
    
    logger.debug('Merging embeddings with dataset on columns {}'.format(intersect))
    dataset = pd.merge(dataset, embeddings, on = intersect)

    if len(dataset) == 0:
        raise ValueError('Merge resulted in an empty dataframe !\n  Columns : {}\n  Embeddings : {}'.format(intersect, embeddings))
    
    dataset = dataset.dropna(
        axis = 'index', subset = ['embedding']
    )
    
    return dataset

def save_embeddings(directory,
                    embeddings,
                    embedding_name  = _embedding_filename,
                    remove_file_prefix  = True
                   ):
    """
        Save `embeddings` to the given path
        
        Arguments :
            - directory : the embeddings' filename (if ending with a valid extension) or the directory
            - embeddings    : the embeddings to save
            embedding_name  : the filename basename
                If no extension, set to `_default_embedding_ext`
                If contains {}, formatted with the embedding dimension (`embeddings.shape[-1]`)
            - remove_file_prefix    : if a `filename` column is there, remove the `remove_file_prefix` from the start of filenames (only working when embeddings is a `pd.DataFrame`)
                If `True`, removes the `get_dataset_dir` prefix
                It is useful when saving datasets as the dataset directory may differ between hosts, meaning that filenames also differ.
                Note that it is removed only at the start of filenames such that if your filename is not in the dataset directory, it will have no effect ;)
    """
    if os.path.splitext(directory)[1] in _allowed_embeddings_ext:
        embedding_file = directory
    else:
        if not directory.endswith('embeddings'):
            directory = os.path.join(directory, 'embeddings')
        os.makedirs(directory, exist_ok = True)
        embedding_file = os.path.join(directory, embedding_name)
    
    if '{}' in embedding_file:
        embedding_dim   = embeddings_to_np(embeddings.iloc[:1]).shape[-1]
        embedding_file  = embedding_file.format(embedding_dim)
    
    ext = os.path.splitext(embedding_file)[1]
    if not ext: embedding_file += _default_embedding_ext
    elif ext not in _allowed_embeddings_ext:
        raise ValueError('Unknown embedding extension !\n  Accepted : {}\n  Got : {}'.format(
            _allowed_embeddings_ext, embedding_file
        ))
    
    if embedding_file.endswith('npy'): embeddings = embeddings_to_np(embeddings)
    elif remove_file_prefix is not None and isinstance(embeddings, pd.DataFrame) and 'filename' in embeddings.columns:
        if remove_file_prefix is True:
            try:
                from utils.datasets import get_dataset_dir
                remove_file_prefix = get_dataset_dir()
            except (ImportError, ModuleNotFoundError):
                pass
        
        if isinstance(remove_file_prefix, str):
            remove_file_prefix = path_to_unix(remove_file_prefix)
            embeddings['filename'] = embeddings['filename'].apply(
                lambda f: path_to_unix(f).lstrip(remove_file_prefix)
            )
    
    dump_data(embedding_file, embeddings)
    
    return embedding_file

def embed_dataset(directory,
                  dataset,
                  embed_fn,
                  batch_size    = 1,
                  
                  load_fn   = None,
                  cache_size    = 10000,
                  max_workers   = cpu_count(),
                  
                  save_every    = 10000,
                  overwrite     = False,
                  embedding_name    = _embedding_filename,
                  
                  tqdm = tqdm,
                  verbose   = True,
                  
                  ** kwargs
                 ):
    if load_fn is None:
        cache_size, round_tqdm, tqdm = batch_size, tqdm, None
    else:
        from utils.threading import Consumer
        
        round_tqdm      = lambda x: x
        cache_size      = max(cache_size, batch_size)
        load_consumer   = Consumer(load_fn, max_workers = max_workers, keep_result = False)
        load_consumer.start()
    
    embeddings = None
    if not overwrite:
        embeddings = load_embedding(
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


def pad_dataset_embedding(dataset, columns = None):
    """
        Pads `columns`' embeddings to have matrices with same dimensions, and adds a n_{col} column with the current number of embeddings (to allow to retrieve the original embeddings' matrix)
        The operation is skipped for columns with single embedding (1D array)
    """
    if columns is None: columns = [c for c in dataset.columns if c.endswith('embedding')]
    if not isinstance(columns, (list, tuple)): columns = [columns]
    
    for col in columns:
        if len(dataset.iloc[0][col].shape) == 1: continue
        
        dataset['n_{}'.format(col)] = dataset[col].apply(lambda e: len(e))
        dataset[col] = list(embeddings_to_np(dataset, col = col))
    
    return dataset

@graph_compile
def compute_centroids(embeddings    : TensorSpec(shape = (None, None), dtype = 'float'),
                      ids           : TensorSpec(shape = (None, )),
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
    uniques, indexes, counts = ops.unique(ids, return_inverse = True, return_counts = True)
    # mask.shape == [len(uniques), len(embeddings)]
    mask = ops.arange(ops.size(uniques))[:, None] == indexes[None, :]
    return uniques, ops.divide_no_nan(
        ops.sum(ops.where(
            mask[:, :, None],
            embeddings[None, :, :],
            ops.convert_to_tensor(0, embeddings.dtype)
        ), axis = 1),
        ops.cast(counts[:, None], dtype = embeddings.dtype)
    )

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
            mask = assignment.isin(ids)
            sub_embeddings, sub_ids = embeddings[mask], assignment[mask]
        ```

        Arguments :
            - embeddings    : `Tensor` with shape `(n_embeddings, embedding_dim)`
            - assignment    : `Tensor` with shape `(n_embeddings, )`, the embeddings ids
            - ids       : `Tensor`, the expected ids to keep
        Return :
            - (embeddings, assignment)  : subset of `embeddings` and `assignment` with valid id
    """
    mask    = ops.any(assignment[:, None] == ids[None, :], axis = -1)
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
    
    if isinstance(metadata, pd.DataFrame):
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
    