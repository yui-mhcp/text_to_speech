# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
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
import random
import logging
import numpy as np

from tqdm import tqdm
from multiprocessing import cpu_count

from .sequence_utils import pad_batch
from .keras import TensorSpec, ops, graph_compile
from .generic_utils import is_dataframe, filter_df, aggregate_df
from .file_utils import load_data, dump_data, path_to_unix, remove_path_prefix

logger = logging.getLogger(__name__)

_embeddings_file_ext    = {'.csv', '.npy', '.pkl', 'pdpkl', '.embeddings.h5', '.h5'}
_default_embeddings_ext = '.h5'

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
    elif is_dataframe(embeddings):
        embeddings = [embeddings_to_np(e) for e in embeddings[col].values]
        if len(embeddings[0].shape) == 1: return np.array(embeddings)
        
        return pad_batch(embeddings)
    elif ops.is_tensor(embeddings):
        return ops.convert_to_numpy(embeddings) if force_np else embeddings
    else:
        raise ValueError("Invalid type of embeddings : {}\n{}".format(
            type(embeddings), embeddings
        ))

def save_embeddings(filename, embeddings, *, directory = None, remove_file_prefix = True):
    """
        Save `embeddings` to the given `filename`
        
        Arguments :
            - filename  : the file in which to store the embeddings.
                          Must have one of the supported file format for embeddings.
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
    elif not filename.endswith(tuple(_allowed_embeddings_ext)):
        raise ValueError('Unsupported embeddings extension !\n  Accepted : {}\n  Got : {}'.format(
            _allowed_embeddings_ext, filename
        ))
    
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    
    if '{}' in filename:
        embedding_dim   = embeddings_to_np(
            embeddings.iloc[:1] if hasattr(embeddings, 'iloc') else embeddings[:1]
        ).shape[-1]
        filename        = filename.format(embedding_dim)
    
    if filename.endswith('.npy'):
        embeddings = embeddings_to_np(embeddings)
    elif remove_file_prefix:
        embeddings = remove_path_prefix(embeddings, remove_file_prefix)
    
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
        ext = _get_embeddings_file_ext(filename)
        if not ext:
            logger.warning('Embeddings file {} does not exist !'.format(filename))
            return dataset
        
        filename += ext
    
    embeddings  = load_data(filename)
    if ops.is_array(embeddings):    return embeddings
    elif isinstance(embeddings, dict):
        import pandas as pd
        embeddings = pd.DataFrame(embeddings)
    
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
            filtered_embeddings = filter_df(embeddings, ** kwargs)

            if len(filtered_embeddings) == 0:
                logger.warning('No embedding respect filters {}'.format(kwargs))
                filtered_embeddings = embeddings
        np_embeddings = embeddings_to_np(filtered_embeddings)
    else:
        np_embeddings = embeddings_to_np(embeddings, force_np = False)
        if len(np_embeddings.shape) == 1: np_embeddings = np_embeddings[None]
    
    if isinstance(mode, int):
        return np_embeddings[mode]
    elif mode in ('mean', 'avg', 'average'):
        return ops.mean(np_embeddings, axis = 0)
    elif mode == 'random':
        idx = random.randrange(len(np_embeddings))
        if logger.isEnabledFor(logging.DEBUG): logger.debug('Selected embedding : {}'.format(idx))
        return np_embeddings[idx]
    elif callable(mode):
        return mode(np_embeddings)
    else:
        raise ValueError("Unknown embedding selection mode !\n  Accepted : {}\n  Got : {}".format(
            "(int, callable, 'mean', 'random')", mode
        ))

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

def _get_embeddings_file_ext(filename):
    """ Returns a valid extension for `filename` such that `filename + ext` exists """
    for ext in _embeddings_file_ext:
        if os.path.exists(filename + ext): return ext
    return None
