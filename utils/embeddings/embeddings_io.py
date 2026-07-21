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

from ..keras import ops
from ..sequence_utils import pad_batch
from ..generic_utils import is_dataframe
from .embeddings_processing import aggregate_embeddings
from ..file_utils import load_data, dump_data, path_to_unix, remove_path_prefix

logger = logging.getLogger(__name__)

_embeddings_file_ext    = {'.csv', '.npy', '.pkl', '.pdpkl', '.embeddings.h5', '.h5'}
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
            return embeddings_to_np(load_embeddings(embeddings), col = col, dtype = dtype)
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
    elif not filename.endswith(tuple(_embeddings_file_ext)):
        raise ValueError('Unsupported embeddings extension !\n  Accepted : {}\n  Got : {}'.format(
            _embeddings_file_ext, filename
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
                from ..datasets import get_dataset_dir
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

def _get_embeddings_file_ext(filename):
    """ Returns a valid extension for `filename` such that `filename + ext` exists """
    for ext in _embeddings_file_ext:
        if os.path.exists(filename + ext): return ext
    return None