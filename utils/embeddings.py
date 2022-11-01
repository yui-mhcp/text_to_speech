
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
import random
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.file_utils import load_data, dump_data
from utils.pandas_utils import filter_df, aggregate_df
from utils.sequence_utils import pad_batch

logger = logging.getLogger(__name__)

_allowed_embeddings_ext = ('.csv', '.npy', '.pkl', '.pdpkl')
_default_embedding_ext  = '.pdpkl'
_embedding_filename = 'embeddings_{}'

_accepted_modes     = ('random', 'mean', 'average', 'avg', 'int', 'callable')

def get_embedding_file_ext(filename):
    """ Return the extension for a supported embedding extension based on a filename without ext """
    for ext in _allowed_embeddings_ext:
        if os.path.exists(filename + ext): return ext
    return None

def embeddings_to_np(embeddings, col = 'embedding'):
    """ Return a numpy matrix of embeddings from a dataframe column """
    # if it is a string representation of a numpy matrix
    if isinstance(embeddings, str):
        if embeddings.startswith('['):
            # if it is a 2D-matrix
            if embeddings.startswith('[['):
                logger.debug(embeddings[1 : -1].split(']')[0][1:])
                return np.array([
                    np.fromstring(xi[1 :]) for xi in embeddings[1:-1].split(']')
                ])
            # if it is a 1D-vector
            return np.fromstring(embeddings[1:-1], dtype = float, sep = '\t').astype(np.float32)
        elif not os.path.isfile(embeddings):
            raise ValueError("You must provide an existing embedding file (got {})".format(embeddings))
        return embeddings_to_np(load_embeddings(embeddings))
    elif isinstance(embeddings, np.ndarray): return embeddings
    elif isinstance(embeddings, tf.Tensor): return embeddings.numpy()
    elif isinstance(embeddings, pd.DataFrame):
        embeddings = [e for e in embeddings[col].values]
        if len(embeddings[0].shape) == 1: return np.array(embeddings)
        return pad_batch(embeddings)
    else:
        raise ValueError("Invalid type of embeddings : {}\n{}".format(type(embeddings), embeddings))

def aggregate_embeddings(dataset, column = 'id', aggregation_name = 'speaker_embedding', mode = 0):
    if aggregation_name in dataset.columns:
        dataset.pop(aggregation_name)
    
    if column not in dataset.columns:
        dataset[column] = 'id'
    if column != 'id':
        for idx, row in dataset.iterrows():
            if column not in row or row[column] is np.nan:
                dataset.at[idx, column] = row['id']
    
    kw = {aggregation_name : mode}
    return aggregate_df(
        dataset, column, 'embedding', merge = True, ** kw
    )

def save_embeddings(directory, embeddings, embedding_name = _embedding_filename):
    """ Save embedding to the given path """
    if os.path.splitext(directory)[1] in _allowed_embeddings_ext:
        embedding_file = directory
    else:
        if not directory.endswith('embeddings'):
            directory = os.path.join(directory, 'embeddings')
        os.makedirs(directory, exist_ok = True)
        embedding_file = os.path.join(directory, embedding_name)
    
    if '{}' in embedding_file:
        embedding_dim   = embeddings_to_np(embeddings).shape[-1]
        embedding_file  = embedding_file.format(embedding_dim)
    
    ext = os.path.splitext(embedding_file)[1]
    if not ext: embedding_file += _default_embedding_ext
    elif ext not in _allowed_embeddings_ext:
        raise ValueError('Unknown embedding extension !\n  Accepted : {}\n  Got : {}'.format(
            _allowed_embeddings_ext, embedding_file
        ))
    
    if embedding_file.endswith('npy'): embeddings = embeddings_to_np(embeddings)
    dump_data(embedding_file, embeddings)
    
    return embedding_file
    

def load_embedding(directory,
                   embedding_name   = _embedding_filename,
                   embedding_dim    = None,
                   dataset          = None,
                   
                   aggregate_on     = 'id',
                   aggregate_mode   = 0,
                   aggregate_name   = 'speaker_embedding',
                   ** kwargs
                  ):
    """
        Load embeddings from file (csv / npy / pkl) and create an aggregation version
        
        Arguments : 
            - directory     : directory in which embeddings are stored (must be 'embeddings' or have a sub-directory named 'embeddings')
                It can also be the embeddings' filename
            - embedding_name    : the embeddings' filename (can contains '{}' which will be formatted by `embedding_dim`)
            - embedding_dim : dimension of the embedding (will format `filename` if required)
            
            - dataset       : the dataset on which to merge embeddings
            - aggregate_on  : the column to aggregate on
            - aggregate_mode    : the mode for the aggregation
            - aggregate_name    : the name for the aggregated embeddings' column (default to `speaker_embedding` for retro-compatibility)
        Return :
            - embeddings or dataset merged with embeddings (merge is done on columns that are both in `dataset` and `embeddings`)
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
    
    if not os.path.exists(emb_file):
        ext = get_embedding_file_ext(emb_file)
        
        if ext is None:
            raise ValueError("Embedding file {} does not exist !".format(emb_file))
        
        emb_file += ext
    
    embeddings  = load_data(emb_file)
    if not isinstance(embeddings, pd.DataFrame): return embeddings

    for embedding_col_name in [col for col in embeddings.columns if col.endswith('embedding')]:
        if isinstance(embeddings.loc[0, embedding_col_name], np.ndarray): continue
        embeddings[embedding_col_name] = embeddings[embedding_col_name].apply(
            lambda x: embeddings_to_np(x)
        )
    
    if aggregate_on is not None and aggregate_on in embeddings.columns:
        embeddings = aggregate_embeddings(
            embeddings, aggregate_on, aggregation_name = aggregate_name, mode = aggregate_mode
        )
    
    if dataset is None: return embeddings
    
    intersect = list(set(embeddings.columns).intersection(set(dataset.columns)))
    
    for col in intersect:
        if embeddings[col].dtype != dataset[col].dtype:
            embeddings[col] = embeddings[col].apply(lambda i: str(i))
    
    logger.debug('Merging embeddings with dataset on columns {}'.format(intersect))
    dataset = pd.merge(dataset, embeddings, on = intersect)

    dataset = dataset.dropna(
        axis = 'index', subset = ['embedding', 'speaker_embedding']
    )
    
    return dataset

def pad_dataset_embedding(dataset, columns = None):
    """
        Pad columns' embeddings to have matrices with same dimensions and add a n_{col} column with the current number of embeddings (to allow to retrieve the original embeddings' matrix)
    """
    if columns is None: columns = [c for c in dataset.columns if c.endswith('_embedding')]
    if not isinstance(columns, (list, tuple)): columns = [columns]
    
    for col in columns:
        if len(dataset.iloc[0][col].shape) == 1: continue
        
        dataset['n_{}'.format(col)] = dataset[col].apply(lambda e: len(e))
        dataset[col] = list(embeddings_to_np(dataset, col = col))
    
    return dataset

def select_embedding(embeddings, mode = 'random', ** kwargs):
    """
        Return a single embedding (np.ndarray with rank 1) from a collection of embeddings 
        
        Arguments : 
            - embeddings    : pd.DataFrame with 'embedding' col or np.ndarray (2D matrix)
            - mode      : selection mode (int / 'avg' / 'mean' / 'average' / 'random' / callable)
        Return :
            - embedding : 1D ndarray
    """
    if isinstance(embeddings, pd.DataFrame):
        filtered_embeddings = filter_df(embeddings, ** kwargs)
        if len(filtered_embeddings) == 0:
            logger.warning('No embedding respects filters {}'.format(kwargs))
            filtered_embeddings = embeddings
        np_embeddings = embeddings_to_np(filtered_embeddings)
    else:
        np_embeddings = embeddings
    
    if isinstance(mode, int):
        return np_embeddings[mode]
    elif mode in ('mean', 'avg', 'average'):
        return np.mean(np_embeddings, axis = 0)
    elif mode == 'random':
        return np_embeddings[random.randrange(len(np_embeddings))]
    elif callable(mode):
        return mode(np_embeddings)
    else:
        raise ValueError("Mode to select embedding unknown\n  Get : {}\n  Accepted : {}".format(
            mode, _accepted_modes
        ))
    
def compute_mean_embeddings(embeddings, ids):
    """
        Compute the mean embeddings for each id
        Arguments :
            - embeddings    : 2D matrix of embeddings
            - ids   : array of ids where embeddings[i] has ids[i]
        Return :
            - (unique_ids, centroids)
                - unique_ids    : vector of unique ids
                - centroids     : centroids[i] is the centroid associated to embeddings of ids[i]
    """
    uniques = tf.unique(ids)[0]
    return uniques, tf.concat([
        tf.reduce_mean(
            tf.gather(embeddings, tf.where(ids == unique_id)), axis = 0
        )
        for unique_id in uniques
    ], axis = 0)

def visualize_embeddings(embeddings,
                         metadata,
                         log_dir    = 'embedding_visualization',
                         images     = None,
                         image_size = 128,
                         label_col  = 'id',
                         metadata_filename = 'metadata.tsv'
                        ):
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
    