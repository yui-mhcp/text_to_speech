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

from ..keras import TensorSpec, ops, graph_compile
from ..generic_utils import is_dataframe, filter_df, aggregate_df

logger = logging.getLogger(__name__)

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
    from .embeddings_io import embeddings_to_np
    
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