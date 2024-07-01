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

import logging
import functools
import collections
import numpy as np
import pandas as pd
import keras.ops as K

from utils.sequence_utils import pad_batch
from utils.distance.distance_method import _str_distance_methods, distance_method_wrapper

logger  = logging.getLogger(__name__)

def text_distance_method_wrapper(fn = None, ** kw):
    def wrapper(fn):
        @functools.wraps(fn)
        def inner(y_true, y_pred, as_matrix = False, ** kwargs):
            y_true  = _normalize(y_true)
            y_pred  = _normalize(y_pred)

            if _is_nested_list(y_true) or _is_nested_list(y_pred):
                assert len(y_true) == len(y_pred), "len(y_true) {} != len(y_pred) {}".format(
                    len(y_true), len(y_pred)
                )
                if not _is_nested_list(y_true): y_true = [[yi] for yi in y_true]
                if not _is_nested_list(y_pred): y_pred = [[yi] for yi in y_pred]

                return pad_batch([
                    inner(y_true_i, y_pred_i, as_matrix = True, ** kwargs)
                    for y_true_i, y_pred_i in zip(y_true, y_pred)
                ], pad_value = -1., dtype = np.float32)
            elif isinstance(y_true, (list, tuple)) and isinstance(y_pred, (list, tuple)):
                if not as_matrix:
                    assert len(y_true) == len(y_pred), "len(y_true) {} != len(y_pred) {}".format(
                        len(y_true), len(y_pred)
                    )
                    return np.array([
                        fn(y_true_i, y_pred_i, ** kwargs)
                        for y_true_i, y_pred_i in zip(y_true, y_pred)
                    ])
                return np.array([
                    inner(y_true_i, y_pred, ** kwargs) for y_true_i in y_true
                ])
            elif isinstance(y_true, (list, tuple)):
                return np.array([
                    fn(y_true_i, y_pred, ** kwargs) for y_true_i in y_true
                ])
            elif isinstance(y_pred, (list, tuple)):
                return np.array([
                    fn(y_true, y_pred_i, ** kwargs) for y_pred_i in y_pred
                ])
            
            return fn(y_true, y_pred, ** kwargs)
        
        key = kw.get('name', fn.__name__.split('_')[0])
        
        wrapped_fn  = distance_method_wrapper(inner, expand = False, ** kw)
        _str_distance_methods[key]  = wrapped_fn
        
        return wrapped_fn
    return wrapper if fn is None else wrapper(fn)

@text_distance_method_wrapper(is_similarity = True)
def edit_distance(hypothesis,
                  truth,
                  partial   = False,
                  deletion_cost     = {},
                  insertion_cost    = {}, 
                  replacement_cost  = {},
                  
                  default_del_cost  = 1,
                  default_insert_cost   = 1,
                  default_replace_cost  = 1,
                  
                  normalize     = True,
                  return_matrix = False,
                  verbose   = False,
                  ** kwargs
                 ):
    """
        Compute a weighted Levenstein distance
        
        Arguments :
            - hypothesis    : the predicted value   (iterable)
            - truth         : the true value        (iterable)
            - partial       : whether to make partial alignment or not
            - insertion_cost    : weights to insert a new symbol
            - replacement_cost  : weights to replace a symbol (a --> b) but 
            is not in both sens (a --> b != b --> a) so you have to specify weights in both sens
            - normalize     : whether to normalize on truth length or not
            - return_matrix : whether to return the matrix or not
            - verbose       : whether to show costs for path or not
        Return :
            - distance if not return_matrix else (distance, matrix)
                - distance  : scalar, the Levenstein distance between `hypothesis` and truth `truth`
                - matrix    : np.ndarray of shape (N, M) where N is the length of truth and M the length of hypothesis. 
        
        Note : if `partial` is True, the distance is the minimal distance
        Note 2 : `distance` (without normalization) corresponds to the "number of errors" between `hypothesis` and `truth`. It means that the start of the best alignment (if partial) is `np.argmin(matrix[-1, 1:]) - len(truth) - distance`
    """
    matrix = np.zeros((len(hypothesis) + 1, len(truth) + 1))
    # Deletion cost
    deletion_costs = np.array([0] + [deletion_cost.get(h, default_del_cost) for h in hypothesis])
    insertion_costs = np.array([insertion_cost.get(t, default_insert_cost) for t in truth])
    
    matrix[:, 0] = np.cumsum(deletion_costs)
    # Insertion cost
    if not partial:
        matrix[0, :] = np.cumsum([0] + [insertion_cost.get(t, default_insert_cost) for t in truth])

    truth_array = truth if not isinstance(truth, str) else np.array(list(truth))
    for i in range(1, len(hypothesis) + 1):
        deletions = matrix[i-1, 1:] + deletion_costs[i]
        
        matches   = np.array([
            replacement_cost.get(hypothesis[i-1], {}).get(t, default_replace_cost) for t in truth
        ])
        matches   = matrix[i-1, :-1] + matches * (truth_array != hypothesis[i-1])
        
        min_costs = np.minimum(deletions, matches)
        for j in range(1, len(truth) + 1):
            insertion   = matrix[i, j-1] + insertion_costs[j-1]

            matrix[i, j] = min(min_costs[j-1], insertion)
    
    if verbose:
        columns = [''] + [str(v) for v in truth]
        index = [''] + [str(v) for v in hypothesis]
        logger.info(pd.DataFrame(matrix, columns = columns, index = index))
    
    distance = matrix[-1, -1] if not partial else np.min(matrix[-1, 1:])
    if normalize:
        distance = distance / len(truth) if not partial else distance / len(hypothesis)
    
    return distance if not return_matrix else (distance, matrix)

@text_distance_method_wrapper(is_similarity = False)
def hamming_distance(hypothesis, truth, replacement_matrix = {}, normalize = True,
                     ** kwargs):
    """
        Compute a weighted hamming distance
        
        Arguments : 
            - hypothesis    : the predicted value   (iterable)
            - truth         : the true value        (iterable)
            - replacement_matrix    : weights to replace element 1 to 2 (from hypothesis to truth). Note that this is not in 2 sens so a --> b != b --> a
            - normalize     : whether to normalize on truth length or not
        Return : distance between hypothesis and truth (-1 if they have different length)
    """
    if len(hypothesis) != len(truth): return -1
    distance = sum([
        replacement_matrix.get(c1, {}).get(c2, 1)
        for c1, c2 in zip(hypothesis, truth) if c1 != c2
    ])
    if normalize: distance = distance / len(truth)
    return distance


@text_distance_method_wrapper(is_similarity = True, name = 'exact_match')
def exact_match(y_true, y_pred, ** _):
    """ Returns the Exact-Match between 2 sentences (i.e. 1 if they are equals, 0 otherwise) """
    return int(y_true == y_pred)

@text_distance_method_wrapper(is_similarity = True, name = 'text_f1')
def text_f1(y_true, y_pred, normalize = True, exclude = None, ** kwargs):
    """
        Compute F1-score
        
        Arguments :
            - y_true    : ground truth (target)
            - y_pred    : prediction (hypothesis)
            - normalize : whether to normalize or not (lowercase + remove spaces)
            - exclude   : list of token to exclude (not take into account)
        Return :
            - if `y_true` and `y_pred` are str : [EM, F1, precision, recall]
            - if `y_true` or `y_pred` is a list (not nested) :
                - if `as_matrix` is False : [n, 4] (n = len(y_true) = len(y_pred))
                - else : [len(y_true), len(y_pred), 4]
            - if `y_true` or `y_pred` is a nested list : np.ndarray of shape [N, n_true, n_pred, 4]
                - N      = len(y_true) = len(y_pred) (can be interpreted as the batch_size)
                - n_true = max(len(y_true_i))
                - n_pred = max(len(y_pred_i))
                
    """
    if exclude: exclude = _normalize(exclude)
    
    if normalize:
        y_true = _normalize_text_f1(y_true, exclude)
        y_pred = _normalize_text_f1(y_pred, exclude)
    elif exclude:
        from utils.text.cleaners import collapse_whitespace, remove_tokens
        
        y_true = collapse_whitespace(remove_tokens(y_true, exclude))
        y_pred = collapse_whitespace(remove_tokens(y_pred, exclude))

    true_tokens = y_true.split()
    pred_tokens = y_pred.split()
    
    common  = collections.Counter(true_tokens) & collections.Counter(pred_tokens)
    nb_same = sum(common.values())

    em = exact_match(y_true, y_pred)

    if len(true_tokens) == 0 or len(pred_tokens) == 0:
        f1 = int(true_tokens == pred_tokens)
        return em, f1, f1, f1
    elif nb_same == 0:
        return 0, 0, 0, 0
    
    precision = 1. * nb_same / len(pred_tokens)
    recall    = 1. * nb_same / len(true_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return em, f1, precision, recall

def _is_nested_list(data):
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple)):
        return True
    return False

def _normalize(data):
    if K.is_tensor(data):   data = K.convert_to_numpy(data)
    if isinstance(data, bytes):         data = data.decode()
    if isinstance(data, np.ndarray):    data = data.tolist()
    if isinstance(data, (list, tuple)) and isinstance(data[0], int):
        data = ' '.join([str(d) for d in data])
    return data

def _normalize_text_f1(text, exclude = []):
    from utils.text.cleaners import collapse_whitespace, remove_tokens, remove_punctuation, lowercase
    return collapse_whitespace(remove_tokens(remove_punctuation(lowercase(text)), exclude)).strip()
