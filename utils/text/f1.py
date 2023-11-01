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

import collections
import numpy as np
import tensorflow as tf

from utils.sequence_utils import pad_batch
from utils.text.cleaners import collapse_whitespace, remove_tokens, remove_punctuation, lowercase

def _normalize_text_f1(text, exclude = []):
    return collapse_whitespace(remove_tokens(remove_punctuation(lowercase(text)), exclude)).strip()

def exact_match(y_true, y_pred):
    return int(y_true == y_pred)

def f1_score(y_true, y_pred, normalize = True, exclude = None, as_matrix = False):
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
                - N = len(y_true) = len(y_pred)
                - n1 = max(len(y_true_i))
                - n2 = max(len(y_pred_i))
                
    """
    def _is_nested_list(data):
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple)):
            return True
        return False
    
    def _normalize(data):
        if isinstance(data, tf.Tensor): data = data.numpy()
        if isinstance(data, bytes):     data = data.decode('utf-8')
        if isinstance(data, np.ndarray):    data = data.tolist()
        if isinstance(data, (list, tuple)) and isinstance(data[0], int):
            data = ' '.join([str(d) for d in data])
        return data
    
    y_true  = _normalize(y_true)
    y_pred  = _normalize(y_pred)
    
    if _is_nested_list(y_true) or _is_nested_list(y_pred):
        if not _is_nested_list(y_true): y_true = [[yi] for yi in y_true]
        if not _is_nested_list(y_pred): y_pred = [[yi] for yi in y_pred]
    
        return pad_batch([
            f1_score(y_true_i, y_pred_i, normalize = normalize, exclude = exclude, as_matrix = True)
            for y_true_i, y_pred_i in zip(y_true, y_pred)
        ], pad_value = -1., dtype = np.float32)
    elif isinstance(y_true, (list, tuple)) and isinstance(y_pred, (list, tuple)):
        if not as_matrix:
            assert len(y_true) == len(y_pred), "Lengths are {} and {}".format(len(y_true), len(y_pred))
            return np.array([
                f1_score(y_true_i, y_pred_i, normalize = normalize, exclude = exclude)
                for y_true_i, y_pred_i in zip(y_true, y_pred)
            ])
        return np.array([
            f1_score(y_true_i, y_pred, normalize = normalize, exclude = exclude) for y_true_i in y_true
        ])
    elif isinstance(y_true, (list, tuple)):
        return np.array([
            f1_score(y_true_i, y_pred, normalize = normalize, exclude = exclude) for y_true_i in y_true
        ])
    elif isinstance(y_pred, (list, tuple)):
        return np.array([
            f1_score(y_true, y_pred_i, normalize = normalize, exclude = exclude) for y_pred_i in y_pred
        ])
    
    if exclude: exclude = _normalize(exclude)
    
    if normalize:
        y_true = _normalize_text_f1(y_true, exclude)
        y_pred = _normalize_text_f1(y_pred, exclude)
    elif exclude:
        y_true = collapse_whitespace(remove_tokens(y_true, exclude))
        y_pred = collapse_whitespace(remove_tokens(y_pred, exclude))
    
    true_tokens = y_true.split()
    pred_tokens = y_pred.split()
    
    common = collections.Counter(true_tokens) & collections.Counter(pred_tokens)
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
