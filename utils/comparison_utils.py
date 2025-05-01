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
import inspect
import numpy as np

def is_equal(target, value, ** kwargs):
    try:
        _compare(target, value, ** kwargs)
        return True, 'Value are equals !'
    except AssertionError as e:
        return False, str(e)

def is_diff(target, value, ** kwargs):
    eq, msg = is_equal(target, value, ** kwargs)
    if eq: msg = 'Value are equals but should not be'
    return not eq, msg
    
def _compare(target, value, ** kwargs):
    """ Compare 2 items and raises an AssertionError if their value differ """
    target  = _maybe_convert_to_numpy(target)
    value   = _maybe_convert_to_numpy(value)
    
    for t, compare_fn in _comparisons.items():
        if isinstance(t, (type, tuple)) and isinstance(target, t):
            return compare_fn(target, value, ** kwargs)
        elif not isinstance(t, (type, tuple)) and t(target):
            return compare_fn(target, value, ** kwargs)

    if hasattr(target, 'get_config'):
        _compare(target.get_config(), value.get_config(), ** kwargs)
    else:
        _compare_primitive(target, value, ** kwargs)
    
def _compare_primitive(target, value, max_err = 0., ** kwargs):
    if isinstance(value, (float, np.floating)):
        assert abs(target - value) <= max_err, 'Values differ : {}'.format(abs(value - target))
    else:
        assert target == value, "Target ({}) != value ({})".format(target, value)

def _compare_list(target, value, ** kwargs):
    """ Compare each item of both iterables (target / value) """
    assert len(target) == len(value), "Target length {} != value length {}".format(
        len(target), len(value)
    )
    
    try:
        if not any(hasattr(v, 'shape') for v in target) and target == value: return
    except Exception as e:
        pass
    
    cmp         = [is_equal(it1, it2, ** kwargs) for it1, it2 in zip(target, value)]
    invalids    = [(i, msg) for i, (eq, msg) in enumerate(cmp) if not eq]

    assert len(invalids) == 0, "Invalid items ({}/{}) :{}{}".format(
        len(invalids), len(target), '\n' if len(invalids) > 1 else ' ',
        '\n'.join(['Item #{} : {}'.format(i, msg) for i, msg in invalids])
    )
    
def _compare_dict(target, value, keys = None, skip_keys = None, skip_missing = False, ** kwargs):
    """
        Compare 2 dict-like
        Arguments :
            - target / value    : the values to compare
            - keys      : the keys to compare
            - skip_keys : the keys to skip
            - skip_missing  : only compares common keys
    """
    if skip_missing:
        if not keys: keys = list(target.keys())
        keys = [k for k in keys if k in value and k in target]
    
    if keys:
        if not isinstance(keys, (list, tuple)): keys = [keys]
        target  = {k : target[k] for k in target if k in keys}
        value   = {k : value[k] for k in value if k in keys}
    
    if skip_keys:
        if not isinstance(skip_keys, (list, tuple)): skip_keys = [skip_keys]
        target  = {k : target[k] for k in target if k not in skip_keys}
        value   = {k : value[k] for k in value if k not in skip_keys}
    
    missing_v_keys  = [k for k in target if k not in value]
    missing_t_keys  = [k for k in value if k not in target]
        
    assert len(missing_v_keys) + len(missing_t_keys) == 0, "Missing keys in value : {}\nAdditionnal keys in value : {}".format(missing_v_keys, missing_t_keys)
    
    cmp         = {k : is_equal(target[k], value[k], ** kwargs) for k in target}
    invalids    = {k : msg for k, (eq, msg) in cmp.items() if not eq}
    
    assert len(invalids) == 0, "Invalid items ({}/{}) :{}{}".format(
        len(invalids), len(target), '\n' if len(invalids) > 1 else ' ',
        '\n'.join(['Key {} : {}'.format(k, msg) for k, msg in invalids.items()])
    )

def _compare_array(target, value, max_err = 1e-6, squeeze = False, normalize = False, ** kwargs):
    """
        Compare 2 arrays with some tolerance (`max_err`) on the error's sum / mean / max / min depending `err_mode`
        `squeeze` allows to skip `1`-dimensions
    """
    value = np.asarray(value)
    if squeeze: target, value = np.squeeze(target), np.squeeze(value)
    
    assert target.shape == value.shape, "Target shape {} != value shape {}".format(target.shape, value.shape)
    
    assert (target.dtype == value.dtype) or (target.dtype in (np.int32, np.int64) and value.dtype in (np.int32, np.int64)), "Target dtype {} != value dtype {}".format(target.dtype, value.dtype)
    
    if target.size == 0: return
    
    if target.dtype in (bool, object):
        assert np.all(target == value), "Vallue differ for target with dtype {} ({} / {} diff, {:.23f} %)".format(
            target.dtype, np.sum(target != value), np.prod(target.shape), np.mean(target != value)
        )
    else:
        if normalize:
            valids = np.isclose(value, target, rtol = max_err)
        else:
            valids = np.isclose(value, target, atol = max_err)
        
        valid = np.all(valids)
        if valid: return
        
        err = np.where(valids, 0., np.abs(target - value))
        
        assert valid, "Values differ ({} / {} diff, {:.3f}%) : max {} - mean {} - min {}".format(
            np.sum(~valids), np.prod(err.shape), np.mean(~valids), np.max(err), np.mean(err), np.min(err)
        )

def _compare_dataframe(target, value, ignore_index = True, ** kwargs):
    """ Compare 2 DataFrames """
    missing_v_cols  = [k for k in target.columns if k not in value.columns]
    missing_t_cols  = [k for k in value.columns if k not in target.columns]
    
    assert len(missing_v_cols) + len(missing_t_cols) == 0, "Missing keys in value : {}\nAdditionnal keys in value : {}".format(missing_v_cols, missing_t_cols)
    
    assert len(target) == len(value), "Target length {} != value length {}".format(
        len(target), len(value)
    )
    
    if not ignore_index:
        diff = np.where(~np.all((target == value).values, axis = -1))[0]
        assert len(diff) == 0, "DataFrames differs ({})\n  Target : \n{}\n  Value : \n{}".format(
            len(diff), target.iloc[diff], value.iloc[diff]
        )
        return
    
    invalids = []
    for idx, row in value.iterrows():
        if not np.any(np.all((row == target).values, axis = -1)):
            invalids.append(idx)
    
    assert len(invalids) == 0, "Some rows are not in target ({}) :\n{}".format(
        len(invalids), value.iloc[invalids]
    )

def _maybe_convert_to_numpy(x):
    if not hasattr(x, 'device'):
        return x
    elif hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    else:
        return np.asarray(x)

_comparisons    = {
    (str, int, float, bool, set)    : _compare_primitive,
    (list, tuple)   : _compare_list,
    dict   : _compare_dict,
    np.ndarray  : _compare_array,
    lambda v: hasattr(v, 'columns') : _compare_dataframe
}