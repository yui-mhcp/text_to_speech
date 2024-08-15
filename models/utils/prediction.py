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

from loggers import timer
from utils import get_entry, expand_path, path_to_unix, convert_to_list

logger = logging.getLogger(__name__)

def prepare_prediction_results(data,
                               storage,
                               primary_key,
                               *,
                               
                               rank = None,
                               filters  = None,
                               expand_files = None,
                               normalize_entry  = None,
                               
                               overwrite    = False,
                               required_keys    = (),
                               
                               ** kwargs
                              ):
    """
        Prepare `data` for prediction by filtering the already predicted ones
        
        Arguments :
            - data  : `list` or `ndarray / Tensor` of data to predict
            - predicted : `dict` containing the already predicted data
            - primary_key   : the primary key used to identify data in `predicted`
            - overwrite     : whether to overwrite already predicted data
            - required_keys : the expected keys in `predicted[data_i]` to not overwrite it
        Return : (results, inputs, indexes, files, duplicates)
            - results   : a `list` of tuple `(restored, output)` containing possibly restored data (in this case, `output` is None) or `None` if the data should be predicted
            - inputs    : `list / ndarray / Tensor` of data to predict
            - indexes   : the respective indexes in results for each data (i.e., `results[indices[i]] = model(inputs[indices[i]])`)
            - keys      : the key associates to each input/indice (may be `None`)
            - duplicates    : a `dict` containing `{key : list_of_indices}`
    """
    data = convert_to_list(data, rank = rank)

    if not isinstance(data, list):
        return [None] * len(data), data, list(range(len(data))), [None] * len(data), {}, []
    
    if 'filename' in primary_key:
        if expand_files is None:    expand_files = True
        if normalize_entry is None: normalize_entry = path_to_unix

    results, inputs, indexes, entries, duplicates = [], [], [], [], {}
    filtered = {} if isinstance(filters, dict) else []
    for inp in data:
        _add_data(
            inp,
            storage,
            
            primary_key = primary_key,
            
            filters     = filters,
            expand_files    = expand_files,
            normalize_entry = normalize_entry,

            _results    = results,
            _inputs     = inputs,
            _indexes    = indexes,
            _entries    = entries,
            _duplicates = duplicates,
            _filtered   = filtered,
            
            overwrite   = overwrite,
            required_keys   = required_keys
        )
    
    return results, inputs, indexes, entries, duplicates, filtered

def should_predict(storage, entry, overwrite = False, required_keys = ()):
    """
        Return whether `entry` should be predicted or not.
        Return `False` only if `overwrite == False` and all keys in `required_keys` are in `storage[entry]`

        Arguments :
            - storage   : `dict` containing the stored information of already predicted data
            - entry     : the new key to (maybe) predict
            - overwrite : whether to overwrite or not
            - required_keys : list of keys that must be in `storage[entry]` to be valid
        Return :
            - should_predict    : `bool`, whether the new entry should be predicted or not
    """
    if overwrite or not isinstance(entry, str) or entry not in storage: return True
    infos = storage[entry]
    return any(k not in infos for k in required_keys)

@timer
def _add_data(data,
              storage,
              primary_key,
              
              _results,
              _inputs,
              _indexes,
              _entries,
              _duplicates,
              _filtered,
              
              normalize_entry   = None,
              expand_files  = False,
              filters   = None,
              entry = None,
              
              ** kwargs
             ):
    if entry is None: entry = get_entry(data, keys = primary_key)
    
    if entry:
        if normalize_entry is not None: entry = normalize_entry(entry)
        
        if expand_files:
            entry = expand_path(entry)
            if isinstance(entry, list):
                for file in entry:
                    if isinstance(data, dict):
                        data = data.copy()
                        data[primary_key] = entry
                    else:
                        data = file
                    
                    _add_data(
                        data,
                        storage,
                        primary_key = primary_key,
                        
                        filters = filters,
                        expand_files    = False,
                        normalize_entry = normalize_entry,

                        _results    = _results,
                        _inputs     = _inputs,
                        _indexes    = _indexes,
                        _entries    = _entries,
                        _duplicates = _duplicates,
                        _filtered   = _filtered,

                        entry   = file,
                        ** kwargs
                    )
                return
            elif isinstance(data, str):
                data = entry
            
        
        if filters is not None:
            if isinstance(filters, dict):
                for k, f in filters.items():
                    if f(entry):
                        _filtered.setdefault(k, []).append(entry)
                        return
            
            elif filters(entry):
                _filtered.append(entry)
                return
                
        if not should_predict(storage, entry, ** kwargs):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('The file {} has already been predicted'.format(file))
            _results.append((storage[entry], None))
            return
        
        _duplicates.setdefault(entry, []).append(len(_results))
        if len(_duplicates[entry]) > 1:
            _results.append(None)
            return

    _inputs.append(data)
    _entries.append(entry)
    _indexes.append(len(_results))
    _results.append(None)
