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
import logging

from .callback import Callback
from utils.generic_utils import import_objects

logger = logging.getLogger(__name__)

globals().update(import_objects(
    __package__.replace('.', os.path.sep), classes = Callback
))


def apply_callbacks(results, index, callbacks, ** kwargs):
    """
        Apply a list of `callbacks` on `results[index :]`, until `results[i] is None`
        
        Arguments :
            - results   : a `list` of `tuple` in the form `(stored_infos, output)`
                          - `stored_infos` is a `dict` containing information (re)stored results
                          - `output` is a `dict` representing the output of the model
                          If `output` is None, the data is supposed to be restored from the
                          `json` mapping file. In this case, all `FileSaver` callbacks are skipped
            - index     : the index to start iterate from
            - callbacks : a list of `callable`  with the following signature :
                          `callback(infos = infos, output = output, ** kwargs) -> bool`
                          The callback has to return `True` if it had a side effect on `infos`
                          This information is used to only apply `JSonSaver` if any callback
                          had a side effect on the stored information
            - kwargs    : fowarded to each `callback`
        Return :
            - last_index    : the latest index on which callbacks have been applied
    """
    if not callbacks: return
    
    side_effect = False
    while index < len(results) and results[index] is not None:
        store, output = results[index]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Result #{} : start applying callbacks'.format(index))
            if output is None:
                logger.debug('- The result was restored without prediction')
        if output is None: output = {}
        
        stop_next = index + 1 == len(results) or results[index + 1] is None
        for callback in callbacks:
            if isinstance(callback, FileSaver) and not output:
                # The results was basically restored and not overwritten, nothing to save
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('- skipping {}'.format(index, callback))
                continue
            elif isinstance(callback, JSonSaver):
                # The information to store has not been affected : no need to save them
                side_effect = callback.update_data(store, output) or side_effect
                if not stop_next or not side_effect:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('- Skipping JSonSaver'.format(index))
                    continue
            
            try:
                side_effect = callback(infos = store, output = output, ** kwargs) or side_effect
            except Exception as e:
                logger.error('- An exception occured while calling {} : {}'.format(
                    callback, e
                ))
        
        index += 1
    
    return index


def apply_callbacks_raw(data, callbacks, ** kwargs):
    """
        Apply a list of `callbacks` on `data`
        
        Arguments :
            - data  : `dict` containing the model output information
            - callbacks : a list of `callable`  with the following signature :
                          `callback(infos = infos, output = output, ** kwargs) -> bool`
                          The callback has to return `True` if it had a side effect on `infos`
                          This information is used to only apply `JSonSaver` if any callback
                          had a side effect on the stored information
            - kwargs    : fowarded to each `callback`
        
        Note : the major difference with `apply_callbacks` is that `data` is supposed to be a new raw data, which was not restored. This enforces all callbacks to be called, while creating new empty storage information.
        This version is therefore more suitable for raw streaming data
    """
    store, output = {}, data
    for callback in callbacks:
        if isinstance(callback, JSonSaver):
            callback.update_data(store, output)

        try:
            callback(infos = store, output = output, ** kwargs) or side_effect
        except Exception as e:
            logger.error('- An exception occured while calling {} : {}'.format(
                callback, e
            ))
        
