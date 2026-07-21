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

import logging
import traceback

from utils.generic_utils import import_submodules
from .callback import Callback
from .file_saver import FileSaver, JSONSaver

_callbacks = {}
for module in import_submodules(__package__):
    _callbacks.update({
        k : v for k, v in vars(module).items()
        if isinstance(v, type) and issubclass(v, Callback)
    })
globals().update(_callbacks)
del _callbacks

logger = logging.getLogger(__name__)

def apply_callbacks(callbacks, infos, output, save = True, ** kwargs):
    """
        Apply a `list` of `Callback` on the provided output.
        
        Arguments :
            - callbacks : the list of `Callback` to apply
            - infos     : the already stored information (`dict`)
                          - if the data is new, this should be empty
                          - if the data should not be overwritten, `output` should be `None`
                          - if the data should be overwritten, the files in `infos` will be overwritten
            - output    : the output information (`dict`)
            - save      : whether to save the output or not
                          If `False`, callbacks with `saves_to_disk` are skipped
            - kwargs    : forwarded to all callbacks
        Return :
            - entry     : the `provides_entry` callback's return (e.g. `JSONSaver`), or None
    """
    if not callbacks: return

    entry = None
    for callback in callbacks:
        if callback.saves_to_disk and not save:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('- Skip {}'.format(callback))
            continue

        try:
            res = callback(infos, output, ** kwargs)
            if callback.provides_entry: entry = res
        except Exception as e:
            logger.error('An exception occured while calling {} :\n{}'.format(
                callback, traceback.format_exc()
            ))
    return entry