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
import logging
import importlib

from .callback import Callback
from .file_saver import FileSaver, JSONSaver

logger = logging.getLogger(__name__)

for module in os.listdir(__package__.replace('.', os.path.sep)):
    if module.startswith(('.', '_')) or '_old' in module: continue
    module = importlib.import_module(__package__ + '.' + module[:-3])
    
    globals().update({
        k : v for k, v in vars(module).items()
        if isinstance(v, type) and issubclass(v, Callback)
    })
    
def apply_callbacks(callbacks, infos, output, save = True, ** kwargs):
    if not callbacks: return
    
    entry = None
    for callback in callbacks:
        if isinstance(callback, FileSaver) and not save:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('- Skip {}'.format(callback))
            continue

        try:
            res = callback(infos, output, ** kwargs)
            if isinstance(callback, JSONSaver): entry = res
        except Exception as e:
            logger.error('- An exception occured while calling {} : {}'.format(callback, e))
    return entry
