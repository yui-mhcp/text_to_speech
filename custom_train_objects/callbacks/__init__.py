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
import keras
import importlib

for module in [keras.callbacks] + os.listdir(__package__.replace('.', os.path.sep)):
    if isinstance(module, str):
        if module.startswith(('.', '_')) or '_old' in module: continue
        module = importlib.import_module(__package__ + '.' + module[:-3])
    
    globals().update({
        k : v for k, v in vars(module).items()
        if isinstance(v, type) and issubclass(v, keras.callbacks.Callback)
    })

_callbacks = {
    k.lower() : v for k, v in globals().items()
    if isinstance(v, type) and issubclass(v, keras.callbacks.Callback)
}


def get_callbacks(callback = None, ** kwargs):
    if not callback: return None
    elif isinstance(callback, (list, tuple)):
        return [get_callbacks(c, ** kwargs) for c in callback]
    elif isinstance(callback, keras.callbacks.Callback):
        return callback
    elif isinstance(callback, dict):
        config = callback['config'] if 'config' in callback else {
            k : v for k, v in callback.items() if 'name' not in k
        }
        kwargs.update(config)
        callback = callback['name' if 'name' in callback else 'class_name']

    return _callbacks[callback.lower()](** kwargs)
