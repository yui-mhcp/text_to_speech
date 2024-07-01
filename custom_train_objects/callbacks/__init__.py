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
import keras

from utils import import_objects, dispatch_wrapper, get_object, print_objects

_callbacks = import_objects(
    [__package__.replace('.', os.path.sep), keras.callbacks],
    classes     = keras.callbacks.Callback,
    exclude     = ('Callback', 'CallbackList', 'History', 'ProgBarLogger')
)
globals().update(_callbacks)

@dispatch_wrapper(_callbacks, 'callback')
def get_callbacks(callback = None, * args, ** kwargs):
    return get_object(
        _callbacks, callback, * args, ** kwargs, print_name = 'callbacks',
        types = keras.callbacks.Callback
    )


def print_callbacks():
    print_objects(_callbacks, 'callbacks')

