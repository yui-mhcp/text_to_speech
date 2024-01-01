# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
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
import glob
import tensorflow as tf

try:
    from keras.losses import LossFunctionWrapper
except:
    from keras.src.losses import LossFunctionWrapper

from utils.generic_utils import import_objects, get_object, print_objects, is_function

def get_loss(loss_name, * args, ** kwargs):
    global _losses
    if isinstance(loss_name, dict) and 'class_name' in loss_name:
        return tf.keras.losses.deserialize(loss_name, _losses)

    if loss_name == 'LossFunctionWrapper': loss_name = kwargs.pop('fn')
    return get_object(
        _losses, loss_name, * args, ** kwargs, types = (type, tf.keras.losses.Loss),
        err = True, print_name = 'loss', function_wrapper = LossFunctionWrapper
    )

def print_losses():
    print_objects(_losses, 'losses')

def add_loss(loss, name = None):
    if name is None: name = loss.__name__ if is_vunction(loss) else loss.__class__.__name__
    
    _losses[name] = loss
    

def _is_class_or_callable(name, val):
    return isinstance(val, type) or callable(val)

_losses = {
    'LossFunctionWrapper'   : LossFunctionWrapper,
    ** import_objects(__package__.replace('.', os.path.sep), types = type),
    ** import_objects(
        [tf.keras.losses],
        filters = _is_class_or_callable,
        exclude = ('get', 'serialize', 'deserialize', 'Reduction')
    )
}
globals().update(_losses)