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
import glob
import keras

from keras.src.losses import LossFunctionWrapper

from utils import import_objects, get_object, print_objects, is_function, dispatch_wrapper
from .loss_with_multiple_outputs import LossWithMultipleOutputs

_losses = import_objects(
    [__package__.replace('.', os.path.sep), keras.losses],
    classes     = keras.losses.Loss,
    signature   = ['y_true', 'y_pred'],
    exclude     = ('Loss', 'LossFunctionWrapper', 'LossWithMultipleOutputs')
)
globals().update(_losses)

@dispatch_wrapper(_losses, 'loss')
def get_loss(loss, * args, ** kwargs):
    if loss == 'crossentropy': return loss
    if isinstance(loss, dict):
        if loss.get('class_name', None) == 'LossFunctionWrapper':
            return keras.losses.deserialize(loss)
        
        name_key    = 'loss' if 'loss' in loss else 'class_name'
        config_key  = 'config' if 'config' in loss else 'loss_config'
        optimizer, kwargs = loss[name_key], loss[config_key]

    if loss == 'LossFunctionWrapper': loss = kwargs.pop('fn')['config']
    else: kwargs.pop('fn', None)
    return get_object(
        _losses, loss, * args, ** kwargs, types = (type, keras.losses.Loss),
        print_name = 'loss', function_wrapper = LossFunctionWrapper
    )

def print_losses():
    print_objects(_losses, 'losses')

def add_loss(loss, name = None):
    if name is None: name = loss.__name__ if is_function(loss) else loss.__class__.__name__
    get_loss.dispatch(loss, name)

