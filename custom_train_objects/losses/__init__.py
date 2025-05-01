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
import inspect
import importlib

from keras.src.losses import LossFunctionWrapper

from .loss_with_multiple_outputs import LossWithMultipleOutputs

for module in [keras.losses] + os.listdir(__package__.replace('.', os.path.sep)):
    if isinstance(module, str):
        if module.startswith(('.', '_')) or '_old' in module: continue
        module = importlib.import_module(__package__ + '.' + module[:-3])
    
    globals().update({
        k : v for k, v in vars(module).items()
        if (not k.startswith('_')) and (
            (isinstance(v, type) and issubclass(v, keras.losses.Loss))
            or (callable(v) and 'y_true' in inspect.signature(v).parameters)
        )
    })

_losses = {
    k.lower() : v for k, v in globals().items()
    if (isinstance(v, type) and issubclass(v, keras.losses.Loss)) or (callable(v))
}

def get_loss(loss, * args, ** kwargs):
    if loss == 'crossentropy' or isinstance(loss, keras.losses.Loss):
        return loss
    elif isinstance(loss, dict):
        if loss.get('class_name', None) == 'LossFunctionWrapper':
            return keras.losses.deserialize(loss)
        
        name_key    = 'loss' if 'loss' in loss else 'class_name'
        config_key  = 'config' if 'config' in loss else 'loss_config'
        loss, kwargs = loss[name_key], loss[config_key]

    if loss == 'LossFunctionWrapper': loss = kwargs.pop('fn')['config']
    else: kwargs.pop('fn', None)
    
    if loss.lower() not in _losses:
        raise ValueError('Unknown loss !\n  Accepted : {}\n  Got : {}'.format(
            tuple(_losses.keys()), loss
        ))
    
    loss = _losses[loss.lower()]
    if isinstance(loss, type):
        return loss(** kwargs)
    else:
        assert callable(loss), str(loss)
        kwargs.setdefault('name', loss.__name__)
        return LossFunctionWrapper(loss, ** kwargs)
