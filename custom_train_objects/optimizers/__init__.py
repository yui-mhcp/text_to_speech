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

import keras
import inspect

from .lr_schedulers import *

globals().update({
    k : v for k, v in vars(keras.optimizers).items()
    if not k.startswith('_') and isinstance(v, type) and issubclass(v, keras.optimizers.Optimizer)
})
globals().update({
    k : v for k, v in vars(keras.optimizers.schedules).items()
    if isinstance(v, type) and issubclass(v, keras.optimizers.schedules.LearningRateSchedule)
})

_optimizers = {
    k.lower() : v for k, v in globals().items()
    if isinstance(v, type) and issubclass(v, keras.optimizers.Optimizer)
}
_lr_schedulers  = {
    k.lower() : v for k, v in globals().items()
    if isinstance(v, type) and issubclass(v, keras.optimizers.schedules.LearningRateSchedule)
}

def get_optimizer(optimizer, ** kwargs):
    if isinstance(optimizer, keras.optimizers.Optimizer):
        return optimizer
    elif isinstance(optimizer, dict):
        name_key    = 'optimizer' if 'optimizer' in optimizer else 'class_name'
        config_key  = 'config' if 'config' in optimizer else 'optimizer_config'
        optimizer, kwargs = optimizer[name_key], optimizer[config_key]
    
    lr = kwargs.pop('lr', None)
    if lr is None: lr = kwargs.get('learning_rate', None)
    if lr is not None:
        kwargs['learning_rate'] = get_lr_scheduler(lr)
    
    opt_class = _optimizers[optimizer.lower()]
    kwargs  = {
        k : v for k, v in kwargs.items()
        if k in inspect.signature(opt_class).parameters
    }
    return opt_class(** kwargs)

def get_lr_scheduler(scheduler, ** kwargs):
    if isinstance(scheduler, (dict, str)):
        if isinstance(scheduler, str):
            scheduler = {'name' : scheduler}
        if 'class_name' in scheduler:
            scheduler = {** scheduler['config'], 'name' : scheduler['class_name']}
        
        name = scheduler.pop('name')
        scheduler = _lr_schedulers[name.lower()](** scheduler)
    
    return scheduler
