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

import keras
import logging
import inspect

from utils import import_objects, get_object, print_objects, dispatch_wrapper
from custom_train_objects.optimizers import lr_schedulers

logger = logging.getLogger(__name__)

def filter_old_kwargs(name):
    def optimizer_init(* args, ** kwargs):
        opt_class = getattr(keras.optimizers, name)
        kwargs  = {
            k : v for k, v in kwargs.items()
            if k in inspect.signature(opt_class).parameters
        }
        return opt_class(* args, ** kwargs)
    return optimizer_init

_optimizers = {
    k : filter_old_kwargs(k) for k in import_objects(
        keras.optimizers,
        classes = keras.optimizers.Optimizer,
        exclude = ('Optimizer', 'LossScaleOptimizer'),
        allow_functions = False
    )
}

_schedulers = import_objects(
    [lr_schedulers, keras.optimizers.schedules],
    classes     = keras.optimizers.schedules.LearningRateSchedule,
    exclude     = ('LearningRateSchedule', 'CustomScheduler'),
    allow_functions = False
)

globals().update(_optimizers)
globals().update(_schedulers)

@dispatch_wrapper(_optimizers, 'optimizer')
def get_optimizer(optimizer, ** kwargs):
    if isinstance(optimizer, dict):
        name_key    = 'optimizer' if 'optimizer' in optimizer else 'class_name'
        config_key  = 'config' if 'config' in optimizer else 'optimizer_config'
        optimizer, kwargs = optimizer[name_key], optimizer[config_key]
    
    lr = kwargs.pop('lr', None)
    if lr is None: lr = kwargs.get('learning_rate', None)
    if lr is not None:
        kwargs['learning_rate'] = get_lr_scheduler(lr)
    
    return get_object(
        _optimizers, optimizer, ** kwargs, print_name = 'optimizer',
        types = keras.optimizers.Optimizer
    )

def print_optimizers():
    print_objects(_optimizers, 'optimizers')

@dispatch_wrapper(_schedulers, 'scheduler')
def get_lr_scheduler(scheduler, ** kwargs):
    if isinstance(scheduler, (dict, str)):
        if isinstance(scheduler, str):
            scheduler = {'name' : scheduler}
        if 'class_name' in scheduler:
            scheduler = {** scheduler['config'], 'name' : scheduler['class_name']}
        
        scheduler = get_object(
            _schedulers, scheduler.pop('name'), ** scheduler, print_name = 'lr scheduler'
        )
    return scheduler
