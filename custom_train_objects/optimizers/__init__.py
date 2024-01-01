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

import logging
import tensorflow as tf

from utils.generic_utils import import_objects, get_object, print_objects
from custom_train_objects.optimizers import lr_schedulers

logger = logging.getLogger(__name__)

def get_optimizer(optimizer_name = 'adam', * args, ** kwargs):
    if isinstance(optimizer_name, dict):
        name_key    = 'optimizer' if 'optimizer' in optimizer_name else 'class_name'
        config_key  = 'config' if 'config' in optimizer_name else 'optimizer_config'
        optimizer_name, kwargs = optimizer_name[name_key], optimizer_name[config_key]
    
    lr = kwargs.pop('lr', None)
    if lr is None: lr = kwargs.get('learning_rate', None)
    if lr is not None:
        if isinstance(lr, (dict, str)):
            if isinstance(lr, str): lr = {'name' : lr}
            if 'class_name' in lr:  lr = {** lr['config'], 'name' : lr['class_name']}
            lr = get_object(_schedulers, lr.pop('name'), ** lr, print_name = 'lr scheduler')
        kwargs['learning_rate'] = lr
    return get_object(
        _optimizers, optimizer_name, * args, ** kwargs, print_name = 'optimizer', err = True,
        types = tf.keras.optimizers.Optimizer
    )

def print_optimizers():
    print_objects(_optimizers, 'optimizers')


def _maybe_use_legacy(name):
    def optimizer_init(* args, ** kwargs):
        try:
            return getattr(tf.keras.optimizers, name)(* args, ** kwargs)
        except Exception:
            logger.info('Optimizer initialization failed, trying legacy optimizer')
            return getattr(tf.keras.optimizers.legacy, name)(* args, ** kwargs)
    return optimizer_init

_optimizers = {
    k : _maybe_use_legacy(k) for k in import_objects([tf.keras.optimizers], types = (type, ))
}

_schedulers = import_objects([lr_schedulers], types = type)

globals().update(_schedulers)
globals().update(_schedulers)
