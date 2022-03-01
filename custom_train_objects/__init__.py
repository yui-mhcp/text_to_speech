
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from utils import get_object, print_objects

from custom_train_objects.history import History

from custom_train_objects.generators import *
from custom_train_objects.losses import _losses
from custom_train_objects.metrics import _metrics, MetricList
from custom_train_objects.callbacks import _callbacks, CallbackList
from custom_train_objects.optimizers import _optimizers, _schedulers


def get_callbacks(callback_name = None, *args, **kwargs):
    return get_object(_callbacks, callback_name, *args, **kwargs, print_name = 'callbacks', 
                      allowed_type = tf.keras.callbacks.Callback)

def get_loss(loss_name, *args, **kwargs):
    return get_object(_losses, loss_name, *args, **kwargs, print_name = 'loss', 
                     allowed_type = tf.keras.losses.Loss)

def get_metrics(metrics_name, *args, **kwargs):
    if isinstance(metrics_name, (list, tuple)):
        return [get_metrics(m, * args, ** kwargs) for m in metrics_name]
    if isinstance(metrics_name, dict):
        kwargs = {** kwargs, ** metrics_name.get('config', {})}
        metrics_name = metrics_name.get('metric', metrics_name)
    return get_object(_metrics, metrics_name, *args, **kwargs, print_name = 'metric', 
                     allowed_type = tf.keras.metrics.Metric)

def get_optimizer(optimizer_name = "adam", *args, **kwargs):
    lr = kwargs.pop('lr', None)
    if lr is None: lr = kwargs.get('learning_rate', None)
    if lr is not None:
        if isinstance(lr, (dict, str)):
            if isinstance(lr, str): lr = {'name' : lr}
            if 'class_name' in lr: 
                name = lr['class_name']
                lr = lr['config']
                lr['name'] = name
            lr = get_object(_schedulers, lr.pop('name'), **lr, print_name = 'lr scheduler')
        kwargs['learning_rate'] = lr
    return get_object(_optimizers, optimizer_name, *args, **kwargs, 
                      print_name = 'optimizer', 
                      allowed_type = tf.keras.optimizers.Optimizer)

def get_policy(policy_name, *args, **kwargs):
    return get_object(_policies, policy_name, *args, **kwargs, print_name = 'policy')


def print_callbacks():
    print_objects(_callbacks, 'callbacks')
    
def print_losses():
    print_objects(_losses, 'losses')
    
def print_metrics():
    print_objects(_metrics, 'metrics')

def print_optimizers():
    print_objects(_optimizers, 'optimizers')

def print_policies():
    print_objects(_policies, 'policies')