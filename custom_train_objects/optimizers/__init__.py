
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

from custom_train_objects.optimizers.lr_schedulers import *

_optimizers = {
    'Adam'          : tf.keras.optimizers.Adam,
    'Ddadelta'      : tf.keras.optimizers.Adadelta,
    'Ddagrad'       : tf.keras.optimizers.Adagrad,
    'Ddam'          : tf.keras.optimizers.Adam,
    'RMSprop'       : tf.keras.optimizers.RMSprop,
    'SGD'           : tf.keras.optimizers.SGD,
}

_schedulers = {
    'DivideByStep'      : DivideByStep, 
    'SinScheduler'      : SinScheduler,
    'TanhDecayScheduler'    : TanhDecayScheduler,
    'WarmupScheduler'   : WarmupScheduler
}