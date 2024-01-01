
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

from utils.generic_utils import get_object, print_objects

from custom_train_objects.callbacks.ckpt_callback import CkptCallback
from custom_train_objects.callbacks.terminate_on_nan import TerminateOnNaN
from custom_train_objects.callbacks.predictor_callback import PredictorCallback

def get_callbacks(callback_name = None, * args, ** kwargs):
    return get_object(
        _callbacks, callback_name, * args, ** kwargs, print_name = 'callbacks', err = True,
        types = tf.keras.callbacks.Callback
    )


def print_callbacks():
    print_objects(_callbacks, 'callbacks')


_callbacks = {
    'CkptCallback'          : CkptCallback,
    'checkpoint'            : CkptCallback,
    'ModelCheckpoint'       : CkptCallback,
    'model_checkpoint'      : CkptCallback,
    'csv_logger'            : tf.keras.callbacks.CSVLogger,
    'CSVLogger'             : tf.keras.callbacks.CSVLogger,
    'early_stopping'        : tf.keras.callbacks.EarlyStopping,
    'EarlyStopping'         : tf.keras.callbacks.EarlyStopping,
    'reduce_lr'             : tf.keras.callbacks.ReduceLROnPlateau,
    'ReduceLROnPlateau'     : tf.keras.callbacks.ReduceLROnPlateau,
    'TerminateOnNaN'        : TerminateOnNaN,
    'terminate_on_nan'      : TerminateOnNaN, 
    'PredictorCallback'     : PredictorCallback,
    'predictor_callback'    : PredictorCallback
}