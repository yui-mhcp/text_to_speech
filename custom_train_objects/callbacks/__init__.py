
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

from custom_train_objects.callbacks.ckpt_callback import CkptCallback
from custom_train_objects.callbacks.terminate_on_nan import TerminateOnNaN
from custom_train_objects.callbacks.predictor_callback import PredictorCallback


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

class CallbackList(object):
    def __init__(self, *callbacks):
        self.callbacks = []
        self._init_callbacks(callbacks)
        
    def _init_callbacks(self, callbacks):
        for callback in callbacks:
            if isinstance(callback, tf.keras.callbacks.Callback):
                self.callbacks.append(callback)
            elif isinstance(callback, (list, tuple)):
                self._init_callbacks(callback)
            else:
                raise ValueError("Callback non valide ! \n  Accepté : tf.keras.callbacks.Callback subclass\n  Reçu : {}".format(type(callback)))
        
    def on_train_batch_begin(self, *args, **kwargs):
        [c.on_train_batch_begin(*args, **kwargs) for c in self.callbacks]
    
    def on_train_batch_end(self, *args, **kwargs):
        [c.on_train_batch_end(*args, **kwargs) for c in self.callbacks]
    
    def on_epoch_begin(self, *args, **kwargs):
        [c.on_epoch_begin(*args, **kwargs) for c in self.callbacks]
    
    def on_epoch_end(self, *args, **kwargs):
        [c.on_epoch_end(*args, **kwargs) for c in self.callbacks]
    