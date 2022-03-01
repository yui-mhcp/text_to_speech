
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

import numpy as np
import tensorflow as tf

class TerminateOnNaN(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(TerminateOnNaN, self).__init__(**kwargs)
        
    def on_train_batch_end(self, batch, logs = None):
        logs = {} if logs is None else logs
        if 'loss' in logs:
            if np.isnan(logs['loss']) or np.isinf(logs['loss']):
                raise ValueError("NaN loss at batch : {}\n  Logs : {}".format(batch, logs))
                