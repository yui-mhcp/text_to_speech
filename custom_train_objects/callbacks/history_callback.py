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

class HistoryCallback(keras.callbacks.Callback):
    def __init__(self, history):
        self.history = history
        
        self.set_config = self.history.set_config
        for name in vars(self.history.__class__).keys():
            if name.startswith('on_'): setattr(self, name, getattr(self.history, name))
        

