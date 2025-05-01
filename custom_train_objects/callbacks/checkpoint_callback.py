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

from keras.callbacks import ModelCheckpoint

class CheckpointCallback(ModelCheckpoint):
    def __init__(self, checkpoint_manager, ** kwargs):
        kwargs.update({
            'filepath'  : checkpoint_manager.best_checkpoint_path,
            'save_best_only'    : True,
            'save_weights_only' : True
        })
        super().__init__(** kwargs)
        self.checkpoint_manager = checkpoint_manager

    def on_train_end(self, logs = None):
        self.checkpoint_manager.load('best')
    
    def _save_model(self, epoch, batch, logs):
        super()._save_model(epoch, batch, logs)
        
        self.checkpoint_manager.set_best_checkpoint_infos(epoch = epoch, logs = logs)
        