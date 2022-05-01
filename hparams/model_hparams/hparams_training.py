
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

from hparams.hparams import HParams

_dataset_config = {
    'batch_size'    : 64,
    'train_batch_size'  : None,
    'valid_batch_size'  : None,
    'test_batch_size'   : 1,
    'shuffle_size'      : 1024
}

HParamsTraining = HParams(   
    ** _dataset_config,
    epochs      = 10,
    
    verbose     = 1,
    
    train_times = 1,
    valid_times = 1,
    
    train_size  = None,
    valid_size  = None,
    test_size   = 4,
    pred_step   = -1
)

HParamsTesting  = HParams(
    batch_size  = _dataset_config['batch_size']
)

