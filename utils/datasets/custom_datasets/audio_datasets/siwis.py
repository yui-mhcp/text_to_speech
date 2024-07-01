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

import os

from .. import Task
from .processing import audio_dataset_wrapper, _get_processed_name

@audio_dataset_wrapper(
    name = 'siwis', task = [Task.TTS, Task.STT], directory = '{}/SIWIS'
)
def load_data(directory, *, lang = 'fr', parts = [1, 2, 3, 5], ** kwargs):
    base_dir = os.path.join(directory, lang)
    
    dataset = []
    for part in parts:
        txt_dir = os.path.join(base_dir, 'text', 'part{}'.format(part))
        
        for filename in os.listdir(txt_dir):
            wav_filename = os.path.join(
                base_dir, 'wavs', 'part{}'.format(part), filename[:-3] + 'wav'
            )
            txt_filename = os.path.join(txt_dir, filename)

            with open(txt_filename, 'r', encoding = 'utf-8') as file:
                text = file.read()
            
            dataset.append({'text' : text, 'filename' : wav_filename})
    
    processed_dirs = [f for f in os.listdir(base_dir) if f.startswith('wavs-')]
    
    for data in dataset:
        data.update({
            _get_processed_name(_dir) : data['filename'].replace('wavs', _dir) + ('.npy' if 'mels_' in col else '')
            for _dir in processed_dirs
        })
    
    
    return base_dir, dataset

