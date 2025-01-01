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
from utils import load_json
from .processing import audio_dataset_wrapper, _get_processed_name

@audio_dataset_wrapper(
    name = 'identification', task = [Task.TTS, Task.STT, Task.SI]
)
def load_data(directory, *, by_part = False, ** kwargs):
    import pandas as pd
    
    if 'parts' not in os.listdir(directory):
        return pd.concat([load_data(
            os.path.join(directory, sub_dir), by_part = by_part, ** kwargs
        ) for sub_dir in os.listdir(directory)], ignore_index = True)
    
    from .. import get_dataset_dir
    
    sub_dir_name = 'parts' if by_part else 'alignments'
    
    directory           = os.path.join(directory, sub_dir_name)
    metadata_filename   = os.path.join(directory, 'map.json')

    data = load_json(metadata_filename)

    dataset = pd.DataFrame(data)
    if not os.path.exists(dataset.loc[0, 'filename']):
        dataset['filename'] = dataset['filename'].apply(
            lambda f: f.replace('D:/datasets', get_dataset_dir())
        )
    if 'indexes' in dataset.columns: dataset.pop('indexes')
    
    return directory, dataset
