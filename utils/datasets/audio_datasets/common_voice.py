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

import os

from ..loader import Task
from .processing import audio_dataset_wrapper

@audio_dataset_wrapper(
    name = 'CommonVoice', task = [Task.TTS, Task.STT, Task.SI], directory = '{}/CommonVoice'
)
def load_data(directory,
              *,
              
              file  = 'validated.tsv',

              age   = None,
              gender    = None,
              accent    = None,

              dropna    = False,
              pop_down  = True,
              pop_votes = True,
              
              ** kwargs
             ):
    def filter_col(dataset, col, values):
        if values is None: return dataset
        if not isinstance(values, (list, tuple)): values = [values]
        values = [str(v).lower() for v in values]
        return dataset[dataset[col].apply(lambda v: isinstance(v, str) and v.lower() in values)]
    
    import pandas as pd
    
    new_columns = {
        'client_id' : 'id', 'path' : 'filename', 'sentence' : 'text'
    }
    
    dataset = pd.read_csv(os.path.join(directory, file), sep = '\t')
    if 'index' in dataset: dataset.pop('index')
    
    dataset['path'] = dataset['path'].apply(lambda f: os.path.join(directory, 'clips', f))

    if dropna: dataset.dropna(inplace = True)

    if pop_down:
        dataset = dataset[dataset['down_votes'] == 0]
        
    if pop_votes:
        dataset.pop('up_votes')
        dataset.pop('down_votes')

    dataset['gender'] = dataset['gender'].apply(
        lambda s: s[0].upper() if isinstance(s, str) else s
    )
    
    dataset = dataset.rename(columns = new_columns)
    
    if age:     dataset = filter_col(dataset, 'age', age)
    if gender:  dataset = filter_col(dataset, 'gender', gender)
    if accent:  dataset = filter_col(dataset, 'accents', accent)
    
    dataset.fillna('?', inplace = True)
    dataset.reset_index(drop = True, inplace = True)
    
    return dataset

