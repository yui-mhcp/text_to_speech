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
import pandas as pd

from .. import Task
from .processing import audio_dataset_wrapper, _get_processed_name

@audio_dataset_wrapper(
    name    = 'LibriSpeech',
    task    = [Task.TTS, Task.STT, Task.SI],
    train   = {'directory' : '{}/LibriSpeech', 'subsets' : ['train-clean-100', 'train-clean-360']},
    valid   = {'directory' : '{}/LibriSpeech', 'subsets' : 'test-clean'}

)
def load_data(directory, *, subsets = None, ** kwargs):
    def process_speaker(line):
        line = [p for p in line.split() if p != '|']
        
        speaker_id, subset = line[0], line[2]
        if subsets and subset not in subsets: return []
        speaker_infos = {
            'id'    : speaker_id,
            'total_time'    : float(line[3]),
            'name'  : ' '.join(line[4:]),
            'gender'    : line[1]
        }
        
        speaker_dir = os.path.join(directory, subset, speaker_id)
        
        infos = []
        for subdir in os.listdir(speaker_dir):
            filename = os.path.join(
                speaker_dir, subdir, '{}-{}.trans.txt'.format(speaker_id, subdir)
            )
            with open(filename, 'r', encoding = 'utf-8') as file:
                lines = file.read().split('\n')
            
            for l in lines:
                if not l: continue
                l = l.split()
                
                filename = os.path.join(speaker_dir, subdir, l[0]) + '.flac'
                text     = ' '.join(l[1:]).capitalize()
                
                infos.append({
                    ** speaker_infos,
                    'filename'  : filename,
                    'text'      : text
                })
        
        return infos
    
    if not isinstance(subsets, (list, tuple)): subsets = [subsets] if subsets else []
    subsets = [s for s in subsets if os.path.exists(os.path.join(directory, s))]
    
    infos_filename = os.path.join(directory, 'SPEAKERS.txt')
    
    with open(infos_filename, 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')

    data = []
    for l in lines:
        if not l or l[0] == ';': continue
        data.extend(process_speaker(l))

    dataset = pd.DataFrame(data)
    
    for sub_dir in os.listdir(directory):
        if not sub_dir.startswith('wavs_'): continue
        
        orig_dir = '-'.join(sub_dir.split('-')[1:])
        if not subsets or any(s in sub_dir for s in subsets):
            dataset[_get_processed_name(sub_dir)] = dataset['filename'].apply(
                lambda f: f.replace(orig_dir, sub_dir).replace('.flac', '.wav')
            )
    
    return dataset
