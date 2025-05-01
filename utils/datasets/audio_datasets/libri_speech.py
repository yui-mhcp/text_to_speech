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
    name    = 'LibriSpeech',
    task    = [Task.TTS, Task.STT, Task.SI],
    train   = {'directory' : '{}/LibriSpeech', 'subsets' : ['train-clean-100', 'train-clean-360']},
    valid   = {'directory' : '{}/LibriSpeech', 'subsets' : 'test-clean'}

)
def load_data(directory, *, subsets = None, ** kwargs):
    if not isinstance(subsets, (list, tuple)): subsets = [subsets] if subsets else []
    subsets = [s for s in subsets if os.path.exists(os.path.join(directory, s))]
    
    infos_filename = os.path.join(directory, 'SPEAKERS.txt')
    with open(infos_filename, 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')

    dataset = []
    for l in lines:
        if not l or l[0] == ';': continue
        dataset.extend(_process_speaker(directory, subsets, l))
    
    return dataset

def _process_speaker(directory, subsets, line):
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
            filename, text = l.partition(' ')
            
            filename = os.path.join(speaker_dir, subdir, filename) + '.flac'
            text     = text.capitalize()

            infos.append({
                ** speaker_infos,
                'filename'  : filename,
                'text'      : text
            })

    return infos
