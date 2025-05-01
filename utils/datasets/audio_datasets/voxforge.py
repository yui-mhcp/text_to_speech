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
    name = 'VoxForge', task = [Task.TTS, Task.STT, Task.SI], directory = '{}/VoxForge'
)
def load_data(directory, *, lang = 'fr', ** kwargs):
    directory = os.path.join(directory, lang)

    data = []
    for name in os.listdir(directory):
        data.extend(_process_speaker(directory, name))

    return directory, data

def _process_speaker(directory, name):
    speaker_dir = os.path.join(directory, name)
    sub_dirs    = os.listdir(speaker_dir)

    filename    = os.path.join(speaker_dir, 'etc', 'prompts-original')
    with open(filename, 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')

    audio_dir = 'wav' if 'wav' in sub_dirs else 'flac'

    speaker_data = []
    for l in lines:
        if not l: continue
        audio_name, _, text = l.partition(' ')

        speaker_data.append({
            'id'    : name,
            'filename'  : os.path.join(speaker_dir, audio_dir, audio_name) + '.' + audio_dir,
            'text'  : text
        })

    return speaker_data
