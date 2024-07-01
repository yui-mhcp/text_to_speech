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
    name = 'VoxForge', task = [Task.TTS, Task.STT, Task.SI], directory = '{}/VoxForge'
)
def load_data(directory, *, lang = 'fr', ** kwargs):
    def process_speaker(main_dir, name):
        speaker_dir = os.path.join(main_dir, name)
        sub_dirs    = os.listdir(speaker_dir)
        
        filename    = os.path.join(speaker_dir, 'etc', 'prompts-original')
        with open(filename, 'r', encoding = 'utf-8') as file:
            lines = file.read().split('\n')
        
        original_audio_dir = 'wav' if 'wav' in sub_dirs else 'flac'
        ext = '.' + original_audio_dir
        
        speaker_data = []
        for l in lines:
            if not l: continue
            l = l.split(' ')
            # Add original informations
            audio_name, text = l[0], ' '.join(l[1:])
            infos = {
                'id'    : name,
                'filename'  : os.path.join(speaker_dir, original_audio_dir, audio_name) + ext,
                'text'  : text
            }
            
            # Add additional preprocessed files (like wavs_22050 if processed)
            for sub_dir in sub_dirs:
                if sub_dir in ('etc', original_audio_dir) or not os.path.isdir(
                    os.path.join(speaker_dir, sub_dir)):
                    continue
                
                infos[_get_processed_name(sub_dir)] = os.path.join(
                    speaker_dir, sub_dir, audio_name
                ) + '.wav'
            
            speaker_data.append(infos)
        
        return speaker_data
    
    directory = os.path.join(directory, lang)

    data = []
    for name in os.listdir(directory):
        data += process_speaker(directory, name)

    return data

