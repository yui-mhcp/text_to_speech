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
import glob
import logging
import numpy as np

from .. import Task
from .processing import image_dataset_wrapper

logger      = logging.getLogger(__name__)

@image_dataset_wrapper(
    name = 'icdar', task = [Task.TEXT_DETECTION, Task.OCR], directory = '{}/ICDAR'
)
def load_data(directory,
              *,

              year    = 2019,
              script  = 'latin',
              tqdm    = lambda x: x,

              ** kwargs
             ):
    from utils.image.bounding_box import BoxFormat, convert_box_format
    
    path    = os.path.join(directory, str(year), 'gt', '*.txt')

    metadata    = {}
    for filename in tqdm(glob.glob(path)):
        with open(filename, 'r', encoding = 'utf-8') as file:
            lines = file.read().split('\n')

        infos = {'boxes' : [], 'label' : []}
        for line in lines:
            if not line: continue
            parts = line.split(',')
            if script and parts[-2].lower() not in script: continue
            if parts[-1] == '###': continue
            
            box = [int(v) for v in parts[:8]]
            infos['boxes'].append(np.reshape(box, [4, 2]))
            infos['label'].append(parts[-1])
            infos['script'] = parts[-2].lower()
        
        if not infos['boxes']: continue
        
        infos['boxes'] = convert_box_format(
            np.array(infos['boxes']), BoxFormat.XYWH, source = BoxFormat.POLY
        )
        img_filename = glob.glob(filename.replace('gt', 'images').replace('.txt', '.*'))[0]
        metadata[img_filename] = infos
    
    return metadata

