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
import logging

from .. import Task
from .processing import image_dataset_wrapper

logger      = logging.getLogger(__name__)

@image_dataset_wrapper(
    name = 'SynthText',
    task = [Task.TEXT_DETECTION, Task.OCR],
    directory = '{}/SynthText/SynthText'
)
def load_data(directory, tqdm = lambda x: x, ** kwargs):
    from scipy.io import loadmat
    
    from utils.image.box_utils import BoxFormat, convert_box_format
    
    metadata_file = os.path.join(directory, 'gt.mat')
    data = loadmat(metadata_file)
    
    dataset = {}
    for i, (img, boxes, words) in enumerate(zip(
        tqdm(data['imnames'][0]), data['wordBB'][0], data['txt'][0]
    )):
        filename = os.path.join(directory, img[0])
        
        cleaned  = []
        for w in words:
            for part in w.split('\n'):
                cleaned.extend(part.strip().split())
        
        if len(boxes.shape) == 2: boxes = np.expand_dims(boxes, axis = -1)
        
        dataset[filename] = {
            'filename'  : filename,
            'boxes'     : convert_box_format(
                np.transpose(boxes, [2, 1, 0]).astype(np.int32),
                target = BoxFormat.XYWH,
                source = BoxFormat.POLY
            ),
            'nb_box'    : boxes.shape[-1],
            'label'     : cleaned
        }
    
    return dataset
