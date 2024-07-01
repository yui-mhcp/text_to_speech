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
    name    = ('Wider', 'Wider Faces'), task = Task.OBJECT_DETECTION,
    train   = {'directory' : '{}/Wider_Face', 'subset' : 'train'},
    valid   = {'directory' : '{}/Wider_Face', 'subset' : 'val'}
)
def load_data(directory,
              *,

              subset  = 'train',
              label_name  = 'face',
              keep_invalid    = False,
              add_box_infos   = False,
              ** kwargs
             ):
    """
        Arguments :
            - filename  : the annotation filename
            - img_dir   : directory where images are stored
            - label_name        : all are faces but you can specify another name
        Return :
            - dict {filename : infos}
    
        Annotation format :
            image_filename
            nb_box
            x, y, w, h, blur, expression, illumination, invalid, occlusion, pose = infos
            x, y, w, h, blur, expression, illumination, invalid, occlusion, pose = infos
            ...
    """
    assert subset in ('train', 'val')
    
    box_info_expl = {
        "blur"          : {0: "clear", 1: "normal", 2: "heavy"},
        "expression"    : {0: "typical", 1: "exagerate"},
        "illumination"  : {0: "normal", 1: "extreme"},
        "occlusion"     : {0: "no", 1: "partial", 2: "heavy"},
        "pose"          : {0: "typical", 1: "atypical"},
        "invalid"       : {0: "false", 1: "true"}
    }

    filename = os.path.join(
        directory, 'wider_face_split', 'wider_face_{}_bbx_gt.txt'.format(subset)
    )
    img_dir = os.path.join(directory, 'WIDER_{}'.format(subset), 'images')
    
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')
    
    i = 0
    metadata = {}
    while i < len(lines):
        if not lines[i]: break
        
        img_filename    = os.path.join(img_dir, lines[i])
        category    = lines[i].split('/')[0]
        nb_box      = int(lines[i + 1])
        
        i += 2
        
        boxes, labels, boxes_infos = [], [], []
        for j in range(nb_box):
            infos = lines[i].split(' ')
            infos = [int(info) for info in infos if info != '']
            
            x, y, w, h, blur, expression, illumination, invalid, occlusion, pose = infos
            
            i+=1
            
            if not keep_invalid and invalid == 1: continue
            
            labels.append(label_name)
            boxes.append([x, y, w, h])

            if add_box_infos:
                box_infos = {
                    'blur' : blur, 'expression' : expression, 'invalid' : invalid, 
                    'occlusion' : occlusion, 'pose' : pose
                }
                for info_name, value in box_infos.items():
                    box_infos[info_name] = box_info_expl[info_name][value]
                boxes_infos.append(box_infos)
        
        metadata[img_filename] = {
            'label' : labels, 'boxes' : boxes, 'category' : category
        }
        if add_box_infos:
            metadata[img_filename]['box_infos'] = boxes_infos
    
    return metadata

