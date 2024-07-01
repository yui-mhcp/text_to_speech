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

from .. import Task, add_dataset
from .processing import image_dataset_wrapper
from utils.file_utils import load_json

logger      = logging.getLogger(__name__)

@image_dataset_wrapper(
    name    = 'COCO',
    task    = [
        Task.OBJECT_DETECTION,
        Task.OBJECT_SEGMENTATION,
        Task.IMAGE_CAPTIONING
    ],
    train   = {
        'directory'     : '{}/COCO',
        'images_dir'    : 'train2017',
        'annotations_file'  : 'annotations/instances_train2017.json'
    },
    valid   = {
        'directory'     : '{}/COCO',
        'images_dir'    : 'val2017',
        'annotations_file'  : 'annotations/instances_val2017.json'
    }
)
def load_data(directory,
              annotation_file,

              images_dir   = None,

              keep_labels  = True,
              keep_boxes   = True,
              keep_caption = True,
              keep_segmentation    = False,
              use_supercategory_as_label = False,

              ** kwargs
             ):
    images_dir  = os.path.join(directory, images_dir) if images_dir else directory
    annotation_file = os.path.join(directory, annotation_file)
    
    infos    = load_json(annotation_file)
    
    metadata = {}
    for image in infos['images']:
        metadata[image['id']] = {
            'filename'  : os.path.join(images_dir, image['file_name']),
            'height'    : image['height'],
            'width'     : image['width']
        }
    
    if keep_caption:
        captions = load_json(os.path.join(directory, annot_file.replace('instances', 'captions')))
        for row in captions['annotations']:
            metadata[row['image_id']].setdefault('text', []).append(row['caption'])
    
    if keep_segmentation or keep_boxes or keep_labels:
        categories  = {row['id'] : row for row in infos['categories']}
        for row in infos['annotations']:
            category = categories[row['category_id']]
            
            metadata[row['image_id']].setdefault('label', []).append(
                category['name'] if not use_supercategory_as_label else category['supercategory']
            )
            
            if keep_boxes:
                metadata[row['image_id']].setdefault('boxes', []).append(
                    [int(c) for c in row['bbox']]
                )
            
            if keep_segmentation:
                metadata[row['image_id']].setdefault('segmentation', []).append(
                    row['segmentation']
                )
    
    return {row['filename'] : {** row, 'id' : k} for k, row in metadata.items()}

add_dataset(
    'fungi',
    task    = Task.OBJECT_DETECTION,
    train   = {'directory' : '{}/Fungi', 'annotation_file' : 'train.json'},
    valid   = {'directory' : '{}/Fungi', 'annotation_file' : 'val.json'},
    processing_fn = 'coco'
)

