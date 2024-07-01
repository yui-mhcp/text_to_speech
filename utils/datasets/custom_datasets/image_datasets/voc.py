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
import xml.etree.ElementTree as ET

from .. import Task, add_dataset
from .processing import image_dataset_wrapper
from utils.file_utils import load_json

logger      = logging.getLogger(__name__)

@image_dataset_wrapper(
    name    = ('VOC', 'Pascal VOC'),
    task    = Task.OBJECT_DETECTION,
    directory       = '{}/VOC2012',
    images_dir      = 'JPEGImages',
    annotations_dir = 'Annotations'
)
def load_data(directory, images_dir, annotations_dir, ** kwargs):
    """
        Arguments :
            - directory : main directory
            - subset    : the dataset's version (default to VOC2012)
    """
    from utils.image import _image_formats
    
    images_dir  = os.path.join(directory, images_dir)
    annotations_dir = os.path.join(directory, annotations_dir)
    
    metadata = {}
    for ann in sorted(os.listdir(annotations_dir)):
        img_filename, image_w, image_h = None, None, None

        tree = ET.parse(os.path.join(annotations_dir, ann))
        
        boxes, labels = [], []
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img_filename = os.path.join(images_dir, str(elem.text))
                if not img_filename.endswith(_image_formats): img_filename += '.jpg'
            if 'width' in elem.tag:     image_w = int(elem.text)
            if 'height' in elem.tag:    image_h = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                for attr in list(elem):
                    if 'name' in attr.tag:
                        label = str(attr.text)
                    
                    if 'bndbox' in attr.tag:
                        x0, y0, x1, y1 = 0, 0, 0, 0
                        for dim in list(attr):
                            if 'xmin' in dim.tag: x0 = int(round(float(dim.text)))
                            if 'ymin' in dim.tag: y0 = int(round(float(dim.text)))
                            if 'xmax' in dim.tag: x1 = int(round(float(dim.text)))
                            if 'ymax' in dim.tag: y1 = int(round(float(dim.text)))
                        
                        labels.append(label)
                        boxes.append([x0, y0, x1 - x0, y1 - y0])

        metadata[img_filename] = {
            'label' : labels, 'height' : image_h, 'width' : image_w, 'boxes' : boxes
        }
    
    return metadata

add_dataset(
    'kangaroo',
    task    = Task.OBJECT_DETECTION,
    directory   = '{}/kangaroo-master',
    images_dir  = 'images',
    annotations_dir = 'annots',
    processing_fn   = 'voc'
)

add_dataset(
    'raccoon',
    task    = Task.OBJECT_DETECTION,
    directory   = '{}/raccoon-master',
    images_dir  = 'images',
    annotations_dir = 'annots',
    processing_fn   = 'voc'
)
