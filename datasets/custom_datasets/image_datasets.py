
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
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
import glob
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET

from tqdm import tqdm
from functools import wraps
from multiprocessing import cpu_count

from loggers import timer
from utils.thread_utils import Consumer
from utils.file_utils import load_json, dump_json
from datasets.custom_datasets import add_dataset
from datasets.dataset_utils import _maybe_load_embedding

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

GAN = 'image generation'
CLASS   = 'object classification'
DETECT  = 'object detection'
SEGMENT = 'object segmentation'
CAPTION = 'image captioning'
FACE_RECOGN = 'face recognition'
SCENE_TEXT  = 'scene text detection'
OCR     = 'OCR'

BOX_KEY     = 'box'
N_BOX_KEY   = 'nb_box'

def image_dataset_wrapper(name, task, ** default_config):
    def wrapper(dataset_loader):
        """
            Wraps a function that loads the image dataset and then apply some post-processing
            The function must return a dict {filename : infos} where `infos` can contain :
                - (optional) label  : the label(s) in the image
                - (optional) box    : the bounding box(es) in the image (in mode [x, y, w, h])
                - (optional) box_infos  : the box(es)' information
                - (optional) segmentation   : the object's segmentation(s)
                
                - (optional) text   : the image's caption(s)
        """
        @timer(name = '{} loading'.format(name))
        @wraps(dataset_loader)
        def _load_and_process(directory,
                              * args,
                              add_image_size = None,
                              
                              keep_empty    = True,
                              accepted_labels   = None,
                              labels_subtitutes = None,
                              
                              min_box_per_image = -1,
                              max_box_per_image = -1,
                              one_line_per_box  = False,
                              
                              one_line_per_caption  = False,
                              
                              return_raw    = False,
                              
                              ** kwargs
                             ):
            assert not (one_line_per_box and one_line_per_caption)
            
            dataset = dataset_loader(directory, * args, ** kwargs)
            if return_raw: return dataset
            
            dataset = [
                {'filename' : file, ** row} for file, row in dataset.items()
            ]
            
            if 'label' in dataset[0]:
                dataset = _replace_labels(dataset, labels_subtitutes)

                dataset = _filter_labels(dataset, accepted_labels, keep_empty)

                if any('mask' in row or 'segmentation' in row for row in dataset):
                    for row in dataset:
                        if 'mask' in row:           key = 'mask'
                        elif 'segmentation' in row: key = 'segmentation'
                        else: continue
                        
                        max_len     = max([len(poly) for poly in row[key]])
                        row[key]    = np.array(
                            [poly + [-1] * (max_len - len(poly)) for poly in row[key]],
                            dtype = np.int32
                        )

                if any(BOX_KEY in row for row in dataset):
                    for row in dataset: row.setdefault(BOX_KEY, [])
                    
                    if min_box_per_image > 0 or max_box_per_image > 0:
                        if max_box_per_image <= 0: max_box_per_image = float('inf')
                        dataset = [
                            row for row in dataset if (
                                len(row[BOX_KEY]) >= min_box_per_image 
                                and len(row[BOX_KEY]) <= max_box_per_image
                            )
                        ]
                
                    if one_line_per_box:
                        dataset = _flatten_dataset(
                            dataset, keys = [BOX_KEY, 'label', 'box_infos', 'segmentation', 'mask']
                        )
            
            if one_line_per_caption and 'text' in dataset[0]:
                dataset = _flatten_dataset(dataset, keys = 'text')
            
            dataset = pd.DataFrame(dataset)

            if BOX_KEY in dataset.columns:
                dataset[BOX_KEY]    = dataset[BOX_KEY].apply(
                    lambda boxes: np.array(boxes, dtype = np.int32) if isinstance(boxes, list) else boxes
                )
                dataset[N_BOX_KEY]  = dataset[BOX_KEY].apply(len) if not one_line_per_box else 1
            
            if add_image_size is None: add_image_size = True if BOX_KEY in dataset.columns else False
            if add_image_size and 'width' not in dataset.columns:
                dataset = _add_image_size(dataset)
            
            dataset = _maybe_load_embedding(directory, dataset, ** kwargs)
            dataset['dataset_name'] = name
            
            return dataset
        
        add_dataset(name, processing_fn = _load_and_process, task = task, ** default_config)
        
        return _load_and_process
    return wrapper

def _replace_labels(dataset, labels_subtitutes):
    if not labels_subtitutes: return dataset
    
    for row in dataset:
        if 'label' not in row: continue
        if not isinstance(row['label'], list):
            row['label'] = labels_subtitutes.get(row['label'], row['label'])
        else:
            row['label'] =  [
                labels_subtitutes.get(l, l) for l in row['label']
            ]
    return dataset

def _filter_labels(dataset, accepted_labels, keep_empty):
    if not accepted_labels:
        return dataset if keep_empty else [row for row in dataset if row.get('label', None)]
    
    if not isinstance(accepted_labels, list): accepted_labels = [accepted_labels]
    
    for row in dataset:
        if 'label' not in row: continue
        if not isinstance(row['label'], list):
            row['label'] = None if row['label'] not in accepted_labels else row['label']
        else:
            for i in reversed(range(len(row['label']))):
                if row['label'][i] not in accepted_labels:
                    for k in ['label', 'box', 'box_infos', 'segmentation']:
                        if k not in row: continue
                        row[k].pop(i)
        
    return dataset if keep_empty else [row for row in dataset if row.get('label', None)]

def _flatten_dataset(dataset, keys):
    if not isinstance(keys, list): keys = [keys]
    if any(not isinstance(dataset[0].get(k, []), (list, tuple, np.ndarray)) for k in keys):
        return dataset
    
    flat = []
    for row in dataset:
        if all(k not in row for k in keys):
            flat.append(row)
            continue
        
        for i in range(len(row[keys[0]])):
            flat.append({
                k : v if k not in keys else v[i] for k, v in row.items()
            })
    return flat

def _add_image_size(dataset):
    from utils.image import get_image_size
    cons = Consumer(get_image_size, max_workers = cpu_count())
    cons.start()
    
    sizes = cons.extend_and_wait(dataset['filename'].values, stop = True)
    
    dataset['height']   = [h for h, _ in sizes]
    dataset['width']    = [w for _, w in sizes]
    
    return dataset

def _rectangularize_boxes(dataset):
    max_box = dataset['nb_box'].max()
    def to_rectangular_box(row):
        rect_boxes = np.zeros((max_box, 4))
        rect_boxes[: len(row['box'])] = row['box']
        
        row['box'] = rect_boxes
        row['label'] = list(row['label']) + [''] * (max_box - len(row['label']))
        return row
    
    return dataset.apply(to_rectangular_box, axis = 1)

@image_dataset_wrapper('image directory', task = CLASS)
def preprocess_image_directory(directory, ** kwargs):
    from utils.image import _image_formats
    
    metadata = {}
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            metadata.update({
                os.path.join(directory, filename, f) : {'label' : filename}
                for f in os.listdir(os.path.join(directory, filename))
                if f.endswith(_image_formats)
            })
        elif filename.endswith(_image_formats):
            metadata[os.path.join(directory, filename)] = {}
    
    return metadata
            

@timer(name = 'yolo output')
def preprocess_annotation_annots(directory, img_dir = None, one_line_per_box = False, ** kwargs):
    """
        Directory of the form
            directory/
                <dir_1>/
                <dir_2>/
                ...
                <dir_1>_boxes.json
                <dir_2>_boxes.json
                ...
        where the `.json` files contains {filename : {'boxes' : [...]}} and `filename` is in <dir_x>/<filename>
    """
    if img_dir is None: img_dir = directory
    
    dataset = {}
    for metadata_file in os.listdir(directory):
        if not metadata_file.endswith('_boxes.json'): continue
        
        sub_img_dir = metadata_file[:-11] # remove _boxes.json
        
        metadata = load_json(os.path.join(directory, metadata_file))
        
        metadata = {
            os.path.join(img_dir, sub_img_dir, img_name) : infos
            for img_name, infos in metadata.items()
        }
        metadata = {file : infos for file, infos in metadata.items() if os.path.exists(file)}
        for file, data in metadata.items():
            data.update({
                'label' : [box[-1] for box in data['boxes']],
                BOX_KEY : [box[:4] for box in data['boxes']]
            })
            data.pop('boxes', None)
        
        dataset.update(metadata)
    
    return dataset

@image_dataset_wrapper(name = 'essex', task = FACE_RECOGN, directory = '{}/faces_essex')
def preprocess_essex_annots(directory, ** kwargs):
    """ Returns a dict {filename : infos} where `infos` contains keys {label:, difficulty:, sex:} """
    def process_dir(dir_path, diff, sex):
        metadata    = {}
        for identifiant in os.listdir(dir_path):
            for img_name in os.listdir(os.path.join(dir_path, identifiant)):
                if not img_name.endswith(_image_formats): continue
                
                metadata[os.path.join(dir_path, identifiant, img_name)] = {
                    'label' : identifiant, 'difficulty' : diff, 'sex' : sex
                }
        return metadata
    
    from utils.image import _image_formats
    
    folders = [
        [os.path.join("faces94", "male"), 1, 'M'],
        [os.path.join("faces94", "female"), 1, 'F'],
        [os.path.join("faces94", "malestaff"), 1, None],
        ["faces95", 2, None],
        ["faces96", 3, None],
        ["grimace", 4, None]
    ]
    
    metadata = {}
    for folder, diff, sex in folders:
        metadata.update(process_dir(os.path.join(directory, folder), diff, sex))
    
    return metadata

@image_dataset_wrapper(
    name = 'CelebA', task = FACE_RECOGN,
    train   = {'directory'  : '{}/CelebA', 'img_dir' : 'img_align_celeba', 'subset' : 'train'},
    valid   = {'directory'  : '{}/CelebA', 'img_dir' : 'img_align_celeba', 'subset' : 'valid'}
)
def preprocess_celeba_annots(directory, img_dir, subset, ** kwargs):
    if subset == 'train':   subset = '0'
    elif subset == 'valid': subset = '1'
    elif subset == 'test':  subset = '2'
    metadata = {}
    
    identity_filename = os.path.join(directory, 'identity_CelebA.txt')
    with open(identity_filename, 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')
    
    for line in lines:
        if len(line) == 0: continue
        img, celeb_id = line.split()
        
        metadata[img] = {'filename' : os.path.join(directory, img_dir, img), 'label' : celeb_id}
    
    with open(os.path.join(directory, 'list_eval_partition.txt'), 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')
    
    eval_split = {}
    for line in lines:
        if len(line) == 0: continue
        img, mode = line.split()
        
        eval_split[img] = mode
    
    return {
        v['filename'] : {'label' : v['label']}
        for k, v in metadata.items() if eval_split.get(k, -1) == subset 
    }

@image_dataset_wrapper(
    name    = 'TDFace', task = FACE_RECOGN,
    train   = {'directory'  : '{}/TDFace', 'acquisition' : 'RGB_A', 'subsets' : [1, 2, 3]},
    valid   = {'directory'  : '{}/TDFace', 'acquisition' : 'RGB_A', 'subsets' : 4}
)
def preprocess_td_face_annots(directory, subsets, acquisition, ** kwargs):
    """
        Dataset with the following structure :
            {directory}/
                TD_{subset}_Set{sets}/
                    <id>/
                        image_1.jpg
                        ...
        
        The `subset` argument refers to the type of acquisition (typically RGB_E, RGB_A, ...) and `sets` refer to (a list of) set number (from 1 to 4 in general)
        By default, sets [1, 2, 3] are used as "train set" and set 4 is used for validation
    """
    metadata = {}
    if isinstance(acquisition, (list, tuple)):
        for acq in acquisition:
            metadata.update(preprocess_td_face_annots(
                directory, subsets = subsets, acquisition = acq, ** kwargs
            ))
        return metadata
    if isinstance(subsets, (list, tuple)):
        for s in subsets:
            metadata.update(preprocess_td_face_annots(
                directory, subsets = s, acquisition = acquisition, ** kwargs
            ))
        return metadata
    
    set_dir = os.path.join(directory, 'TD_{}_Set{}'.format(acquisition, subsets))
    metadata = preprocess_image_directory(set_dir)
    
    if 'E' in acquisition:
        for file, row in metadata.items(): row['expression'] = file[:-4].split('_')[-1]
    elif 'A' in acquisition:
        for file, row in metadata.items():
            row.update({
                'position'  : file[:-4].split('_')[-1],
                'expression'    : file.split('_')[-2]
            })
    
    return metadata

@image_dataset_wrapper(
    name    = 'Youtube Faces', task = [DETECT, FACE_RECOGN],
    train   = {'directory'  : '{}/YoutubeFaces', 'subsets' : [1, 2, 3]},
    valid   = {'directory'  : '{}/YoutubeFaces', 'subsets' :  4}
)
def preprocess_youtube_faces_annots(directory, subsets, overwrite = False,
                                    skip_frames = -1, frames_per_video = -1,
                                    tqdm = lambda x: x, ** kwargs):
    @timer(name = 'frame extraction')
    def extract_frames(video_id, person_id, ** kwargs):
        video_dir = os.path.join(extracted_dir, video_id)
        metadata_filename = os.path.join(video_dir, 'metadata.json')
        if os.path.exists(metadata_filename) and not overwrite:
            return load_json(metadata_filename)

        filename = glob.glob('{}/**/**/{}.npz'.format(directory, video_id))
        if len(filename) != 1:
            logger.error('No / multiple result(s) for video ID {} : {}'.format(video_id, filename))
            return []

        from utils.image import save_image

        subset = int(os.path.dirname(filename[0])[-1:])
        video = np.load(filename[0])

        os.makedirs(video_dir, exist_ok = True)

        infos = {}
        for i, img in enumerate(np.transpose(video['colorImages'], [3, 0, 1, 2])):
            img_name = os.path.join(video_dir, 'frame_{}.jpg'.format(i))
            save_image(filename = img_name, image = img)

            infos.append({
                'id'       : person_id,
                'video_id' : video_id,
                'frame'    : i,
                'subset'   : subset,
                'filename' : img_name,
                'bbox'     : video['boundingBox'][:,0,i],
                ** kwargs
            })

        video.close()

        dump_json(metadata_filename, infos)
        return infos
    
    def convert_bbox(box):
        x1, y1, x2, y2 = [int(c) for c in box]
        return [x1, y1, x2 - x1, y2 - y1]
    
    if not isinstance(subsets, (list, tuple)): subsets = [subsets]
    
    extracted_dir = os.path.join(directory, 'extracted')
    os.makedirs(extracted_dir, exist_ok = True)

    infos = pd.read_csv('{}/youtube_faces_with_keypoints_full.csv'.format(directory))
    
    metadata = {}
    for idx, row in tqdm(infos.iterrows()):
        frames = extract_frames(
            row['videoID'], person_id = row['personName'], height = row['imageHeight'], width = row['imageWidth']
        )
        step = skip_frames
        if step <= 1 and frames_per_video > 0: step = max(1, len(frames) // frames_per_video)
        if step > 1: frames = frames[::step]
        if frames_per_video > 0: frames = frames[:frames_per_video]
        
        metadata.update({
            frame['filename'] : frame for frame in frames if frame['subset'] in subsets
        })
    
    for _, frame in metadata.items(): frame[BOX_KEY] = convert_bbox(frame.pop('bbox'))
    
    return metadata

@timer(name = 'yolo output loading')
def preprocess_yolo_output_annots(filename, img_dir=None, one_line_per_box=False, 
                           box_mode=0, accepted_labels=None, labels_substituts=None, 
                           keep_empty=False, **kwargs):
    """
        Prend en argument :
        - filename    : le chemin d'un fichier structuré comme suit:
            nom_fichier
            nombre_box
            x y w h label #box1
            x y w h label #box2
            ...
        - one_line_per_box : voir la valeur de retour
        - img_dir       : nouveau chemin pour les images
        - box_mode      : modifie le type de box de retour
            -- 0 : box = [x, y, w, h]
            -- 1 : box = [x_min, y_min, x_max, y_max]
            -- 2 : box = {'name':label, 'xmin':x0, 'xmax':x1, 'ymin':y0, 'ymax':y1, 'width':w, 'height':h} #width et height representent la hauteur/largeur de la box et non de l'image ! h = y1 - y0 et w = x1 - x0
        - accepted_labels : liste delabels acceptes, si le label n'est pas dedans il ne sera pas ajouté à la DataFrame (None pour accepter tous les labels)
        - replace_label : si un label n'est pas dans accepted_labels mais que len(accepted_labels) == 1, remplace le label par l'unique label accepté (utile pour remplacer des 'sous-categories' par une categorie générique comme 'labrador, berger allemand, ...' --> 'chien')
    
        Renvoie une DataFrame contenant les colonnes suivantes : 
        - si one_line_per_box == True:
            - filename  : le chemin complet du fichier image (soit tel qu'écritdans le fichier soit combiné avec image_path si image_path != None) (peut apparaitre plusieurs fois si plusieurs objets dans l'images)
            - label     : le label présent à l'endroit dela box
            - box       : 'box' représentant la position dans l'image où se trouve l'objet. 
        - si one_line_per_box == False:
            - filename  : le chemin complet du fichier image (soit tel qu'écritdans le fichier soit combiné avec image_path si image_path != None) (en théorie unique sauf si erreur dans le fichier d'annotation)
            - label     : liste des objets présents dans l'image
            - box       : liste de liste 'box + [label]'
        - width     : largeur de l'image
        - height    : hauteur de l'image
                
        ainsi qu'un dictionnaire reprenant le nombre de fois que chaque label a été vu. 
    """
    from utils.image import get_image_size
    
    assert box_mode in (0, 1, 2)
    assert one_line_per_box in (True, False)
    
    columns = ["filename", "label", "box", "n_box", "width", "height"]
    datas = []
    
    with open(filename, 'r', encoding='utf-8') as fichier:
        lines = fichier.read().split('\n')
        
    i = 0
    while i < len(lines):
        if lines[i] == '': break
        if img_dir is None:
            img_filename = lines[i]
        else:
            img_filename = os.path.basename(lines[i])
            img_filename = os.path.join(img_dir, img_filename)
        i += 1
        n_pers = int(lines[i])
        i += 1
        if not os.path.exists(img_filename):
            i += n_pers
            logger.warning('{} is in annotation file but does not exist'.format(img_filename))
            continue
            
        image_w, image_h = get_image_size(img_filename)
        
        boxes, labels = [], []
        for j in range(n_pers):
            infos = lines[i].split(' ')
            i += 1
            
            label = infos[4]
            if labels_substituts is not None:
                if label in labels_substituts:
                    label = labels_substituts[label]
                elif '*' in labels_substituts:
                    label = labels_substituts['*']
            
            if accepted_labels is not None and label not in accepted_labels:
                continue
                                
            box = [int(b) for b in infos[:4]]
            if box_mode == 1:
                x, y, w, h = box
                box = [x, y, x + w, y + h]
            elif box_mode == 2:
                x, y, w, h = box
                box = {'name':label, 'xmin':x, 'ymin':y, 'xmax':x+w, 'ymax':y+h, 'width':w, 'height':h}
                
            if one_line_per_box:
                datas.append([img_filename, label, box, 1, image_w, image_h])
            else:
                if box_mode != 2: box += [label]
                labels.append(label)
                boxes.append(box)
        
        if not one_line_per_box:
            if len(boxes) > 0 or keep_empty: 
                datas.append([img_filename, labels, boxes, len(boxes), image_w, image_h])
                
    return pd.DataFrame(datas, columns=columns)

@image_dataset_wrapper(
    name    = 'Wider', task = DETECT,
    train   = {'directory' : '{}/Wider_Face', 'subset' : 'train'},
    valid   = {'directory' : '{}/Wider_Face', 'subset' : 'val'}
)
def preprocess_wider_annots(directory, subset, label_name = 'face', keep_invalid = False,
                            add_box_infos = False, ** kwargs):
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
    
    with open(filename, 'r', encoding='utf-8') as fichier_annot:
        lines = fichier_annot.read().split('\n')
    
    i = 0
    metadata = {}
    while i < len(lines):
        if lines[i] == '': break
        
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
            
            box_infos = {
                'blur' : blur, 'expression' : expression, 'invalid' : invalid, 
                'occlusion' : occlusion, 'pose' : pose
            }
            for info_name, value in box_infos.items():
                box_infos[info_name] = box_info_expl[info_name][value]
            
            box = [x, y, w, h]
            
            labels.append(label_name)
            boxes.append(box)
            boxes_infos.append(box_infos)
        
        metadata[img_filename] = {
            'label' : labels,
            BOX_KEY : boxes,
            'category'  : category
        }
        if add_box_infos:
            metadata[img_filename]['box_infos'] = boxes_infos
    
    return metadata
                    
@image_dataset_wrapper(
    name = 'VOC', task = DETECT, directory = '{}/VOC2012',
    annotation_dir = 'Annotations', img_dir = 'JPEGImages'
)
def preprocess_VOC_annots(directory, annotation_dir = 'Annotations', img_dir = 'JPEGImages',
                          ** kwargs):
    """
        Arguments :
            - directory : main directory
            - subset    : the dataset's version (default to VOC2012)
    """
    from utils.image import _image_formats
    
    annotation_dir  = os.path.join(directory, annotation_dir)
    img_dir         = os.path.join(directory, img_dir)
    
    metadata = {}
    for ann in sorted(os.listdir(annotation_dir)):
        img_filename, image_w, image_h = None, None, None

        tree = ET.parse(os.path.join(annotation_dir, ann))
        
        boxes, labels = [], []
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img_filename = os.path.join(img_dir, str(elem.text))
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
            'label' : labels, 'height' : image_h, 'width' : image_w, 'box' : boxes
        }
    
    return metadata

@image_dataset_wrapper(
    name = 'COCO', task = [DETECT, SEGMENT, CAPTION],
    train   = {
        'directory' : '{}/COCO', 'img_dir' : 'train2017',
        'annot_file'    : os.path.join('annotations', 'instances_train2017.json')
    },
    valid   = {
        'directory' : '{}/COCO', 'img_dir' : 'val2017',
        'annot_file'    : os.path.join('annotations', 'instances_val2017.json')
    }
)
def preprocess_COCO_annots(directory,
                           annot_file,
                           img_dir,
                           
                           keep_labels  = True,
                           keep_boxes   = True,
                           keep_caption = True,
                           keep_segmentation    = False,
                           use_supercategory_as_label = False,
                           
                           ** kwargs
                          ):
    img_dir = os.path.join(directory, img_dir) if img_dir else directory
    annot_file  = os.path.join(directory, annot_file)
    
    infos    = load_json(os.path.join(directory, annot_file))
    
    metadata = {}
    for image in infos['images']:
        metadata[image['id']] = {
            'filename'  : os.path.join(img_dir, image['file_name']),
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
                metadata[row['image_id']].setdefault(BOX_KEY, []).append(
                    [int(c) for c in row['bbox']]
                )
            
            if keep_segmentation:
                metadata[row['image_id']].setdefault('segmentation', []).append(row['segmentation'])
    
    return {row['filename'] : {** row, 'id' : k} for k, row in metadata.items()}

@image_dataset_wrapper(
    name = 'COCO_Text', task = [SCENE_TEXT, OCR],
    train   = {
        'directory' : '{}/COCO', 'img_dir' : 'train2017', 'subset' : 'train',
        'annot_file'    : os.path.join('annotations', 'cocotext.v2.json')
    },
    valid   = {
        'directory' : '{}/COCO', 'img_dir' : 'train2017', 'subset' : 'val',
        'annot_file'    : os.path.join('annotations', 'cocotext.v2.json')
    }
)
def preprocess_COCO_text_annots(directory,
                                annot_file,
                                img_dir,
                                subset      = 'train',
                                
                                skip_illegible  = True,
                                default_label   = None,
                                keep_boxes      = True,
                                keep_segmentation   = True,
                                
                                ** kwargs
                               ):
    assert subset in ('train', 'val')

    coco = preprocess_COCO_annots(
        directory   = directory,
        annot_file  = os.path.join(os.path.dirname(annot_file), 'instances_train2017.json'),
        img_dir     = img_dir,
        keep_boxes  = False,
        keep_labels = False,
        keep_caption    = False,
        return_raw      = True
    )
    coco = {info['id'] : info for info in coco.values()}
    
    annot_file  = os.path.join(directory, annot_file)
    
    infos    = load_json(os.path.join(directory, annot_file))
    
    for img_id, ann_ids in infos['imgToAnns'].items():
        if subset and infos['imgs'][img_id]['set'] != subset: continue
        
        annots = [infos['anns'][str(ann_id)] for ann_id in ann_ids]
        if skip_illegible:
            annots = [a for a in annots if a['legibility'] == 'legible']
        
        for ann in annots:
            coco[int(img_id)].setdefault('label', []).append(
                ann['utf8_string'] if not default_label else default_label
            )
            if keep_boxes:
                coco[int(img_id)].setdefault(BOX_KEY, []).append(ann['bbox'])
            if keep_segmentation:
                coco[int(img_id)].setdefault('mask', []).append(ann['mask'])
    
    return {row['filename'] : row for _, row in coco.items() if 'label' in row}

@image_dataset_wrapper(
    name = 'SynthText', task = [SCENE_TEXT, OCR], directory = '{}/SynthText/SynthText'
)
def preprocess_synthtext_annots(directory, tqdm = lambda x: x, ** kwargs):
    from scipy.io import loadmat
    
    from utils.image.box_utils import BoxFormat, convert_box_format
    
    metadata_file = os.path.join(directory, 'gt.mat')
    data = loadmat(metadata_file)
    
    dataset = {}
    for i, (img, boxes, words) in enumerate(zip(tqdm(data['imnames'][0]), data['wordBB'][0], data['txt'][0])):
        filename = os.path.join(directory, img[0])
        
        cleaned  = []
        for w in words:
            for part in w.split('\n'):
                cleaned.extend(part.strip().split())
        
        if len(boxes.shape) == 2: boxes = np.expand_dims(boxes, axis = -1)
        dataset[filename] = {
            'filename'  : filename,
            'box'       : convert_box_format(
                np.transpose(boxes, [2, 1, 0]).astype(np.int32), BoxFormat.XYWH, box_mode = BoxFormat.POLY
            ),
            'nb_box'    : boxes.shape[-1],
            'label'     : cleaned
        }
    
    return dataset

add_dataset(
    'anime_faces', processing_fn = 'image_directory', task = GAN, directory = '{}/anime_faces/images'
)

add_dataset(
    'fungi', processing_fn = 'coco', task = DETECT,
    train   = {'directory' : '{}/Fungi', 'img_dir' : '', 'annot_file' : 'train.json'},
    valid   = {'directory' : '{}/Fungi', 'img_dir' : '', 'annot_file' : 'val.json'}
)

add_dataset(
    'kangaroo', processing_fn = 'voc', task = DETECT,
    directory = '{}/kangaroo-master', annotation_dir = 'annots', img_dir = 'images'
)

add_dataset(
    'raccoon', processing_fn = 'voc', task = DETECT,
    directory = '{}/raccoon-master', annotation_dir = 'annots', img_dir = 'images'
)