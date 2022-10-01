
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
from PIL import Image
from multiprocessing import cpu_count

from loggers import timer
from utils.thread_utils import Consumer
from utils.file_utils import load_json, dump_json

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

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

@timer(name = 'image dir loading')
def preprocess_image_directory(directory, ** kwargs):
    from utils.image import _image_formats
    
    metadata = []
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            metadata += [
                {'filename' : os.path.join(directory, filename, f), 'label' : filename}
                for f in os.listdir(os.path.join(directory, filename))
            ]
        elif filename.endswith(_image_formats):
            metadata.append({'filename' : os.path.join(directory, filename)})
    
    return pd.DataFrame(metadata)
            

@timer(name = 'yolo output loading')
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
    
    dataset = []
    for metadata_file in os.listdir(directory):
        if not metadata_file.endswith('_boxes.json'): continue
        
        sub_img_dir = metadata_file[:-11] # remove _boxes.json
        
        metadata = load_json(os.path.join(directory, metadata_file))
        
        metadata = [
            {'filename' : os.path.join(img_dir, sub_img_dir, img_name), ** infos}
            for img_name, infos in metadata.items()
        ]
        metadata = [m for m in metadata if os.path.exists(m['filename'])]
        for data in metadata:
            data['label'] = [box[-1] for box in data['boxes']]
            data['boxes']  = [box[:4] for box in data['boxes']]
            
            image = Image.open(data['filename'])
            w, h = image.size
            data.update({'height' : h, 'width' : w})
        
        dataset.extend(metadata)
    
    dataset = pd.DataFrame(dataset)
    dataset = dataset.rename(columns = {'boxes' : 'box'})
    dataset['nb_box'] = dataset['box'].apply(len)
    
    return dataset

@timer(name = 'essex loading')
def preprocess_essex_annots(annotation_dir, box_filename=None, box_mode=0, 
                            one_line_per_box=True, accepted_labels=None, 
                            labels_substituts=None, ** kwargs):
    """
        Retourne une DataFrame avec les colonnes suivantes : 
            - label     : l'identifiant de la personne. 
            - chemin    : le chemin complet d'accès à l'image. 
            - difficulte    : la difficulté de classification (1 à 4)
            - sexe      : le sexe de la personne (si mentionné 'M', 'F', None)
            - width     : largeur de l'image
            - height    : hauteur de l'image
    """
    def process_dir(dir_path, difficulte, sexe):
        metadata    = []
        for identifiant in os.listdir(dir_path):
            label = identifiant
            if labels_substituts is not None:
                if label in labels_substituts: 
                    label = labels_substituts[label]
                elif '*' in labels_substituts:
                    label = labels_substituts['*']
                    
            if accepted_labels is not None and label not in accepted_labels:
                continue
                    
            for img_name in os.listdir(os.path.join(dir_path, identifiant)):
                if '.jpg' not in img_name: continue
                image_path = os.path.join(dir_path, identifiant, img_name)
                image = Image.open(image_path)
                image_w, image_h = image.size
                
                metadata.append([image_path, label, difficulte, sexe, image_w, image_h])
        return metadata
                
    columns = ["filename", "label", "difficulte", "sexe", "width", "height"]
    folders = [
        [os.path.join("faces94", "male"), 1, 'M'],
        [os.path.join("faces94", "female"), 1, 'F'],
        [os.path.join("faces94", "malestaff"), 1, None],
        ["faces95", 2, None],
        ["faces96", 3, None],
        ["grimace", 4, None]
    ]
    metadata = []
    
    for folder, difficulte, sexe in folders:
        data = process_dir(os.path.join(annotation_dir, folder), difficulte, sexe)
        
        metadata += data
                    
    metadata_df = pd.DataFrame(metadata, columns=columns)
    
    if box_filename is not None:
        boxes_df = preprocess_yolo_output(os.path.join(chemin, box_filename), 
                                          image_path        = None, 
                                          box_mode          = box_mode, 
                                          one_line_per_box  = one_line_per_box
                                         )
        
        metadata_df['box'] = [None] * len(metadata_df)
        metadata_df['n_box'] = [0] * len(metadata_df)
        for idx, row in metadata_df.iterrows():
            box = boxes_df[boxes_df['filename'] == row['filename']].reset_index()
            if len(box) == 0: continue
            box = box.at[0, 'box']
            metadata_df.at[idx, 'box'] = box
            metadata_df.at[idx, 'n_box'] = len(box) if not one_line_per_box else 1
            
    return metadata_df

@timer(name = 'CelebA loading')
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
        
        metadata[img] = {'filename' : os.path.join(directory, img_dir, img), 'id' : celeb_id}
    
    with open(os.path.join(directory, 'list_eval_partition.txt'), 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')
    
    eval_split = {}
    for line in lines:
        if len(line) == 0: continue
        img, mode = line.split()
        
        eval_split[img] = mode
    
    metadata = [v for k, v in metadata.items() if eval_split.get(k, -1) == subset]

    return pd.DataFrame(metadata)

@timer(name = 'TDFace loading')
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
    if isinstance(acquisition, (list, tuple)):
        return pd.concat([
            preprocess_td_face_annots(directory, subsets = subsets, acquisition = acq, ** kwargs)
            for acq in acquisition
        ], axis = 0, ignore_index = True)
    if isinstance(subsets, (list, tuple)):
        return pd.concat([
            preprocess_td_face_annots(directory, subsets = s, acquisition = acquisition, ** kwargs)
            for s in subsets
        ], axis = 0, ignore_index = True)
    
    set_dir = os.path.join(directory, 'TD_{}_Set{}'.format(acquisition, subsets))
    metadata = preprocess_image_directory(set_dir)
    metadata = metadata.rename(columns = {'label' : 'id'})
    
    if 'E' in acquisition:
        metadata['expression'] = metadata['filename'].apply(
            lambda f: f[:-4].split('_')[-1]
        )
    elif 'A' in acquisition:
        metadata['position'] = metadata['filename'].apply(
            lambda f: f[:-4].split('_')[-1]
        )
        metadata['expression'] = metadata['filename'].apply(
            lambda f: f.split('_')[-2]
        )

    
    return metadata

@timer(name = 'Youtube Faces with keypoints loading')
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

        infos = []
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
    
    def convert_bbox_format(box):
        x1, y1, x2, y2 = [int(c) for c in box]
        return np.array([x1, y1, x2 - x1, y2 - y1])
    
    if not isinstance(subsets, (list, tuple)): subsets = [subsets]
    
    extracted_dir = os.path.join(directory, 'extracted')
    os.makedirs(extracted_dir, exist_ok = True)

    infos = pd.read_csv('{}/youtube_faces_with_keypoints_full.csv'.format(directory))
    
    metadata = []
    for idx, row in tqdm(infos.iterrows()):
        frames = extract_frames(
            row['videoID'], person_id = row['personName'], height = row['imageHeight'], width = row['imageWidth']
        )
        step = skip_frames
        if step <= 1 and frames_per_video > 0: step = max(1, len(frames) // frames_per_video)
        if step > 1: frames = frames[::step]
        if frames_per_video > 0: frames = frames[:frames_per_video]
        metadata.extend(frames)
    
    metadata = pd.DataFrame(metadata)
    metadata = metadata[metadata['subset'].isin(subsets)]
    metadata['bbox'] = metadata['bbox'].apply(lambda b: convert_bbox_format(b))
    
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
            
        image = Image.open(img_filename)
        image_w, image_h = image.size
        
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

@timer(name = 'wider faces loading')
def preprocess_wider_annots(filename, img_dir, one_line_per_box = False,
                            box_as_dict = False, label_name = 'face', keep_empty = False,
                            min_box_per_image = 0, max_box_per_image = 1000, 
                            keep_invalid = False, rectangular_boxes = False,
                            with_infos = False, ** kwargs):
    """
        Arguments :
            - filename  : the annotation filename
            - img_dir   : directory where images are stored
            - one_line_per_box  : whether to group images for a single image or not
            - box_as_dict       : whether to put boxes as [x, y, w, h] or dict
            - label_name        : all are faces but you can specify another name
            - {min / max}_box_per_image : min / max boxes perimage (other are skipped)
            - rectangular_boxes : whether to padd boxes (necessary for tf.data.Dataset)
            - with_infos        : whether to include 'box_infos' column or not
        Return :
            - pd.DataFrame with columns :
                [filename, height, width, nb_box, box, label, category, box_infos]
    
        Annotation format :
            image_filename
            nb_box
            x, y, w, h, blur, expression, illumination, invalid, occlusion, pose = infos
            x, y, w, h, blur, expression, illumination, invalid, occlusion, pose = infos
            ...
    """
    box_info_expl = {
        "blur"          : {0: "clear", 1: "normal", 2: "heavy"},
        "expression"    : {0: "typical", 1: "exagerate"},
        "illumination"  : {0: "normal", 1: "extreme"},
        "occlusion"     : {0: "no", 1: "partial", 2: "heavy"},
        "pose"          : {0: "typical", 1: "atypical"},
        "invalid"       : {0: "false", 1: "true"}
    }
    
    with open(filename, 'r', encoding='utf-8') as fichier_annot:
        lines = fichier_annot.read().split('\n')
    
    i = 0
    metadata = []
    while i < len(lines):
        if lines[i] == '': break
        
        img_filename    = os.path.join(img_dir, lines[i])
        category    = lines[i].split('/')[0]
        nb_box      = int(lines[i + 1])
        
        i += 2
        if nb_box > max_box_per_image or nb_box < min_box_per_image:
            i += nb_box
            continue
        
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
            
            box = [x, y, w, h] if not box_as_dict else {
                'xmin' : x, 'ymin' : y, 'xmax' : x + w, 'ymax' : y + h,
                'label' : label_name, 'width' : w, 'height' : h
            }
                
            if one_line_per_box:
                metadata.append({
                    'filename' : img_filename, 'label' : label_name, 'box' : box,
                    'nb_box' : 1, 'category' : category, ** box_infos
                })
            else:
                labels.append(label_name)
                boxes.append(box)
                boxes_infos.append(box_infos)
        
        if not one_line_per_box:
            if len(boxes) > 0 or keep_empty:
                metadata.append({
                    'filename' : img_filename, 'label' : labels, 'box' : boxes,
                    'nb_box' : len(boxes), 'category' : category
                })
                if with_infos: metadata[-1]['box_infos'] = boxes_infos
                
    dataset = pd.DataFrame(metadata)
    
    dataset = _add_image_size(dataset)
    
    if not box_as_dict and rectangular_boxes:
        dataset = _rectangularize_boxes(dataset)
    
    return dataset
                    
@timer(name = 'pascal voc loading')
def preprocess_VOC_annots(annotation_dir, img_dir, box_as_dict = False, 
                          one_line_per_box = False, accepted_labels = None,
                          aliases = None, keep_empty = False, 
                          rectangular_boxes = False, ** kwargs):
    """
        Arguments :
            - annotation_dir    : directory of annotations' files
            - img_dir           : images' directory
            - box_as_dict       : whether to put box as [x, y, w, h] or dict
            - one_line_per_box  : whether to group boxes for a same image or not
            - accepted_labels   : accepted labels (other are skipped)
            - aliases       : dict containing new name for labels
            - keep_empty        : whether to keep images with 0 box
            - rectangular_boxes : whether to padd boxes (necessary for tf.data.Dataset)
        Return :
            - pd.DataFrame with columns :
                [filename, height, width, nb_box, box, label]
    """
    metadata = []
    for ann in sorted(os.listdir(annotation_dir)):
        img_filename, image_w, image_h = None, None, None

        tree = ET.parse(os.path.join(annotation_dir, ann))
        
        boxes, labels = [], []
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img_filename = os.path.join(img_dir, str(elem.text))
                if not img_filename.endswith('.jpg'): img_filename += '.jpg'
            if 'width' in elem.tag:     image_w = int(elem.text)
            if 'height' in elem.tag:    image_h = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                for attr in list(elem):
                    if 'name' in attr.tag:
                        label = str(attr.text)
                        if aliases is not None:
                            if label in aliases:
                                label = aliases[label]
                            elif '*' in aliases:
                                label = aliases['*']
                        
                        if accepted_labels is not None and label not in accepted_labels:
                            break
                        
                    if 'bndbox' in attr.tag:
                        x0, y0, x1, y1 = 0, 0, 0, 0
                        for dim in list(attr):
                            if 'xmin' in dim.tag: x0 = int(round(float(dim.text)))
                            if 'ymin' in dim.tag: y0 = int(round(float(dim.text)))
                            if 'xmax' in dim.tag: x1 = int(round(float(dim.text)))
                            if 'ymax' in dim.tag: y1 = int(round(float(dim.text)))
                        
                        box = [x0, y0, x1 - x0, y1 - y0] if not box_as_dict else {
                            'xmin' : x0, 'xmax' : x1, 'ymin' : y0, 'ymax' : y1,
                            'width' : x1 - x0, 'height' : y1 - y0, 'label' : label
                        }
                        if one_line_per_box:
                            metadata.append({
                                'filename' : img_filename, 'nb_box' : 1, 'box' : box, 
                                'label' : label, 'height' : image_h, 'width' : image_w
                            })
                        else:
                            labels.append(label)
                            boxes.append(box)

        if not one_line_per_box:
            if len(boxes) > 0 or keep_empty:
                metadata.append({
                    'filename' : img_filename, 'label' : labels, 'box' : boxes,
                    'nb_box' : len(boxes), 'height' : image_h, 'width' : image_w
                })

    dataset = pd.DataFrame(metadata)

    if not box_as_dict and rectangular_boxes:
        dataset = _rectangularize_boxes(dataset)
    
    return dataset

@timer(name = 'coco loading')
def preprocess_COCO_annots(directory,
                           subset,
                           
                           keep_empty = False,
                           accepted_labels = None,
                           use_supercategory_as_label = False,
                           
                           box_mode = 0,
                           one_line_per_box  = False,
                           keep_segmentation = False,
                           
                           one_line_per_caption = False,
                           
                           columns_to_drop = [
                               "coco_url", "flickr_url", "rights_holder",
                               "iscrowd", "license", "id_x", "id_y", "date_captured"
                           ],
                           replace_name    = {
                               'name' : 'category', 'file_name' : 'filename', 'bbox' : 'box'
                           },
                           ** kwargs
                          ):
    from utils.image.box_utils import convert_bbox
    assert box_mode in (0, 1, 2)
    assert not (one_line_per_box and one_line_per_caption)
    
    if not keep_segmentation: columns_to_drop.append('segmentation')
    columns_to_drop   = set(columns_to_drop)
    columns_to_concat = set(['area', 'box', 'category_id', 'supercategory', 'category', 'label'])
    
    @timer
    def update_row(row):
        label = row['name'] if not use_supercategory_as_label else row['supercategory']
        if accepted_labels and label not in accepted_labels:
            label = ''

        if 'bbox' in row:
            row['bbox'] = convert_bbox(
                * [int(c) for c in row['bbox']], label = label, mode = box_mode
            )
        
        row.update({'n_box' : 1, 'label' : label, 'filename' : os.path.join(img_dir, row.pop('file_name'))})
        
        return {replace_name.get(k, k) : v for k, v in row.items() if k not in columns_to_drop}
    
    time_logger.start_timer('loading')

    img_dir  = os.path.join(directory, subset)
    
    infos    = load_json(os.path.join(directory, 'annotations', 'instances_{}.json'.format(subset)))
    captions = load_json(os.path.join(directory, 'annotations', 'captions_{}.json'.format(subset)))

    time_logger.stop_timer('loading')
    time_logger.start_timer('merging')
    
    cap = {}
    for row in captions['annotations']:
        cap.setdefault(row['image_id'], []).append(row['caption'])
    
    captions    = cap
    categories  = {row['id'] : row for row in infos['categories']}
    images      = {row['id'] : {** row, 'text' : cap[row['id']]} for row in infos['images']}
    
    metadata    = [
        update_row({** row, ** categories[row['category_id']], ** images[row['image_id']]})
        for row in infos['annotations']
    ]
    if not keep_empty:
        metadata = [row for row in metadata if row['label']]
    
    time_logger.stop_timer('merging')

    if not one_line_per_box:
        time_logger.start_timer('flattening')

        new_metadata = {}
        for row in metadata:
            image_id = row['filename']
            if image_id not in new_metadata:
                new_metadata[image_id] = {
                    k : v if k not in columns_to_concat else [v]
                    for k, v in row.items()
                }
            else:
                for col_name in columns_to_concat:
                    new_metadata[image_id][col_name] += [row[col_name]]
                new_metadata[image_id]['n_box'] += 1
        
        metadata = [{'filename' : k, ** v} for k, v in new_metadata.items()]

        time_logger.stop_timer('flattening')
    
    if one_line_per_caption:
        time_logger.start_timer('extending captions')
        
        new_metadata = []
        for row in metadata:
            new_metadata.extend([
                {** row, 'text' : caption} for caption in row['text']
            ])
        
        metadata = new_metadata
        
        time_logger.stop_timer('extending captions')
    
    return pd.DataFrame(metadata)

_custom_image_datasets = {
    'anime_faces'   : {
        'type_annots'   : 'directory',
        'directory'     : '{}/anime_faces/images'
    },
    'celeba'   : {
        'train' : {
            'directory'  : '{}/CelebA', 'img_dir' : 'img_align_celeba',  'subset' : 'train'
        },
        'valid' : {
            'directory'  : '{}/CelebA', 'img_dir' : 'img_align_celeba',  'subset' : 'valid'
        }
    },
    'coco'      : {
        'train' : {
            'directory' : '{}/COCO', 'subset'    : 'train2017'
        },
        'valid' : {
            'directory' : '{}/COCO', 'subset'    : 'val2017'
        }
    },
    'essex'     : {
        'annotation_dir' : '{}/faces_essex'
    },
    'fungi'     : {
        'type_annots'   : 'coco',
        'train' : {
            'filename'  : '{}/Fungi/train.json', 
            'img_dir'   : '{}/Fungi'
        },
        'valid' : {
            'filename'  : '{}/Fungi/val.json', 
            'img_dir'   : '{}/Fungi'
        }
    },
    'kangaroo'  : {
        'type_annots'   : 'voc',
        'annotation_dir'    : '{}/kangaroo-master/annots',
        'img_dir'       : '{}/kangaroo-master/images'
    },
    'raccoon'   : {
        'type_annots'   : 'voc',
        'annotation_dir'    : '{}/raccoon-master/annots',
        'img_dir'       : '{}/raccoon-master/images'
    },
    'td_face'   : {
        'train' : {
            'directory'  : '{}/TDFace', 'acquisition' : 'RGB_A', 'subsets' : [1, 2, 3]
        },
        'valid' : {
            'directory'  : '{}/TDFace', 'acquisition' : 'RGB_A', 'subsets' : 4
        }
    },
    'voc'       : {
        'annotation_dir'   : '{}/VOC2012/Annotations',
        'img_dir'   : '{}/VOC2012/JPEGImages'
    },
    'wider'     : {
        'train' : {
            'filename'  : '{}/Wider_Face/wider_face_split/wider_face_train_bbx_gt.txt', 
            'img_dir'   : '{}/Wider_Face/WIDER_train/images'
        },
        'valid' : {
            'filename'  : '{}/Wider_Face/wider_face_split/wider_face_val_bbx_gt.txt', 
            'img_dir'   : '{}/Wider_Face/WIDER_val/images'
        }
    },
    'yolo_output'   : {},
    'youtube_faces' : {
        'train' : {
            'directory'  : '{}/YoutubeFaces', 'subsets' : [1, 2, 3]
        },
        'valid' : {
            'directory'  : '{}/YoutubeFaces', 'subsets' :  4
        }
    }
}

_image_dataset_processing  = {
    'img_annots'    : preprocess_annotation_annots,
    'coco'          : preprocess_COCO_annots,
    'celeba'        : preprocess_celeba_annots,
    'directory'     : preprocess_image_directory,
    'essex'         : preprocess_essex_annots,
    'td_face'       : preprocess_td_face_annots,
    'voc'           : preprocess_VOC_annots,
    'wider'         : preprocess_wider_annots,
    'yolo_output'   : preprocess_yolo_output_annots,
    'youtube_faces' : preprocess_youtube_faces_annots
}