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
import enum
import logging
import importlib
import pandas as pd

from sklearn.utils import shuffle as sklearn_shuffle

from .preprocessing import *
from utils.wrapper_utils import dispatch_wrapper

logger  = logging.getLogger(__name__)

_dataset_dir = os.environ.get(
    'DATASET_DIR', 'D:/datasets' if os.path.exists('D:/datasets') else '/storage'
)

_tasks      = {}
_custom_datasets    = {}
_custom_processing  = {}

class Task(enum.Enum):
    TEXT_DETECTION      = 'text detection'
    OBJECT_DETECTION    = 'object detection'
    OBJECT_SEGMENTATION = 'object segmentation'
    FACE_RECOGNITION    = 'face recognition'
    
    OCR     = 'OCR'
    IMAGE_CAPTIONING    = 'image captioning'
    
    TTS     = 'Text To Speech'
    STT     = 'Speech To Text'
    SI      = 'Speaker Identification'
    
    QA      = 'Question Answering (Q&A)'
    

def set_dataset_dir(directory):
    global _dataset_dir
    _dataset_dir = directory

def get_dataset_dir(dataset = None):
    global _dataset_dir, _custom_datasets
    if not dataset: return _dataset_dir
    return _custom_datasets.get(clean_dataset_name(dataset), {}).get('directory', None)

def clean_dataset_name(name):
    if isinstance(name, (list, tuple)): return [clean_dataset_name(n) for n in name]
    return ''.join([c for c in name.lower() if c.isalnum()])

def add_dataset(name, processing_fn, task, ** kwargs):
    if not isinstance(name, (list, tuple)): name = [name]
    canonical_name = clean_dataset_name(name)
    
    if isinstance(processing_fn, str):
        processing_fn = _custom_processing.get(processing_fn, processing_fn)
    load_custom_dataset.dispatch(processing_fn, sorted(set(list(name) + canonical_name)))
    
    if kwargs:
        for n in canonical_name:
            _custom_datasets[n]    = kwargs
    
    if not isinstance(task, (list, tuple)): task = [task]
    for t in task: _tasks.setdefault(t, []).append(name[0])

def is_custom_dataset(dataset):
    if isinstance(dataset, (list, tuple)): return [is_custom_dataset(ds) for ds in dataset]
    return clean_dataset_name(dataset) in _custom_datasets

def show_custom_datasets(task = None):
    for t, datasets in _tasks.items():
        if task and t not in task: continue
        logger.info('{} :\t{}'.format(t, tuple(datasets)))

@dispatch_wrapper(_custom_processing, 'dataset')
def load_custom_dataset(dataset,
                        *,

                        subset = None,
                        annotation_type    = None,

                        shuffle    = False,
                        random_state   = 10,
                 
                        ** kwargs
                       ):
    def format_kwargs(kwargs):
        return {k : v.format(dataset_dir) if isinstance(v, str) else v for k, v in kwargs.items()}

    if isinstance(dataset, (list, tuple)):
        datasets = {}
        
        for i, name in enumerate(dataset):
            dataset_i = load_custom_dataset(
                name,
                subset  = subset,
                annotation_type = annotation_type,
                random_state    = random_state,
                shuffle = shuffle,
                ** kwargs
            )
            if not isinstance(dataset_i, dict): dataset_i = {'train' : dataset_i}
            
            for _set, data in dataset_i.items():
                if 'dataset' not in data.columns: data['dataset'] = name
                datasets.setdefault(_set, []).append(data)
        
        for subset, data in datasets.items():
            data = pd.concat(data, ignore_index = True, sort = False).dropna(axis = 'columns')
            if shuffle: data = sklearn_shuffle(data, random_state = random_state)
            datasets[subset] = data
        
        return datasets if len(datasets) > 1 else list(datasets.values())[0]
    
    cleaned_name = clean_dataset_name(dataset)
    if cleaned_name not in _custom_datasets and annotation_type is None:
        raise ValueError("Unknown dataset !\n  Got : {}\n  Accepted : {}".format(
            dataset, list(_custom_datasets.keys())
        ))
    
    if annotation_type is None:
        annotation_type = _custom_datasets[cleaned_name].get('annotation', cleaned_name)
    else:
        annotation_type = clean_dataset_name(annotation_type)
    
    if annotation_type not in _custom_processing:
        raise ValueError("Unknown annotation type !\n  Got : {}\n  Accepted : {}".format(
            type_annots, list(_custom_processing.keys())
        ))
    
    dataset_dir = get_dataset_dir()
    
    process_fn      = _custom_processing[annotation_type]
    dataset_infos   = _custom_datasets.get(cleaned_name, {})
    
    logger.info('Loading dataset {}...'.format(dataset))
    
    if 'train' not in dataset_infos:
        if subset:
            logger.warning('The dataset {} has no predefined subsets. The function will return the complete dataset'.format(dataset))
        dataset_infos, subset = {'train' : dataset_infos}, ['train']
    
    if subset is None:      subset = ('train', 'valid', 'test')
    elif not isinstance(subset, (list, tuple)): subset = (subset, )
    
    datasets = {}
    for prefix in subset:
        if prefix not in dataset_infos: continue
        
        dataset = process_fn(** format_kwargs({
            'subset'    : prefix,
            ** {** dataset_infos, ** dataset_infos[prefix]},
            ** kwargs,
            ** {k.replace(prefix + '_', '') : v for k, v in kwargs.items() if k.startswith(prefix)}
        }))
        if shuffle: dataset = sklearn_shuffle(dataset, random_state = random_state)
        datasets[prefix] = dataset
    
    return datasets if len(datasets) > 1 else list(datasets.values())[0]

def maybe_load_embedding(directory, dataset, ** kwargs):
    if 'embedding' in dataset.columns: return dataset
    if any(k.startswith('embedding_') for k in kwargs.keys()):
        from utils.embeddings import load_embeddings
        return load_embeddings(directory, dataset = dataset, ** kwargs)
    return dataset

for module in os.listdir(__path__[0]):
    if module.startswith(('_', '.')) or '_old' in module: continue
    importlib.import_module(__package__ + '.' + module.replace('.py', ''))
