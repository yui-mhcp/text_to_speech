
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

import logging
import pandas as pd

from sklearn.utils import shuffle as sklearn_shuffle

from datasets.custom_datasets.audio_datasets import _custom_audio_datasets
from datasets.custom_datasets.audio_datasets import _audio_dataset_processing

from datasets.custom_datasets.image_datasets import _custom_image_datasets
from datasets.custom_datasets.image_datasets import _image_dataset_processing

from datasets.custom_datasets.text_datasets import _custom_text_datasets
from datasets.custom_datasets.text_datasets import _text_dataset_processing

from datasets.custom_datasets.preprocessing import *

logger  = logging.getLogger(__name__)

_dataset_dir = os.environ.get('DATASET_DIR', 'D:/datasets')

def set_dataset_dir(dataset_dir):
    global _dataset_dir
    _dataset_dir = dataset_dir

def load_dataset(ds_name, dataset_dir = None, type_annots = None,
                 modes = ['train', 'valid'], train_kw = {}, valid_kw = {}, 
                 size = None, shuffle = False, random_state = 10, ** kwargs):
    def format_kwargs(kwargs):
        formatted = {}
        for k, v in kwargs.items():
            formatted[k] = v.format(dataset_dir) if type(v) is str else v
        return formatted
    global _dataset_dir
    if dataset_dir is None: dataset_dir = _dataset_dir
    
    if isinstance(ds_name, (list, tuple)):
        datasets = {}
        
        for i, name in enumerate(ds_name):
            config = {
                'type_annots'   : type_annots,
                'dataset_dir'   : dataset_dir,
                'modes'         : modes,
                'size'          : size[i] if isinstance(size, (list, tuple)) else size,
                'shuffle'       : shuffle,
                'random_state'  : random_state,
                ** kwargs
            }
            
            if isinstance(name, dict):
                config = {** config, ** name}
            else:
                config['ds_name'] = name
            
            dataset = load_dataset(** config)
            if not isinstance(dataset, dict): dataset = {'train' : dataset}
            
            for mode, data in dataset.items():
                data['dataset'] = name if not isinstance(name, dict) else name['ds_name']
                datasets.setdefault(mode, [])
                datasets[mode].append(data)
        
        for mode in datasets.keys():
            dataset = pd.concat(datasets[mode], ignore_index = True).dropna(axis = 'columns')
            if shuffle: dataset = sklearn_shuffle(dataset, random_state = random_state)
            datasets[mode] = dataset
        
        return datasets if len(datasets) > 1 else list(datasets.values())[0]
    
    if ds_name not in _custom_datasets and type_annots is None:
        raise ValueError("Unknown dataset !\n  Got : {}\n  Accepted : {}".format(ds_name, list(_custom_datasets.keys())))
    
    if type_annots is None:
        type_annots = _custom_datasets[ds_name].get('type_annots', ds_name)
    
    if type_annots not in _custom_processing:
        raise ValueError("Annotation type unknown !\n  Got : {}\n  Accepted : {}".format(type_annots, list(_custom_processing.keys())))
        
    process_fn = _custom_processing[type_annots]
    
    logger.info('Loading dataset {}...'.format(ds_name))
    
    if 'train'  not in _custom_datasets.get(ds_name, {}):
        ds_kwargs = {**_custom_datasets.get(ds_name, {}), ** kwargs}
        ds_kwargs = format_kwargs(ds_kwargs)

        datasets = process_fn(** ds_kwargs)
        
        if shuffle: datasets = sklearn_shuffle(datasets, random_state = random_state)
        if size: datasets = datasets[:size]
    else:
        datasets = {}
        
        if 'train' in modes:
            train_kwargs = {** _custom_datasets[ds_name]['train'], ** kwargs, ** train_kw}
            train_kwargs = format_kwargs(train_kwargs)

            train_dataset = process_fn(** train_kwargs)
            if shuffle: train_dataset = sklearn_shuffle(train_dataset, random_state = random_state)
            if size: train_dataset = train_dataset[:size]
            
            datasets['train'] = train_dataset
            
        if 'valid' in modes:
            valid_kwargs = {** _custom_datasets[ds_name]['valid'], ** kwargs, ** valid_kw}
            valid_kwargs = format_kwargs(valid_kwargs)

            valid_dataset = process_fn(** valid_kwargs)
            if shuffle: valid_dataset = sklearn_shuffle(valid_dataset, random_state = random_state)
            if size: valid_dataset = valid_dataset[:size]
            
            datasets['valid'] = valid_dataset
        
        if len(datasets) == 1: datasets = list(datasets.values())[0]
    
    return datasets

_custom_datasets = {
    ** _custom_audio_datasets,
    ** _custom_image_datasets,
    ** _custom_text_datasets
}

_custom_processing  = {
    ** _audio_dataset_processing,
    ** _image_dataset_processing,
    ** _text_dataset_processing
}