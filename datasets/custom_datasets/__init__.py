
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
import logging
import pandas as pd

from sklearn.utils import shuffle as sklearn_shuffle

from datasets.custom_datasets.preprocessing import *

logger  = logging.getLogger(__name__)

_dataset_dir = os.environ.get(
    'DATASET_DIR', 'D:/datasets' if os.path.exists('D:/datasets') else '/storage'
)

def __load():
    for module_name in os.listdir('datasets/custom_datasets'):
        if module_name in ['__init__.py', '__pytache__']: continue
        module_name = 'datasets.custom_datasets.' + module_name.replace('.py', '')

        module = __import__(module_name)

def add_dataset(name, processing_fn, task, ** kwargs):
    global _custom_datasets, _custom_processing, _dataset_tasks
    
    name = name.lower().replace(' ', '_')
    
    if kwargs:
        _custom_datasets[name] = kwargs
    if processing_fn is not None:
        if isinstance(processing_fn, str): processing_fn = _custom_processing[processing_fn]
        _custom_processing[name]    = processing_fn
    
    if not isinstance(task, (list, tuple)): task = [task]
    for t in task: _dataset_tasks.setdefault(t, []).append(name)

def set_dataset_dir(dataset_dir):
    global _dataset_dir
    _dataset_dir = dataset_dir

def get_dataset_dir():
    global _dataset_dir
    return _dataset_dir

def load_dataset(ds_name,
                 dataset_dir    = None,
                 type_annots    = None,

                 modes  = ['train', 'valid'],
                 train_kw   = {},
                 valid_kw   = {}, 

                 size   = None,
                 shuffle    = False,
                 random_state   = 10,
                 ** kwargs
                ):
    def format_kwargs(kwargs):
        return {k : v.format(dataset_dir) if isinstance(v, str) else v for k, v in kwargs.items()}
    
    def load_and_shuffle(config):
        data = process_fn(** config)
        if shuffle: data = sklearn_shuffle(data, random_state = random_state)
        if size:    data = data.iloc[:size]
        return data
        
    if dataset_dir is None: dataset_dir = get_dataset_dir()
    
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
                datasets.setdefault(mode, []).append(data)
        
        for mode, datas in datasets.keys():
            dataset = pd.concat(datas, ignore_index = True, sort = False).dropna(axis = 'columns')
            if shuffle: dataset = sklearn_shuffle(dataset, random_state = random_state)
            datasets[mode] = dataset
        
        return datasets if len(datasets) > 1 else list(datasets.values())[0]
    
    if ds_name not in _custom_datasets and type_annots is None:
        raise ValueError("Unknown dataset !\n  Got : {}\n  Accepted : {}".format(
            ds_name, list(_custom_datasets.keys())
        ))
    
    if type_annots is None:
        type_annots = _custom_datasets[ds_name].get('type_annots', ds_name)
    
    if type_annots not in _custom_processing:
        raise ValueError("Unknown annotation type !\n  Got : {}\n  Accepted : {}".format(
            type_annots, list(_custom_processing.keys())
        ))
        
    process_fn = _custom_processing[type_annots]
    
    logger.info('Loading dataset {}...'.format(ds_name))
    
    if 'train'  not in _custom_datasets.get(ds_name, {}):
        datasets = load_and_shuffle(format_kwargs({**_custom_datasets.get(ds_name, {}), ** kwargs}))
    else:
        datasets = {}
        
        if 'train' in modes:
            datasets['train'] = load_and_shuffle(format_kwargs({
                ** _custom_datasets[ds_name]['train'], ** kwargs, ** train_kw
            }))

        if 'valid' in modes:
            datasets['valid'] = load_and_shuffle(format_kwargs({
                ** _custom_datasets[ds_name]['valid'], ** kwargs, ** valid_kw
            }))

        if len(datasets) == 1: datasets = list(datasets.values())[0]
    
    return datasets

def show_datasets(task = None):
    global _dataset_tasks
    
    for t, datasets in _dataset_tasks.items():
        if task and t not in tasks: continue
        print('Task {} :\t{}'.format(t, tuple(datasets)))

_custom_processing  = {}
_custom_datasets    = {}
_dataset_tasks      = {}

__load()