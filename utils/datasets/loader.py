# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
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
import enum
import logging
import importlib

from loggers import timer
from ..wrappers import dispatch_wrapper

logger  = logging.getLogger(__name__)

_dataset_dir = os.environ.get(
    'DATASET_DIR', 'D:/datasets' if os.path.exists('D:/datasets') else '/storage'
)

_tasks      = {}
_custom_datasets    = {}

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


@timer(name = 'dataset loading')
def get_dataset(dataset, *, source = None, ** kwargs):
    if isinstance(dataset, (list, tuple)):
        if all(is_custom_dataset(ds) for ds in dataset):
            return load_custom_dataset(dataset, ** kwargs)
        return [get_dataset(ds, source = source, ** kwargs) for ds in dataset]
    
    elif isinstance(dataset, dict):
        return [get_dataset(ds, ** config) for ds, config in dataset.items()]
    
    elif is_custom_dataset(dataset) or source == 'custom':
        return load_custom_dataset(dataset, ** kwargs)
    elif source in ('tensorflow', 'tensorflow_datasets', 'tf', 'tfds'):
        import tensorflow_datasets as tfds
        
        return tfds.load(dataset, ** kwargs)
    elif source == 'keras':
        import keras
        return getattr(keras.datasets, dataset).load_data(** kwargs)
    elif callable(source):
        return source(dataset, ** kwargs)
    else:
        raise ValueError("Unknown dataset {} (source {}) !".format(dataset, source))


def set_dataset_dir(directory):
    global _dataset_dir
    _dataset_dir = directory

def get_dataset_dir(dataset = None):
    global _dataset_dir, _custom_datasets
    if not dataset: return _dataset_dir
    dataset = _clean_dataset_name(dataset)
    return _custom_datasets[dataset].infos.get('directory', '').format(_dataset_dir)

def show_datasets(task = None):
    for t, datasets in _tasks.items():
        if task and t not in task: continue
        logger.info('{} :\t{}'.format(t, tuple(datasets)))

def is_custom_dataset(dataset):
    if isinstance(dataset, (list, tuple)): return [is_custom_dataset(ds) for ds in dataset]
    return _clean_dataset_name(dataset) in _custom_datasets

@dispatch_wrapper(_custom_datasets, 'dataset')
def load_custom_dataset(dataset, *, subset = None, ** kwargs):
    import pandas as pd

    if isinstance(dataset, (list, tuple)):
        datasets = {}
        
        for i, name in enumerate(dataset):
            dataset_i = load_custom_dataset(
                name,
                subset  = subset,
                annotation_type = annotation_type,
                ** kwargs
            )
            if not isinstance(dataset_i, dict): dataset_i = {'train' : dataset_i}
            
            for _set, data in dataset_i.items():
                if 'dataset' not in data.columns: data['dataset'] = name
                datasets.setdefault(_set, []).append(data)
        
        datasets = {
            k : v[0] if len(v) == 1 else pd.concat(v, ignore_index = True, sort = False).dropna(axis = 'columns')
            for k, v in datasets.items()
        }
        
        return datasets if len(datasets) > 1 else list(datasets.values())[0]
    
    cleaned_name = _clean_dataset_name(dataset)
    if cleaned_name not in _custom_datasets:
        raise ValueError("Unknown dataset !\n  Accepted : {}\n  Got : {}".format(
            list(_custom_datasets.keys()), dataset
        ))
    
    process_fn      = _custom_datasets[cleaned_name]
    dataset_infos   = getattr(process_fn, 'infos')
    
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
        
        datasets[prefix] = process_fn(** _format_kwargs({
            'subset'    : prefix,
            ** {** dataset_infos, ** dataset_infos[prefix]},
            ** kwargs,
            ** {k.replace(prefix + '_', '') : v for k, v in kwargs.items() if k.startswith(prefix)}
        }))
    
    return datasets if len(datasets) > 1 else list(datasets.values())[0]

def add_dataset(processing_fn, name = None, task = None):
    if not name: name = processing_fn.dataset
    if not task: task = processing_fn.task
    
    if not isinstance(name, (list, tuple)): name = [name]
    canonical_names = _clean_dataset_name(name)
    
    load_custom_dataset.dispatch(processing_fn, sorted(set(list(name) + canonical_names)))
    
    if not isinstance(task, (list, tuple)): task = [task]
    for t in task: _tasks.setdefault(t, []).append(name[0])

def _format_kwargs(kwargs):
    dataset_dir = get_dataset_dir()
    return {k : v.format(dataset_dir) if isinstance(v, str) else v for k, v in kwargs.items()}

def _clean_dataset_name(name):
    if isinstance(name, (list, tuple)): return [_clean_dataset_name(n) for n in name]
    return ''.join([c for c in name.lower() if c.isalnum()])

for ds_module in os.listdir(os.path.dirname(__file__)):
    if not ds_module.endswith('_datasets') or ds_module.startswith('_'): continue
    for module in os.listdir(os.path.join(os.path.dirname(__file__), ds_module)):
        if module.startswith(('_', '.')) or '_old' in module: continue
        module = importlib.import_module('{}.{}.{}'.format(
            __package__, ds_module, module[:-3]
        ))
        for key in dir(module):
            fn = getattr(module, key)
            if hasattr(fn, 'dataset') and hasattr(fn, 'task'):
                add_dataset(fn)
