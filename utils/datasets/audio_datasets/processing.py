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
import logging

from functools import wraps
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from loggers import timer

logger = logging.getLogger(__name__)

def audio_dataset_wrapper(name, task, ** infos):
    def wrapper(dataset_loader):
        @timer(name = '{} loading'.format(name))
        @wraps(dataset_loader)
        def _load_and_process(directory, * args, rate = None, ** kwargs):
            import pandas as pd
            
            dataset = dataset_loader(directory, * args, ** kwargs)
            if isinstance(dataset, tuple):  directory, dataset = dataset
            if isinstance(dataset, list):   dataset = pd.DataFrame(dataset)
            
            dataset = _add_resampled_dirs(directory, dataset)
            dataset = _add_default_rate(dataset)

            if rate:
                dataset = resample_dataset(dataset, directory, rate, ** kwargs)
            
            if 'id' not in dataset.columns: dataset['id'] = name
            
            return dataset
        
        
        _load_and_process.task    = task
        _load_and_process.infos   = infos
        _load_and_process.dataset = name
        
        return _load_and_process
    return wrapper

def resample_dataset(dataset,
                     directory,
                     rate,
                     *,
                     
                     tqdm = lambda x: x,
                     max_workers = cpu_count(),

                     ** kwargs
                    ):
    from ...audio import resample_file
    
    new_col = 'wavs_{}'.format(rate)
    if new_col not in dataset.columns:
        dataset[new_col] = dataset['filename'].apply(
            lambda f: _replace_dir(f, new_col, prefix = directory).rpartition('.')[0] + '.wav'
        )
        to_process = dataset
    else:
        processed   = dataset[new_col].apply(os.path.exists)
        to_process  = dataset[~processed]

        logger.info("Resampling dataset to {} Hz\n  {} files already processed\n  {} files to process".format(
            rate, processed.sum(), len(to_process)
        ))
    
    if len(to_process):
        with ThreadPool(max_workers) as pool:
            results = [
                pool.apply_async(resample_file, (row['filename'], rate, row[new_col]))
                for _, row in to_process.iterrows()
            ]
            for r in tqdm(results): r.get()
    
    return dataset

def _add_default_rate(dataset):
    if len(dataset) == 0: raise ValueError("Dataset is empty !")
    import librosa
        
    default_rate = librosa.get_samplerate(dataset.at[0, 'filename'])
    dataset['wavs_{}'.format(default_rate)] = dataset['filename']
    
    return dataset

def _add_resampled_dirs(directory, dataset):
    resampled = [d for d in os.listdir(directory) if d.startswith('wavs_')]
    
    prefix_len = len(os.path.normpath(directory).split(os.path.sep))
    
    for d in resampled:
        key = d
        if '-' in d:
            key, _, orig_dir = d.partition('-')[2]
        else:
            orig_dir = os.path.normpath(dataset.at[0, 'filename']).split(os.path.sep)[prefix_len]
        
        if key not in dataset.columns: dataset[key] = dataset['filename']
        dataset[key] = dataset[key].apply(
            lambda f: f.replace(orig_dir, d).rpartition('.')[0] + '.wav'
        )
    return dataset

def _replace_dir(filename, new_dir, prefix):
    parts = os.path.normpath(filename).split(os.path.sep)
    index = len(os.path.normpath(prefix).split(os.path.sep))
    parts[index] = '{}-{}'.format(new_dir, parts[index])
    return os.path.join(* parts)
