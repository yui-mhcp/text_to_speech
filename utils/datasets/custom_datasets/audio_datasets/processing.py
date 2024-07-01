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
import multiprocessing
import pandas as pd

from functools import wraps
from multiprocessing import cpu_count

from loggers import timer
from .. import add_dataset, maybe_load_embedding

logger = logging.getLogger(__name__)

def audio_dataset_wrapper(name, task, ** default_config):
    def wrapper(dataset_loader):
        @timer(name = '{} loading'.format(name))
        @wraps(dataset_loader)
        def _load_and_process(directory, * args, rate = None, add_audio_time = False, ** kwargs):
            dataset = dataset_loader(directory, * args, ** kwargs)
            if isinstance(dataset, tuple): directory, dataset = dataset
            if isinstance(dataset, list): dataset = pd.DataFrame(dataset)
            
            dataset = _add_default_rate(dataset)

            if rate:
                dataset = resample_dataset(dataset, rate, ** kwargs)
            
            if 'time' not in dataset.columns:
                if add_audio_time: dataset = _add_audio_time(dataset)
                else: dataset['time'] = -1.
            
            if 'id' not in dataset.columns: dataset['id'] = name
            
            dataset = maybe_load_embedding(directory, dataset, ** kwargs)
            
            return dataset
        
        add_dataset(name, processing_fn = _load_and_process, task = task, ** default_config)
        
        return _load_and_process
    return wrapper

def _add_audio_time(dataset):
    import librosa
    
    dataset['time'] = dataset['filename'].apply(lambda f: librosa.get_duration(filename = f))
    return dataset

def _add_default_rate(dataset):
    import librosa
    if len(dataset) == 0: raise ValueError("Dataset is empty !")
        
    default_rate = librosa.get_samplerate(dataset.at[0, 'filename'])
    dataset['wavs_{}'.format(default_rate)] = dataset['filename']
    
    return dataset

def _get_processed_name(processed_dir):
    if '-' not in processed_dir: return processed_dir
    elif processed_dir.startswith('wavs_'): return processed_dir.split('-')[0]
    return processed_dir

def resample_dataset(dataset, new_rate, max_workers = cpu_count(), tqdm = None, ** _):
    from utils.audio import resample_file
    
    if tqdm is None: tqdm = lambda x: x
    
    new_col = 'wavs_{}'.format(new_rate)
    
    if new_col not in dataset.columns:
        audio_dir = os.path.commonpath(dataset['filename'].values.tolist())
        audio_dir = os.path.basename(audio_dir)

        new_dir = '{}-{}'.format(new_col, audio_dir)

        dataset[new_col] = dataset['filename'].apply(
            lambda f: os.path.splitext(f.replace(audio_dir, new_dir))[0] + '.wav'
        )
        
        for d in set(dataset[new_col].apply(os.path.dirname).values):
            os.makedirs(d, exist_ok = True)

    processed = dataset[new_col].apply(os.path.exists)
    
    to_process = dataset[~processed]
    
    logger.info("Resampling dataset to {} Hz\n  {} files already processed\n  {} files to process".format(
        new_rate, processed.sum(), len(to_process)
    ))
    
    if len(to_process):
        with multiprocessing.pool.ThreadPool(max_workers) as pool:
            results = [
                pool.apply_async(resample_file, (row['filename'], new_rate, row[new_col]))
                for _, row in to_process.iterrows()
            ]
            results = [r.get() for r in tqdm(results)]
    
    return dataset
